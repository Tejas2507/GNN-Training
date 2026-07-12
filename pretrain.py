from pathlib import Path
import os
import os.path as osp
import random
import time
import csv
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

from torch.optim import AdamW
from torch_geometric.utils import negative_sampling, mask_feature, dropout_adj
from torch_geometric.loader import NeighborLoader

from data.pretrain_data import unified_data
from model.encoder import Encoder, InnerProductDecoder
from model.pretrain_model import PretrainModel
from utils.utils import seed_everything, get_scheduler, get_device_from_model, check_path
from utils.args import get_args_pretrain
from utils.loader import get_pt_loader

import wandb

get_loader = get_pt_loader


def pretrain(model, loader, optimizer, scheduler=None, **kwargs):
    model.train()
    device = get_device_from_model(model)
    params = kwargs['params']
    epoch = kwargs['epoch']

    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch}/{params['epochs']}",
        dynamic_ncols=True,
    )

    print("Waiting for first NeighborLoader batch...")

    epoch_start = time.time()

    running_loss = 0.0
    running_feat = 0.0
    running_topo = 0.0
    running_sem = 0.0
    running_align = 0.0

    for batch_idx, data in enumerate(pbar, 1):

        if batch_idx == 1:
            print("✓ First batch received")
        bs = data.batch_size

        x = data.node_text_feat[data.x].to(device)
        edge_index = data.edge_index.to(device)
        graph = [x, edge_index]

        x1, _ = mask_feature(x, p=params["feat_p"])
        edge_index1, _ = dropout_adj(edge_index, p=params["edge_p"], force_undirected=True, num_nodes=x.size(0))
        aug_graph1 = [x1, edge_index1]

        x2, _ = mask_feature(x, p=params["feat_p"])
        edge_index2, _ = dropout_adj(edge_index, p=params["edge_p"], force_undirected=True, num_nodes=x.size(0))
        aug_graph2 = [x2, edge_index2]

        fw = time.time()

        losses = model(graph, aug_graph1, aug_graph2, bs=bs, params=params)

        fw = time.time() - fw
        loss = losses['loss']

        bw = time.time()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bw = time.time() - bw

        running_loss += loss.item()
        running_feat += losses["feat_loss"].item()
        running_topo += losses["topo_loss"].item()
        running_sem += losses["sem_loss"].item()
        running_align += losses["align_reg"].item()

        pbar.set_postfix(
            batch=f"{batch_idx}/{len(loader)}",
            loss=f"{running_loss/batch_idx:.4f}",
            feat=f"{running_feat/batch_idx:.3f}",
            topo=f"{running_topo/batch_idx:.3f}",
            sem=f"{running_sem/batch_idx:.3f}",
        )

        if scheduler:
            scheduler.step()

        wandb.log(
            {
                "loss/feat_loss": losses["feat_loss"].item(),
                "loss/topo_loss": losses["topo_loss"].item(),
                "loss/sem_loss": losses["sem_loss"].item(),
                "loss/align_reg": losses["align_reg"].item(),
                "loss/loss": loss.item(),
            }
        )

    return {
        "loss": running_loss / len(loader),
        "feat": running_feat / len(loader),
        "topo": running_topo / len(loader),
        "sem": running_sem / len(loader),
        "align": running_align / len(loader),
    }


def run(params):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(params["seed"])
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU

    pretrain_data, task_node_idx_dict = unified_data(params)
    train_nodes = torch.concat(list(task_node_idx_dict.values()))
    if params['train_ratio'] != 1:
        train_nodes = torch.tensor(random.sample(train_nodes.tolist(), int(len(train_nodes) * params['train_ratio'])))
    print("Number of training nodes is {}".format(len(train_nodes)))

    encoder = Encoder(
        input_dim=params["input_dim"], hidden_dim=params["hidden_dim"], activation=params["activation"],
        num_layers=params["num_layers"], backbone=params["backbone"], normalize=params["normalize"],
        dropout=params["dropout"]
    )
    feat_decoder = nn.Linear(params["hidden_dim"], params["input_dim"])
    topo_decoder = InnerProductDecoder(hidden_dim=params["hidden_dim"], output_dim=params["hidden_dim"])
    pretrain_model = PretrainModel(encoder=encoder, feat_decoder=feat_decoder, topo_decoder=topo_decoder, ).to(device)

    optimizer = AdamW(pretrain_model.parameters(), lr=params["lr"], weight_decay=params["decay"])
    scheduler = get_scheduler(optimizer, params["use_schedular"], params["epochs"])

    for i in range(1, params["epochs"] + 1):

        print("\n" + "="*80)
        print(f"Epoch {i}/{params['epochs']}")
        print("="*80)

        t_loader = time.time()
        loader = get_loader(pretrain_data, train_nodes, params)
        print(f"✓ NeighborLoader built in {time.time()-t_loader:.2f} sec")
        print(f"✓ Mini-batches : {len(loader)}")

        epoch_start = time.time()

        stats = pretrain(
            model=pretrain_model,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            params=params,
            epoch=i,
        )

        # Save model
        template = "lr_{}_hidden_{}_layer_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}"
        if params['train_ratio'] != 1:
            template += "_{}".format(params['train_ratio'])

        save_path = osp.join(params['model_path'], template.format(
            params["lr"], params["hidden_dim"], params['num_layers'], params["backbone"],
            params["feat_p"], params["edge_p"], params["align_reg_lambda"], params["pretrain_dataset"]))
        check_path(save_path)

        pretrain_model.save_encoder(osp.join(save_path, f"encoder_{i}.pt"))
        epoch_time = time.time() - epoch_start

        avg_loss = stats["loss"]
        avg_feat = stats["feat"]
        avg_topo = stats["topo"]
        avg_sem = stats["sem"]
        avg_align = stats["align"]

        print("\n" + "="*80)
        print(f"Epoch {i}/{params['epochs']} Summary")
        print("="*80)
        print(f"Loss      : {avg_loss:.4f}")
        print(f"Feature   : {avg_feat:.4f}")
        print(f"Topology  : {avg_topo:.4f}")
        print(f"Semantic  : {avg_sem:.4f}")
        print(f"Align     : {avg_align:.4f}")
        print(f"Time      : {epoch_time:.2f} sec")
        print(f"Throughput: {len(loader)/epoch_time:.2f} batches/sec")
        print("="*80)

        log_file = Path(params['model_path']) / "training_log.csv"

        write_header = not log_file.exists()

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "epoch",
                    "loss",
                    "feat_loss",
                    "topo_loss",
                    "sem_loss",
                    "align_loss",
                    "time_sec"
                ])

            writer.writerow([
                i,
                avg_loss,
                avg_feat,
                avg_topo,
                avg_sem,
                avg_align,
                epoch_time
            ])

        print("Save the model at epoch {}".format(i))

    wandb.finish()


if __name__ == "__main__":
    params = get_args_pretrain()
    params['data_path'] = osp.join(os.path.dirname(__file__), 'cache_data')
    params['model_path'] = osp.join(os.path.dirname(__file__), 'model', 'pretrain_model')

    wandb.init(
        project="GIT-Pretrain",
        name="LR:{} | Layers:{} | Fan:{}".format(params["lr"], params["num_layers"], params["fanout"]),
        mode="disabled" if params["debug"] else "online",
        group=params['group'],
        config=params,
    )

    run(params)
