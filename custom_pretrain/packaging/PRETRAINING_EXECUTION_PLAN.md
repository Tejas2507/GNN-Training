# GIT Pretraining Execution Plan for Fraud Graph Foundation Models

This document outlines the execution and training plan for pretraining the GIT model on our standard fraud datasets (IBM AML, BUPT, and Elliptic++) inside the Kaggle environment.

---

## 1. Complete Pretraining Call Graph

Starting from `pretrain.py`, execution flows through the following sequential graph:

```text
pretrain.py (__main__)
    ↓  calls get_args_pretrain() [argument parsing]
    ↓  sets params['data_path'] & params['model_path']
    ↓  calls wandb.init() [disables if --debug]
    ↓  calls run(params)
pretrain.py::run
    ↓  calls seed_everything(params["seed"])
    ↓  calls unified_data(params)
data/pretrain_data.py::unified_data
    ↓  loops over datasets in pretrain_datasets[params['pretrain_dataset']]
    ↓  calls get_data(...) -> loads geometric_data_processed.pt
    ↓  calls preprocess(data) -> x = torch.arange(num_nodes)
    ↓  calls VirtualNodeAugmentor.augment(data, task) -> adds virtual nodes/edges
    ↓  calls postprocess(data) -> keeps only ['x', 'edge_index', 'node_text_feat']
    ↓  calls preprocess_data_dict() -> shifts x and node indices for batching
    ↓  calls Batch.from_data_list() -> creates union Batch
    ↓  returns batch & train_nodes to run()
pretrain.py::run (continued)
    ↓  creates GNN Encoder, feat_decoder (MLP), and topo_decoder (InnerProductDecoder)
    ↓  creates PretrainModel(encoder, feat_decoder, topo_decoder)
    ↓  creates AdamW optimizer and lr scheduler
    ↓  loops over epochs 1 to epochs:
        ↓  calls get_pt_loader(pretrain_data, train_nodes, params) -> NeighborLoader
        ↓  calls pretrain(model, loader, optimizer, scheduler, params)
pretrain.py::pretrain (loop over NeighborLoader mini-batches)
    ↓  loops for data in loader:
        ↓  x = data.node_text_feat[data.x].to(device) [CPU-side indexing to save VRAM]
        ↓  edge_index = data.edge_index.to(device)
        ↓  views = mask_feature(x) & dropout_adj(edge_index) [computes two augmented views]
        ↓  losses = model(graph, aug_graph1, aug_graph2, ...) [PretrainModel.forward()]
            ↓  z1 = encoder(aug_g1), z2 = encoder(aug_g2)
            ↓  z1 = mean_aggr(z1), z2 = mean_aggr(z2) [GNN pooled representations]
            ↓  computes sem_loss (contrastive loss against EMA copy)
            ↓  computes feat_loss (MSE node attribute reconstruction) [if --multitask]
            ↓  computes topo_loss (BCE negative-sampling edge prediction) [if --multitask]
        ↓  optimizer.zero_grad()
        ↓  loss.backward() -> nn.utils.clip_grad_norm_() -> optimizer.step()
        ↓  wandb.log(...)
    ↓  calls pretrain_model.save_encoder(...) [saves state dict of GNN encoder]
```

---

## 2. Implemented Pretraining Objective

The `PretrainModel` implements three self-supervised reconstruction targets:

1. **Semantic Reconstruction Loss (`sem_loss`)**: Contrastive learning between two augmented views of the same node subgraph. Cosine similarity is computed between the GNN-pooled representations of both views, regularized by an EMA model's encoded representations.
2. **Feature Reconstruction Loss (`feat_loss`)**: Measures how well the latent representation `z` can reconstruct the original text embedding `x` via a Linear decoder (`feat_decoder`). Uses MSE loss.
3. **Topology Reconstruction Loss (`topo_loss`)**: Measures how well the latent representations `z` predict edge connectivity (adjacency). Positive edges (from the graph) are scored using an inner product decoder, and negative edges are sampled using `negative_sampling`. Uses binary cross-entropy loss.
4. **KL Regularization (`align_reg`)**: KL divergence between the log-softmax of the batch representations and the softmax of the batch mean representation (optional, active when `align_reg_lambda > 0`).

* **Augmentations**:
  - `mask_feature(x, p=feat_p)`: Randomly masks (zeroes out) features of the nodes with probability `feat_p`.
  - `dropout_adj(edge_index, p=edge_p)`: Randomly drops edges from the adjacency list with probability `edge_p`.
* **Positives and Negatives**:
  - Semantic loss: Positive pairs are the GNN-pooled representation of the same node under the two different augmented views (`z1` and `z2`). Cosine distance is minimized.
  - Topology loss: Positive pairs are the existing edges in the subgraph. Negative pairs are sampled using PyG's `negative_sampling` (which randomly samples node pairs that do not have an edge in the subgraph).
* **Combination of Losses**:
  `loss = feat_loss * feat_lambda + topo_loss * topo_lambda + sem_loss * sem_lambda + align_reg`
* **Representations Learned**: Node-level embeddings that capture both local semantic text attributes (via `node_text_feat`), neighborhood structure (via SAGEConv message passing), and global class alignment.
* **Tensors Consumed**:
  - Only `data.x` (indices), `data.edge_index` (edges), and `data.node_text_feat` (embeddings) are actually consumed. All other tensors (labels, splits, class embeddings) are set to `None` in `postprocess(data)` before training starts.

---

## 3. Configuration Hyperparameter Table

| Hyperparameter | Default Value | Recommended for Fraud pretraining | Action (Keep / Modify) & Rationale |
|---|---|---|---|
| `lr` | `1e-7` | `5e-5` | **Modify**: `1e-7` is too small for training decoders and GNN representations from scratch; `5e-5` / `1e-4` is standard. |
| `decay` | `1e-8` | `1e-5` | **Modify**: Standard weight decay to prevent model overfitting. |
| `bs` | `4096` | `2048` | **Modify**: Keeps memory consumption inside Kaggle Tesla T4 (16GB VRAM) limits while preserving stable gradient updates. |
| `hidden_dim` | `768` | `768` | **Keep**: Keeps GNN hidden representations at the same dimension as BGE embeddings. |
| `num_layers` | `2` | `2` | **Keep**: 2 layers is optimal to capture two-hop neighborhoods without over-smoothing. |
| `backbone` | `'sage'` | `'sage'` | **Keep**: GraphSAGE is stable and highly efficient. |
| `fanout` | `10` | `10` | **Keep**: Standard neighborhood size (samples up to 100 neighbors). |
| `feat_p` | `0.2` | `0.2` | **Keep**: 20% masking is optimal for semantic reconstruction. |
| `edge_p` | `0.2` | `0.2` | **Keep**: 20% dropout is optimal for topology reconstruction. |
| `align_reg_lambda`| `10.0` | `10.0` | **Keep**: Regularizes representation distribution alignment. |
| `multitask` | `False` | `True` | **Modify**: MUST set to `True` to enable feature and topology reconstruction (essential for graph pretraining). |
| `epochs` | `20` | `20` | **Keep**: 20 epochs is sufficient for pretraining convergence. |
| `train_ratio` | `1.0` | `1.0` | **Keep**: Train on all available graph nodes. |

---

## 4. Recommended Configuration for BUPT (Dry-run / Verification)

Use this configuration to verify that the loader, GNN, and optimizer initialize and train for one epoch:
```bash
python pretrain.py --pretrain_dataset BUPT --epochs 1 --bs 1024 --lr 5e-5 --multitask --debug
```

---

## 5. Recommended Configuration for Joint Dataset Pretraining (Full Run)

Use this configuration to train the unified Fraud Graph Foundation Model on all three datasets:
```bash
python pretrain.py --pretrain_dataset fraud --epochs 20 --bs 2048 --lr 5e-5 --multitask
```

---

## 6. GPU & RAM Estimates

Because GNN neighborhood sampling is done on the CPU and **only mini-batch node feature slices** are sent to the GPU, memory scaling is highly efficient:

* **BUPT**:
  * **GPU VRAM**: `~1.5 - 2.0 GB`
  * **CPU RAM**: `~2 - 3 GB`
  * **Disk usage**: `~1.5 GB`
* **Elliptic**:
  * **GPU VRAM**: `~1.5 - 2.0 GB`
  * **CPU RAM**: `~2 - 3 GB`
  * **Disk usage**: `~2.0 GB`
* **IBM_AML**:
  * **GPU VRAM**: `~2.5 - 3.0 GB`
  * **CPU RAM**: `~25 - 30 GB` (requires holding the 15.8 GB node embedding matrix in memory)
  * **Disk usage**: `~4.5 GB`

* **Are two T4 GPUs sufficient?** **Yes.** A single T4 (16GB VRAM) is more than sufficient.
* **Is CPU RAM sufficient on Kaggle?** **Yes.** Kaggle provides 30GB CPU RAM, which is enough to hold the IBM AML embeddings.

---

## 7. Expected Runtime on Kaggle (Tesla T4)

* **BUPT (1 epoch)**: `~30 seconds` | **Full Pretraining (20 epochs)**: `~10 minutes`
* **Elliptic (1 epoch)**: `~45 seconds` | **Full Pretraining (20 epochs)**: `~15 minutes`
* **IBM_AML (1 epoch)**: `~15 minutes` | **Full Pretraining (20 epochs)**: `~5 hours`
* **Joint (fraud) (20 epochs)**: `~5.5 hours`

---

## 8. Potential Failure Points & How to Debug

1. **Out-of-Memory (OOM) on CPU**: 
   * *Cause*: Loading BUPT, Elliptic, and IBM AML graphs simultaneously inside `unified_data` exceeds 30GB CPU RAM.
   * *Fix*: Reduce dataset count (e.g. pretrain BUPT and Elliptic first, or train IBM AML independently).
2. **PyTorch Geometric Custom Class Deserialization Error**:
   * *Cause*: PyTorch 2.4+ secure unpickling defaults to `weights_only=True` which blocks PyG custom classes.
   * *Fix*: Our applied patch (`weights_only=False` in `data/finetune_data.py`) prevents this.
3. **WandB Prompts blocking script**:
   * *Cause*: WandB prompts for login/API key during background run.
   * *Fix*: Pass `--debug` to disable online logging.
