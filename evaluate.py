import os
import os.path as osp
import sys
import json
import csv
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, average_precision_score
)

# Add current workspace directory to system path
workspace_dir = osp.dirname(osp.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.append(workspace_dir)

from data.finetune_data import get_data
from utils.split import get_split
from utils.loader import get_ft_loader
from model.encoder import Encoder
from model.finetune_model import TaskModel


def get_checkpoint_path(dataset_name):
    # Search paths for robustness across local Mac and Kaggle directory names
    possible_dirs = [
        osp.join(workspace_dir, "model", "finetune_model", dataset_name),
        osp.join(workspace_dir, "model", "finetuned model", dataset_name),
        osp.join(workspace_dir, "model", "finetuned model", dataset_name.replace("_", " ")),
        osp.join(workspace_dir, "model", "finetune_model", dataset_name.replace("_", " ")),
    ]
    
    for d in possible_dirs:
        p = osp.join(d, "best_model.pt")
        if osp.exists(p):
            return p
    return None


def run_evaluation(dataset_name):
    print(f"\nEvaluating dataset: {dataset_name}")
    print("=" * 80)
    
    # 1. Locate Checkpoint
    checkpoint_path = get_checkpoint_path(dataset_name)
    if checkpoint_path is None:
        print(f"❌ Error: Checkpoint 'best_model.pt' not found for dataset '{dataset_name}'.")
        print("Please verify your model directory contains the trained weights.")
        return False

    print(f"Loading checkpoint from: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint safely for PyTorch 2.6+
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    best_epoch = checkpoint.get("epoch", "N/A")
    best_validation = checkpoint.get("best_val", "N/A")
    ckpt_params = checkpoint.get("params", {})
    
    # Ensure dataset name in params matches current run
    ckpt_params["dataset"] = dataset_name
    
    # Resolve local data_path fallback if running locally
    local_cache = osp.join(workspace_dir, "cache_data")
    if osp.exists(local_cache):
        ckpt_params["data_path"] = local_cache
    
    # 2. Load Dataset using repository code
    print("Loading dataset...")
    graph = get_data(ckpt_params)
    print(f"Dataset Loaded: {dataset_name} | Nodes: {graph.num_nodes} | Edges: {graph.num_edges} | Classes: {graph.num_classes}")
    
    # 3. Resolve splits
    splits = get_split(graph, ckpt_params)
    split = splits[0]
    test_mask = split["test"]
    num_test_nodes = int(test_mask.sum().item()) if isinstance(test_mask, torch.Tensor) else len(test_mask)
    
    # 4. Construct Encoder & TaskModel
    print("Constructing model...")
    activation_str = ckpt_params.get("activation", "relu")
    activation_cls = nn.ReLU if activation_str == "relu" else nn.LeakyReLU
    
    encoder = Encoder(
        input_dim=ckpt_params["input_dim"],
        hidden_dim=ckpt_params["hidden_dim"],
        activation=activation_cls,
        num_layers=ckpt_params["num_layers"],
        backbone=ckpt_params["backbone"],
        normalize=ckpt_params["normalize"],
        dropout=ckpt_params["dropout"]
    )
    
    model = TaskModel(encoder, num_classes=graph.num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("✓ Model successfully loaded and moved to device.")
    
    # 5. Run Inference (Supporting Full-Graph and NeighborLoader modes)
    print("Running inference...")
    y_true_list = []
    logits_list = []
    
    is_loader = ckpt_params.get("bs", 0) > 0
    
    with torch.no_grad():
        if not is_loader:
            # Full graph inference
            x = graph.node_text_feat.to(device)
            edge_index = graph.edge_index.to(device)
            z = model.encode(x, edge_index)
            z = model.pooling_lin(z)
            logits = model.classify(z)
            
            y_true = graph.y.squeeze().cpu()
            logits = logits.cpu()
        else:
            # NeighborLoader inference
            # Get loader using repository utility
            _, _, test_loader = get_ft_loader(graph, split, ckpt_params)
            
            for sg in test_loader:
                bs = sg.batch_size
                x = sg.node_text_feat.to(device)
                edge_index = sg.edge_index.to(device)
                y_batch = sg.y[:bs].squeeze().to(device)
                
                z = model.encode(x, edge_index)[:bs]
                z = model.pooling_lin(z)
                logits_batch = model.classify(z)
                
                y_true_list.append(y_batch.cpu())
                logits_list.append(logits_batch.cpu())
                
            y_true = torch.cat(y_true_list, dim=0)
            logits = torch.cat(logits_list, dim=0)
            
    # Filter predictions by test mask
    test_y_true = y_true[test_mask].numpy()
    test_logits = logits[test_mask]
    
    # Compute probabilities and predictions
    test_probs = torch.softmax(test_logits, dim=1).numpy()
    test_preds = torch.argmax(test_logits, dim=1).numpy()
    
    # 6. Compute Metrics
    accuracy = accuracy_score(test_y_true, test_preds)
    macro_precision = precision_score(test_y_true, test_preds, average='macro', zero_division=0)
    weighted_precision = precision_score(test_y_true, test_preds, average='weighted', zero_division=0)
    macro_recall = recall_score(test_y_true, test_preds, average='macro', zero_division=0)
    weighted_recall = recall_score(test_y_true, test_preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(test_y_true, test_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(test_y_true, test_preds, average='weighted', zero_division=0)
    
    # Binary metrics (ROC AUC and Average Precision)
    is_binary = graph.num_classes == 2
    roc_auc = None
    average_precision = None
    if is_binary:
        # Probability of positive class (class 1)
        pos_probs = test_probs[:, 1]
        try:
            roc_auc = roc_auc_score(test_y_true, pos_probs)
            average_precision = average_precision_score(test_y_true, pos_probs)
        except Exception as e:
            print(f"⚠️ Warning: Could not compute AUC/AP: {e}")
            
    # Get text reports
    report_str = classification_report(test_y_true, test_preds, digits=4, zero_division=0)
    cm = confusion_matrix(test_y_true, test_preds)
    
    # 7. Print Console Outputs
    print("\n" + "=" * 54)
    print(f"{'Evaluation Results':^54}")
    print("=" * 54)
    print(f"Dataset           : {dataset_name}")
    print(f"Checkpoint Path   : {checkpoint_path}")
    print(f"Best Epoch        : {best_epoch}")
    print(f"Best Validation   : {best_validation:.4f}" if isinstance(best_validation, float) else f"Best Validation   : {best_validation}")
    print("-" * 54)
    print(f"Accuracy          : {accuracy * 100:.4f}%")
    print(f"Macro Precision   : {macro_precision * 100:.4f}%")
    print(f"Weighted Precision: {weighted_precision * 100:.4f}%")
    print(f"Macro Recall      : {macro_recall * 100:.4f}%")
    print(f"Weighted Recall   : {weighted_recall * 100:.4f}%")
    print(f"Macro F1          : {macro_f1 * 100:.4f}%")
    print(f"Weighted F1       : {weighted_f1 * 100:.4f}%")
    if is_binary:
        print(f"ROC AUC           : {roc_auc * 100:.4f}%" if roc_auc is not None else "ROC AUC           : N/A")
        print(f"Average Precision : {average_precision * 100:.4f}%" if average_precision is not None else "Average Precision : N/A")
    print("=" * 54)
    print("\nClassification Report:\n", report_str)
    print("Confusion Matrix:\n", cm)
    
    # 8. Create Output Directories
    output_dir = osp.join(workspace_dir, "evaluation", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 9. Save Files
    # a. metrics.json
    metrics_dict = {
        "dataset": dataset_name,
        "checkpoint_path": checkpoint_path,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "num_classes": graph.num_classes,
        "num_test_nodes": num_test_nodes,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
        "average_precision": average_precision
    }
    
    with open(osp.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    # b. metrics.csv
    csv_headers = list(metrics_dict.keys())
    with open(osp.join(output_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerow(metrics_dict)
        
    # c. classification_report.txt
    with open(osp.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report_str)
        
    # d. predictions.csv
    # Columns: node_index, ground_truth, prediction, probability_class_0, probability_class_1...
    pred_headers = ["node_index", "ground_truth", "prediction"]
    for c in range(graph.num_classes):
        pred_headers.append(f"probability_class_{c}")
        
    # Find original graph node indices for test nodes
    test_mask_np = test_mask.cpu().numpy() if isinstance(test_mask, torch.Tensor) else np.array(test_mask)
    if test_mask_np.dtype == bool:
        test_indices = np.where(test_mask_np == True)[0]
    else:
        test_indices = test_mask_np
    
    with open(osp.join(output_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(pred_headers)
        for idx in range(len(test_indices)):
            row = [
                int(test_indices[idx]),
                int(test_y_true[idx]),
                int(test_preds[idx])
            ]
            # Append class probabilities
            row.extend([float(test_probs[idx, c]) for c in range(graph.num_classes)])
            writer.writerow(row)
            
    # e. confusion_matrix.png
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Support class labels if available
    tick_marks = np.arange(graph.num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(tick_marks)
    ax.set_yticklabels(tick_marks)
    
    ax.set_ylabel('Ground Truth')
    ax.set_xlabel('Prediction')
    ax.set_title(f'Confusion Matrix - {dataset_name}')
    
    # Annotate cell counts
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    plt.savefig(osp.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    print(f"✓ All evaluation artifacts saved inside: {output_dir}")
    print("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(description="Downstream GFM Model Evaluation Script")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["BUPT", "IBM_AML", "Elliptic", "all"],
        help="Dataset name or 'all' to evaluate BUPT, IBM_AML, and Elliptic sequentially"
    )
    args = parser.parse_args()
    
    if args.dataset == "all":
        datasets = ["BUPT", "Elliptic", "IBM_AML"]
    else:
        datasets = [args.dataset]
        
    success = True
    for d in datasets:
        status = run_evaluation(d)
        if not status:
            success = False
            
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
