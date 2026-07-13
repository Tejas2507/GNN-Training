import os
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, average_precision_score
)

def save_metrics(y_true, logits, test_mask, save_dir, dataset_name, epoch, best_val):
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Move to CPU and numpy
    y_true_cpu = y_true.detach().cpu().numpy()
    logits_cpu = logits.detach().cpu().numpy()
    
    # Ensure test_mask is boolean mask/array
    test_mask_np = test_mask.detach().cpu().numpy() if isinstance(test_mask, torch.Tensor) else np.array(test_mask)
    
    # Filter test split targets
    test_y_true = y_true_cpu[test_mask_np]
    test_logits = logits_cpu[test_mask_np]
    
    # 2. Compute Probabilities and Predictions
    test_logits_tensor = torch.tensor(test_logits)
    test_probs = F.softmax(test_logits_tensor, dim=1).numpy()
    test_preds = np.argmax(test_logits, axis=1)
    
    num_classes = test_logits.shape[1]
    num_test_nodes = len(test_y_true)
    
    # 3. Compute Classification Metrics
    accuracy = accuracy_score(test_y_true, test_preds)
    macro_precision = precision_score(test_y_true, test_preds, average='macro', zero_division=0)
    weighted_precision = precision_score(test_y_true, test_preds, average='weighted', zero_division=0)
    macro_recall = recall_score(test_y_true, test_preds, average='macro', zero_division=0)
    weighted_recall = recall_score(test_y_true, test_preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(test_y_true, test_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(test_y_true, test_preds, average='weighted', zero_division=0)
    
    is_binary = (num_classes == 2)
    roc_auc = None
    average_precision = None
    if is_binary:
        # Probability of positive class (class 1)
        pos_probs = test_probs[:, 1]
        try:
            roc_auc = roc_auc_score(test_y_true, pos_probs)
            average_precision = average_precision_score(test_y_true, pos_probs)
        except Exception as e:
            print(f"⚠️ Warning: Could not compute ROC-AUC or PR-AUC: {e}")
            
    # Reports and matrices
    report_str = classification_report(test_y_true, test_preds, digits=4, zero_division=0)
    cm = confusion_matrix(test_y_true, test_preds)
    
    # 4. Save metrics.json
    metrics_dict = {
        "dataset": dataset_name,
        "best_epoch": int(epoch),
        "best_validation": float(best_val) if isinstance(best_val, (float, np.float32, np.float64)) else best_val,
        "num_classes": int(num_classes),
        "num_test_nodes": int(num_test_nodes),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "weighted_precision": float(weighted_precision),
        "macro_recall": float(macro_recall),
        "weighted_recall": float(weighted_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "average_precision": float(average_precision) if average_precision is not None else None
    }
    
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    # 5. Save metrics.csv
    csv_headers = list(metrics_dict.keys())
    with open(os.path.join(save_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerow(metrics_dict)
        
    # 6. Save classification_report.txt
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report_str)
        
    # 7. Save predictions.csv
    pred_headers = ["node_id", "ground_truth", "prediction"]
    for c in range(num_classes):
        pred_headers.append(f"prob_class{c}")
        
    # Resolve node indices inside original graph for test nodes
    if isinstance(test_mask, torch.Tensor):
        test_indices = torch.where(test_mask)[0].cpu().numpy()
    else:
        test_indices = np.where(np.array(test_mask) == True)[0]
        
    with open(os.path.join(save_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(pred_headers)
        for idx in range(len(test_indices)):
            row = [
                int(test_indices[idx]),
                int(test_y_true[idx]),
                int(test_preds[idx])
            ]
            row.extend([float(test_probs[idx, c]) for c in range(num_classes)])
            writer.writerow(row)
            
    # 8. Save confusion_matrix.png
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    tick_marks = np.arange(num_classes)
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
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    print(f"✓ Saved evaluation metrics and reports to: {save_dir}")
