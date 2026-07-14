# Downstream Fine-Tuning & Evaluation Execution Plan

This document outlines the detailed architecture, execution flow, parameter mappings, and command interfaces for both training and evaluating downstream models in the GIT repository using our pretrained GFM checkpoints.

---

## 1. Complete Call Graph & Execution Flows

### A. Fine-Tuning Pipeline (`finetune.py`)
When running `finetune.py`, execution flows through the following sequential path:

```text
finetune.py (main)
    ↓  calls get_args_finetune() [argument parsing]
    ↓  resolves model paths (pt_model_path, sft_model_path, ft_model_path)
    ↓  overwrites params with config/base.yaml if --use_params is active
    ↓  calls wandb.init() [disables if --debug]
    ↓  calls run(params)
finetune.py::run
    ↓  calls get_data(params)
data/finetune_data.py::get_data
    ↓  routes based on registered datasets list
    ↓  calls single_graph(params) -> loads pt dataset and symmetrizes edge_index
finetune.py::run (continued)
    ↓  calls get_split(graph, params) -> returns train_mask, val_mask, test_mask
    ↓  instantiates GNN Encoder class
    ↓  loads pretrained encoder_20.pt weights using load_params() [if pt_data != 'na']
        ↓  calls utils/utils.py::load_params() -> model.load_state_dict(torch.load())
    ↓  instantiates TaskModel wrapping the GNN encoder & nn.Linear(hidden_dim, num_classes)
    ↓  loops over splits:
        ↓  seeds run using seed_everything(idx)
        ↓  calls get_loader() -> instantiates FallbackNeighborLoader for train/val/test
        ↓  instantiates AdamW optimizer and CosineAnnealingLR scheduler
        ↓  instantiates EarlyStopping(patience)
        ↓  loops over epochs (1 to epochs):
            ↓  calls loss = ft_node(...) [task/node.py]
                ↓  loops batches: forward pass -> classification logits -> CrossEntropy loss
                ↓  backward pass -> optimizer.step()
            ↓  calls result = evaluate(...) [task/node.py]
                ↓  computes train/val/test Accuracy using torchmetrics
            ↓  calls early_stopping(result) -> breaks loop if validation loss stops improving
    ↓  logs best run metrics to wandb and exits
```

### B. Evaluation Pipeline (`evaluate.py`)
When running `evaluate.py`, execution flows through the following sequential path:

```text
evaluate.py (main)
    ↓  parses CLI arguments (--dataset dataset_name or 'all')
    ↓  loops through target datasets:
        ↓  calls run_evaluation(dataset_name)
evaluate.py::run_evaluation
    ↓  locates checkpoint path dynamically (supporting 'finetune_model' and 'finetuned model' dirs)
    ↓  calls torch.load(..., weights_only=False) -> retrieves params, best_epoch, best_val, and model_state_dict
    ↓  calls get_data(params) -> loads processed dataset PyG Data object
    ↓  calls get_split(graph, params) -> resolves splits and retrieves test_mask
    ↓  instantiates GNN Encoder using parameters retrieved from checkpoint
    ↓  instantiates TaskModel(encoder, num_classes)
    ↓  calls model.load_state_dict(checkpoint["model_state_dict"])
    ↓  moves model to GPU if available and sets model.eval()
    ↓  checks inference mode (params["bs"] > 0):
        ├─ If bs == 0 (Full graph inference):
        │   ↓ runs model.forward(x, edge_index) over the entire graph
        │   ↓ extracts test logits using test_mask
        └─ If bs > 0 (NeighborLoader inference):
            ↓ calls get_ft_loader(graph, split, params)
            ↓ loops through test_loader -> computes logits per batch
            ↓ concatenates batch predictions -> extracts test logits using test_mask
    ↓  computes softmax probabilities (probabilities = softmax(logits, dim=1))
    ↓  computes predicted labels (argmax(dim=1))
    ↓  computes metrics: Accuracy, Macro/Weighted Precision, Recall, F1 (via sklearn)
    ↓  if num_classes == 2:
        ↓ computes ROC AUC and Average Precision (PR AUC) using probability of positive class
    ↓  creates directory evaluation/<dataset_name>/
    ↓  saves: metrics.json, metrics.csv, classification_report.txt, predictions.csv, confusion_matrix.png
    ↓  prints formatted metrics tables, classification report, and confusion matrix to console
```

---

## 2. GNN Encoder Construction

The downstream `Encoder` class is imported directly from `model/encoder.py` and is instantiated in `finetune.py` (lines 89–97) with the following parameters:
*   `input_dim`: `768` (dimension of the text features).
*   `hidden_dim`: `768`.
*   `activation`: `nn.ReLU` or `nn.LeakyReLU` (depending on `--activation` parameter).
*   `num_layers`: `2`.
*   `backbone`: `'sage'` (GraphSAGE convolution).
*   `normalize`: `'batch'` (Batch Normalization).
*   `dropout`: `0.15`.

**Comparison**: The downstream encoder construction is **100% identical** to the pretraining encoder definition, ensuring perfect layer compatibility.

---

## 3. Checkpoint Loading Analysis

The repository **already natively supports** loading pretrained GNN encoders. It uses the following logic:
*   **Search Directory**: `model/pretrain_model/lr_{pt_lr}_hidden_{hidden_dim}_layer_{num_layers}_backbone_{backbone}_fp_{pt_feat_p}_ep_{pt_edge_p}_alignreg_{pt_align_reg_lambda}_pt_data_{pt_data}/`
*   **Filename**: `encoder_{pt_epochs}.pt` (derived from the `--pt_epochs` argument).
*   **Kaggle Placement**: Place `encoder_20.pt` at:
    ```text
    /kaggle/working/GNN-Training/model/pretrain_model/lr_5e-05_hidden_768_layer_2_backbone_sage_fp_0.2_ep_0.2_alignreg_10.0_pt_data_fraud/encoder_20.pt
    ```

---

## 4. Parameter & State Dict Compatibility

*   **Encoder State Dict**: The saved `encoder_20.pt` checkpoint contains only the keys of the GNN `Encoder` class (message-passing layers, normalizations, and projection mlps).
*   **Classifier Separation**: The downstream linear classification layers (such as `TaskModel.decoder`) are created fresh in `TaskModel` and are **not** present in the checkpoint.
*   **Verdict**: Loading with `strict=True` **succeeds perfectly** because the state dict in the checkpoint maps 1-to-1 to the `encoder` instance being loaded.

---

## 5. Minimal Required Patch

No source code modifications are needed to load the weights. However, to prevent `KeyError` crashes when running fine-tuning with `--use_params` (which pulls parameters from YAML), the fraud datasets have been registered under the `node` section of **`config/base.yaml`**:
```yaml
  BUPT:
    normalize: "batch"
    sft_lr: 0.000001
    sft_epochs: 100
    lr: 0.0001
    decay: 0.000001
  Elliptic:
    normalize: "batch"
    sft_lr: 0.000001
    sft_epochs: 100
    lr: 0.0001
    decay: 0.000001
  IBM_AML:
    normalize: "batch"
    sft_lr: 0.000001
    sft_epochs: 100
    lr: 0.0001
    decay: 0.000001
```

---

## 6. Classifier Architecture & Training Strategy

*   **Classifier Architecture**: `TaskModel` wraps the GNN `encoder` and adds a linear layer mapping from `hidden_dim` (768) to the dataset's `num_classes` (4 for BUPT, 2 for Elliptic, 2 for IBM AML).
*   **Training Mode**: By default, the entire model is **fully fine-tuned (end-to-end)**. The optimizer accepts `task_model.parameters()`, meaning both the GNN encoder weights and the new classification head are updated.
*   **Freezing**: There is no active freezing parameter in `finetune.py`.

---

## 7. Optimizer, Scheduler, and Evaluation

*   **Optimizer**: AdamW with learning rate (default `1e-4`) and weight decay (default `1e-6`).
*   **Scheduler**: CosineAnnealingLR (decaying down to 0 over the training epochs).
*   **Evaluation Metrics**: Computes accuracy metrics via `torchmetrics.Accuracy(task="multiclass")` on the train, validation, and test node sets using mask slicing.
*   **Early Stopping**: Stops training if validation accuracy does not improve for `early_stop` (default `200`) epochs. The best epoch performance is recorded in memory (no checkpoints are written to disk).

---

## 8. Command Execution Interfaces

### A. Fine-Tuning Command (Kaggle)
To fine-tune BUPT using `encoder_20.pt` for 100 epochs:
```bash
python finetune.py --dataset BUPT --pt_data fraud --pt_epochs 20 --pt_lr 5e-5 --epochs 100 --early_stop 200
```

### B. Standalone Evaluation Command (`evaluate.py`)
To evaluate the fine-tuned checkpoints:
*   **For BUPT**:
    ```bash
    python evaluate.py --dataset BUPT
    ```
*   **For all three datasets sequentially**:
    ```bash
    python evaluate.py --dataset all
    ```

---

## 9. Evaluation Output Directory & Artifact Specifications

Evaluation outputs are saved inside `evaluation/<dataset_name>/` (e.g. `evaluation/BUPT/`):

1.  **`metrics.json`**:
    ```json
    {
        "dataset": "BUPT",
        "checkpoint_path": "model/finetune_model/BUPT/best_model.pt",
        "best_epoch": 100,
        "best_validation": 99.9788,
        "num_classes": 4,
        "num_test_nodes": 18857,
        "accuracy": 0.999788,
        "macro_precision": 0.9998,
        "weighted_precision": 0.9998,
        "macro_recall": 0.9997,
        "weighted_recall": 0.9998,
        "macro_f1": 0.9997,
        "weighted_f1": 0.9998,
        "roc_auc": null,
        "average_precision": null
    }
    ```
2.  **`metrics.csv`**: Single-row CSV containing exactly the same headers and values as `metrics.json`.
3.  **`classification_report.txt`**: Standard classification report text with precision, recall, and f1-score per class.
4.  **`predictions.csv`**:
    *   Columns: `node_index`, `ground_truth`, `prediction`, `probability_class_0`, `probability_class_1`, ...
    *   Contains one row per test node.
5.  **`confusion_matrix.png`**: Heatmap of the confusion matrix with count annotations inside cells and axis labels.

---

## 10. Potential Downstream Failure Points

1.  **Incorrect Checkpoint Path**:
    *   *Cause*: Mismatched pretraining hyperparameters (`--pt_lr`, `--pt_feat_p`, `--pt_edge_p`, `--pt_align_reg_lambda`) causes `finetune.py` to search the wrong pretraining subfolder.
    *   *Fix*: Ensure all `--pt_*` flags match the pretraining settings exactly.
2.  **Missing config base values**:
    *   *Cause*: Passing `--use_params` without registering defaults in `config/base.yaml` raises a `KeyError`.
    *   *Fix*: We have already registered the default parameters inside `config/base.yaml` to prevent this.
3.  **Local vs. Kaggle Data Paths**:
    *   *Cause*: Absolute paths from Kaggle stored inside `best_model.pt` parameters will cause loading errors when run locally.
    *   *Fix*: `evaluate.py` dynamically overrides `data_path` to look for a local `cache_data/` directory before resorting to the checkpoint's stored path.
