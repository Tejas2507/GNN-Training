# Fraud Graph Foundation Model (GFM) Preprocessing & Packaging Summary

This document provides a comprehensive end-to-end summary of the engineering work completed to convert multiple heterogeneous fraud datasets into a unified foundation model representation and package them for pretraining using the ICML 2025 **GIT** (Towards Graph Foundation Models) framework.

---

## 1. Unified FraudGraph Representation

To resolve structural discrepancies across the three target datasets, we established a standardized `FraudGraph` schema under `custom_pretrain/schema/graph_schema.py`:
*   **Nodes**: Unified representation supporting arbitrary attributes, integer fraud labels, and natural-language text descriptions.
*   **Edges**: Directed transactions containing weights, timestamps, and attributes.
*   **Graph Converters**:
    *   `convert_ibm.py`: Processes the IBM AML multi-agent transaction flow.
    *   `convert_bupt.py`: Resolves mobile/telecommunication fraud logs.
    *   `convert_elliptic.py`: Standardizes Elliptic++ Bitcoin wallet transaction graphs.

---

## 2. Natural-Language Generation & Text Normalization

We converted raw feature vectors into descriptive, semantic sentences optimized for SentenceTransformers:
1.  **Generation Stage**: Created template-based generators for nodes, edges, and fraud classes under `custom_pretrain/text_generation/`.
2.  **Normalization Stage** (`normalize_text.py`): Compacted raw details to prevent truncation by dense encoders.
    *   *IBM AML*: Normalized bank accounts and transaction structures (~60 chars).
    *   *BUPT*: Compacted mobile logs (~320 chars).
    *   *Elliptic++*: Compressed excessively long raw transaction vectors (~1665 chars) into dense semantic summaries.

---

## 3. Embedding Generation

Using the normalized text descriptions, we generated dense vector embeddings using **`BAAI/bge-base-en-v1.5`** (768-dimension):
*   **Node Embeddings**: Generated offline using a multi-GPU Kaggle script (`multi_gpu_embed.py`) and saved as `node_embeddings.npy` and `node_ids.json`.
*   **Class Embeddings**: Handled via `prepare_class_embeddings.py` to map category text descriptions into 768-dim class representations.
*   **Edge Embeddings**: Supported via `prepare_edge_embeddings.py` (with a `--dummy` zeros utility for fast verification).

---

## 4. Universal GFM Dataset Packager

We implemented a dataset compiler under `custom_pretrain/packaging/` to package variables into compliant PyTorch Geometric formats:
*   **`split_generator.py`**: Splits labeled nodes into `70% train`, `15% val`, and `15% test` masks. Chronological splitting is selected if node timesteps are present (e.g. Elliptic++), falling back to seeded random splits (`seed=42`).
*   **`universal_converter.py`**: Maps topological adjacency indices to match the order of `node_ids.json`. Saves the resulting PyG `Data` object inside a python list of length 1: `torch.save([data], path)`.
*   **`verify_processed.py`**: Asserts tensor shapes, dtypes, range bounds, absence of NaNs, and checks that masks do not overlap.

---

## 5. GIT Repository Compatibility Patches

We audited the original GIT codebase and applied the following minimal compatibility patches:
1.  **PyTorch 2.6+ secure loading patch**: Added `weights_only=False` in `torch.load` calls in `data/finetune_data.py` to allow loading custom GNN model/PyG classes.
2.  **Dataset Registry registration**: Registered `BUPT`, `IBM_AML`, and `Elliptic` in the `citation_datasets` list in `data/finetune_data.py` so they map automatically to node classification workflows.
3.  **PyG 2.8+ propagate compatibility patch**: Patched `model/encoder.py` (MySAGEConv) to omit `edge_attr` in `self.propagate` calls when it is `None` (resolving a PyG 2.8 runtime error).

---

## 6. Pretraining Execution Plan & Instrumentation

We mapped out the execution flow of the self-supervised pretraining phase and enhanced it with telemetry:
*   **Objective**: Optimizes Semantic Contrastive Loss (using views augmented via `mask_feature` and `dropout_adj`), MSE Feature Reconstruction Loss, and BCE Topology Reconstruction Loss (with negative edge sampling).
*   **Memory Efficiency**: Graph sampling is done on the CPU and only batch slices are sent to the GPU, allowing the 15.8 GB IBM AML dataset to be pretrained on a single 16GB Tesla T4 GPU.
*   **Patched `pretrain.py`**:
    *   Added timing metrics for loader construction, epoch times, and first-batch latency.
    *   Tracks forward and backward pass speeds.
    *   Plots running loss metrics inside rich `tqdm` progress bars.
    *   Automatically records training metrics per epoch into a persistent `training_log.csv` file.

---

## 7. Kaggle Execution Guide

To run pretraining on Kaggle:

1.  **Install compatible wheels** (bypasses `pyg-lib`/`torch-sparse` errors):
    ```python
    # Dynamic wheel installer (paste into a Kaggle cell)
    import torch, sys, subprocess
    torch_ver = torch.__version__.split('+')[0]
    cuda_ver = torch.version.cuda
    cuda_str = "cu" + cuda_ver.replace(".", "") if cuda_ver else "cpu"
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_str}.html"
    packages = ["pyg-lib", "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"]
    subprocess.run([sys.executable, "-m", "pip", "install"] + packages + ["-f", wheel_url, "-q"])
    ```
2.  **Run conversion & verification**:
    ```bash
    python custom_pretrain/packaging/prepare_class_embeddings.py --dataset all
    python custom_pretrain/packaging/prepare_edge_embeddings.py --dataset all --dummy
    python custom_pretrain/packaging/universal_converter.py --dataset BUPT
    python custom_pretrain/packaging/universal_converter.py --dataset Elliptic
    python custom_pretrain/packaging/universal_converter.py --dataset IBM_AML
    python custom_pretrain/packaging/verify_processed.py --dataset BUPT
    ```
3.  **Start GFM pretraining**:
    ```bash
    python pretrain.py --pretrain_dataset fraud --epochs 20 --bs 2048 --lr 5e-5 --multitask
    ```
