# GGFM: Graph Foundation Models for Universal Fraud Detection

GGFM (Graph Foundation Model) is a self-supervised graph neural network framework designed for universal, cross-dataset fraud detection. By combining Graph Inductive Training (GIT) with natural language entity descriptions, GGFM aligns topological transaction networks with dense semantic spaces. This allows the foundation model to generalize across diverse domains (such as telecom fraud, bank account transaction laundering, and Bitcoin illicit laundering) with minimal downstream fine-tuning.

![GGFM Overall Pipeline](assets/1_overall_ggfm_pipeline.png)

![GGFM Fraud Foundation Model Impact](assets/12_universal_fraud_foundation_model_impact.png)

---

## 🚀 Key Features

*   **Universal Graph Alignment**: Maps heterogeneous transaction networks (different node types, edge features, and dimensional shapes) into a unified 768-dimensional space.
*   **Semantic Text Embeddings**: Uses pretrained sentence transformers (`BAAI/bge-base-en-v1.5`) to encode tabular transaction metadata and node statistics into rich semantic representations.
*   **Self-Supervised Pretraining**: Pretrains a deep GraphSAGE GNN backbone using a multi-task objective combining feature reconstruction, topological reconstruction, and semantic contrastive learning (EMA targets).
*   **End-to-End Fine-Tuning**: Downstream classification training with NeighborLoader batching to prevent GPU OutOfMemory errors on large graphs.
*   **Automated Evaluation Reports**: Saves best validation checkpoints and exports complete evaluation reports (precision, recall, F1, ROC-AUC, PR-AUC, predictions logs, and confusion matrices) natively inside the output directories.

---

## 📁 Repository Structure

```text
GIT/
├── config/                  # YAML configurations (hyperparameters for BUPT, Elliptic, IBM_AML)
├── data/                    # Data loaders & PyG graph assembly (finetune_data.py, pretrain_data.py)
├── docs/                    # Detailed architectural and methodology reports
│   ├── TECHNICAL_METHODOLOGY.md           # Deep-dive system methodology & integration guide
│   ├── FINETUNING_EXECUTION_PLAN.md       # Fine-tuning execution details and call graphs
│   ├── PREPROCESSING.md                   # Raw-to-schema translation pipeline
│   ├── DATA_FORMAT.md                     # PyG Data attribute specifications
│   └── CONVERTER_DESIGN.md                # Converter architectures
├── model/                   # Core neural architectures (encoder.py, pretrain_model.py, finetune_model.py)
├── task/                    # Downstream tasks (node.py, edge.py, graph.py)
├── utils/                   # Shared scripts (args.py, utils.py, eval.py, loader.py, save_metrics.py)
├── custom_pretrain/         # Self-Supervised Graph Foundation Pretraining pipeline
├── finetune.py              # Downstream Fine-Tuning Execution Loop
└── pretrain.py              # Self-Supervised Foundation Pretraining Execution Loop
```

---

## 🛠️ Quick Start

### 1. Installation
Create the environment and install dependencies:
```bash
conda env create -f environment.yml
conda activate GGFM
```

### 2. Self-Supervised Pretraining
To pretrain the GGFM encoder on your combined fraud graphs:
```bash
python pretrain.py --pretrain_dataset fraud --epochs 20 --pt_lr 5e-5
```
This saves the pretrained encoder weights at:
`model/pretrain_model/.../encoder_20.pt`

### 3. Downstream Fine-Tuning & Evaluation
To fine-tune the model on a downstream dataset (e.g., `BUPT`) using the pretrained encoder weights:
```bash
python finetune.py --dataset BUPT --pt_data fraud --pt_epochs 20 --pt_lr 5e-5 --use_params
```
Fine-tuning will run end-to-end with validation early stopping. Once a new best validation performance is achieved, the script automatically generates complete evaluation reports at:
`model/finetune_model/BUPT/`

---

## 📊 Generated Artifacts
Every downstream fine-tuning run automatically saves the following evaluation files beside the model checkpoint:
*   `best_model.pt`: Downstream classifier model weights.
*   `metrics.json` & `metrics.csv`: Accuracy, Macro/Weighted Precision, Recall, F1, and binary ROC-AUC / Average Precision.
*   `classification_report.txt`: Standard scikit-learn text report.
*   `predictions.csv`: Log containing `node_id`, `ground_truth`, `prediction`, and class probabilities.
*   `confusion_matrix.png`: Heatmap visual of prediction counts.

---

## 📈 Downstream Benchmarks & Results

To demonstrate the cross-dataset generalization of GGFM, the model was evaluated on three major fraud detection datasets. Below are the key results, showcasing high accuracy and macro F1 scores after transferring the pretrained encoder:

### Key Metrics
*   **IBM AML (Bank Money Laundering)**:
    *   **Test Accuracy**: **99.87%**
    *   **Macro F1-Score**: **97.28%** (Macro Precision: **98.39%**, Macro Recall: **96.23%**)
    *   **ROC-AUC**: **99.86%**
    *   **PR-AUC (Average Precision)**: **97.82%**
*   **BUPT (Telecom Calling Fraud)**:
    *   **Test Accuracy**: **99.98%**
    *   **Macro F1-Score**: **99.96%** (Macro Precision: **99.96%**, Macro Recall: **99.96%**)
*   **Elliptic (Bitcoin UTXO Transactions)**:
    *   **Test Accuracy**: **94.39%**
    *   **Macro F1-Score**: **54.21%** (Baseline transfer on high-imbalance UTXO flows)

For detailed metric definitions, validation training logs, and analysis of these benchmarks, refer to the [Technical Methodology Report](docs/TECHNICAL_METHODOLOGY.md#10-experimental-results--benchmark-evaluation).

---

## 📖 Documentation Index
For advanced guides and deep dives, refer to the files in the `docs/` folder:
*   [Technical Methodology](docs/TECHNICAL_METHODOLOGY.md): Reverse-engineered overview of data preprocessing, GGFM pretraining architecture, downstream loading, and new dataset integration guide (including starting from raw CSV files).
*   [Fine-Tuning Execution Plan](docs/FINETUNING_EXECUTION_PLAN.md): Detailed call graphs, encoder parameters, checkpoint details, and command execution configurations.
*   [Preprocessing Pipeline](docs/PREPROCESSING.md): Translating tabular features and topology into semantic descriptions.
*   [Data Formats](docs/DATA_FORMAT.md): Detailed schemas for PyG Data attributes.

---

## 📦 Kaggle Replication Dataset

To facilitate rapid replication and testing, we have packaged all pre-computed graph models, language embeddings, raw datasets, and trained checkpoints into a single public Kaggle dataset. This allows researchers and judges to skip the computationally expensive steps (such as encoding millions of node sentences using SentenceTransformers or running full pretraining loops) and immediately run downstream fine-tuning.

🔗 **[Access the Kaggle Replication Dataset here](https://www.kaggle.com/datasets/your-username/your-dataset-name)** *(Please edit this link with your active dataset URL)*

### Dataset Artifact Structure
The Kaggle dataset contains the following pre-built components (totalling ~9.5 GB):
*   `datasets/` (~1.85 GB, 11 files): Raw datasets (node lists, features, and transaction edgelists for BUPT, Elliptic, and IBM AML).
*   `cache_output/` (~2.03 GB, 3 files): Serialized `FraudGraph` schema pickles compiled from raw inputs.
*   `embeddings/` (~2.61 GB, 9 files): Pre-computed 768-dimensional node and class embeddings generated via `BAAI/bge-base-en-v1.5`.
*   `cached data/` (~2.70 GB, 3 files): Unified, compiled PyTorch Geometric `geometric_data_processed.pt` files ready for GNN training.
*   `pretrained model/` (~23.7 MB, 3 files): Pretrained self-supervised universal GGFM encoder checkpoints (`encoder_20.pt`).
*   `finetuned model/` (~113 MB, 21 files): Pre-trained downstream classification model weights (`best_model.pt`) and evaluation report logs for BUPT, Elliptic, and IBM AML.

---

## 👥 Credits & Citations

This repository is built upon the **OFA (One for All)** and **GIT (Graph Inductive Training)** foundation model codebase developed by Zehong Wang et al. 

If you use this work, please credit and cite the original authors and their research paper:

```bibtex
@inproceedings{wang2024one,
  title={One for All: Towards a Universal Foundation Model for Graphs},
  author={Wang, Zehong and Shen, Yifei and Zhang, Jiacheng and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

### 🛠️ Our Augmentations
We have augmented the original GFM baseline specifically to support universal fraud detection:
*   Added custom pipeline scripts to serialize tabular fraud data (`convert_bupt.py`, `convert_ibm.py`, etc.) into GFM-compliant graphs.
*   Implemented automated in-training downstream evaluation (`utils/save_metrics.py`) exporting precise metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC), confusion matrices, and detailed predictions.
*   Fixed out-of-memory errors by mapping sequentially-ordered evaluation loaders.

