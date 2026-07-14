# GGFM: Graph Foundation Models for Universal Fraud Detection — Technical Methodology and Engineering Specifications

---

## 1. Repository Layout & Architecture

The repository is organized modularly, separating data pipeline interfaces, core GNN models, tasks, utilities, configurations, and a custom self-supervised pretraining pipeline:

```text
GIT/
├── config/                  # YAML configurations (e.g. base.yaml, zero_shot.yaml)
├── data/                    # Data loaders & graph assembly (finetune_data.py, pretrain_data.py)
├── model/                   # Core neural architectures (encoder.py, pretrain_model.py, finetune_model.py)
├── task/                    # Task training and evaluation scripts (node.py, edge.py, graph.py)
├── utils/                   # Shared scripts (args.py, utils.py, eval.py, loader.py, save_metrics.py)
├── custom_pretrain/         # Self-Supervised Graph Foundation Pretraining pipeline
│   ├── schema/              # Universal graph schema definition (graph_schema.py)
│   ├── converters/          # Dataset-specific parsers to FraudGraph (convert_bupt.py, convert_ibm.py, etc.)
│   ├── text_generation/     # Natural language describers (node_describer.py, class_describer.py)
│   └── packaging/           # Embedding alignment and PyG packaging (universal_converter.py, split_generator.py)
├── finetune.py              # Downstream Fine-Tuning Execution Loop
├── pretrain.py              # Self-Supervised Foundation Pretraining Execution Loop
├── TECHNICAL_METHODOLOGY.md # (This document)
└── FINETUNING_EXECUTION_PLAN.md
```

---

## 2. Universal Data Preprocessing & Graph Compilation

To ingest heterogeneous fraud datasets (BUPT, Elliptic, IBM AML) under a single Graph Foundation Model (GFM) architecture, a unified graph serialization and text embedding pipeline is executed.

```
+-------------+
|   Raw CSV   |
|  / EdgeList |
+------+------+
       | (converters/)
       v
+------+------+     +-------------+
|  FraudGraph | +-->+  Text Cache | (text_generation/)
|   Pickle    | |   +------+------+
+------+------+ |          | (SentenceTransformers)
       |        |          v
       |        |   +------+------+
       |        |   | Embeddings  | (node/class/edge.npy)
       |        |   +------+------+
       |        |          |
       v        v          v
+------+--------+----------+------+
|     universal_converter.py      |
+-----------------+---------------+
                  |
                  v
+-----------------+---------------+
|   geometric_data_processed.pt   |
+---------------------------------+
```

### A. Graph Extraction & Serialization (`custom_pretrain/schema/`)
Raw dataset elements are structured into a universal object model using a standardized schema. This schema ensures a dataset-agnostic interface for representing nodes, edges, labels, features, and degree configurations.
*   **`Node`**: Represents a physical entity (e.g., phone number, transaction, bank account). Contains fields: `id` (str), `type` (str), `features` (dict of numerical properties), `text` (str), and `label` (int/None).
*   **`Edge`**: Represents a connection between entities (e.g., call, transaction, money transfer). Contains fields: `src` (str), `dst` (str), `edge_type` (str), `weight` (float), `timestamp` (str/None), and `features` (dict).
*   **`FraudGraph`**: A collection of nodes and edges. It provides methods for adding elements, fetching degrees, checking node existence, and validation.

### B. Dataset-Specific Conversions to schema (`custom_pretrain/converters/`)
Each dataset has distinct structures that must be mapped to the `FraudGraph` format:
*   **BUPT (`convert_bupt.py`)**: Loads `TF.features` (space-separated tabular node features), `TF.labels` (node labels), and `TF.edgelist` (adjacency). Since phone numbers do not have intrinsic semantic attributes, the numerical values are parsed directly. In-degree and out-degree are calculated dynamically and appended to `features`.
*   **Elliptic (`convert_elliptic.py`)**: Parses standard transaction features (representing local and neighborhood statistics of Bitcoin transactions) and maps them into node features. Edges represent transaction flows (directed input-to-output addresses).
*   **IBM AML (`convert_ibm.py`)**: Parses transaction edge records (containing timestamps, transaction types, source/destination bank accounts, amounts, and fraud labels). Because bank account transactions are edge-centric, bank account node features (lifetime, total amount transferred, degree stats) are generated programmatically and mapped to Node instances, while transactions are stored as Edge instances.

### C. Semantic Text Generation (`custom_pretrain/text_generation/`)
GFM aligns tabular metrics with natural language sentences, allowing a text encoder to extract generalizable semantic embeddings.
*   **Node Descriptions (`node_describer.py`)**:
    *   *IBM AML*: Compiles account degree statistics, lifetime, transaction counts, total transferred amounts, and known fraud ratios into a descriptive biography (e.g., `"Bank account AC123. Incoming transactions: 5. Outgoing transactions: 2. Total transferred: $10500.00. Account lifetime: 120.4 days."`).
    *   *BUPT & Elliptic*: Since raw features are anonymized, descriptions are compiled using the numerical feature index alongside communication counts (e.g., `"Phone number node. Incoming communications: 12. Outgoing communications: 4. feature_0 = -0.1542. feature_1 = 0.8412."`).
*   **Class Descriptions (`class_describer.py`)**: Generates language definitions for class labels (e.g., `"Fraudulent account/phone number"` vs. `"Normal/benign bank account"`). These are encoded separately to build class prototype embeddings (`class_node_text_feat`).

### D. Dense Node Embedding Generation (`custom_pretrain/multi_gpu_embed.py`)
Once text descriptions are generated, they are encoded into 768-dimensional continuous vector embeddings:
*   **Model**: Pretrained SentenceTransformer (`BAAI/bge-base-en-v1.5`) mapped onto GPU memory.
*   **Processing**: Short descriptions are tokenized with `max_seq_length = 64`. Embeddings are computed in half-precision (`half()`) and written to a memory-mapped NumPy file (`node_embeddings.npy`) along with a sorted list of node IDs (`node_ids.json`).
*   **Class and Edge Embeddings**: `prepare_class_embeddings.py` and `prepare_edge_embeddings.py` use the same language model to compile class and edge text descriptions into matrices (`class_embeddings.npy` and `edge_embeddings.npy`).

### E. Packaging and Serialization (`universal_converter.py`)
Finally, the graph structure, split masks, and node/class embeddings are packaged into a unified PyTorch Geometric `Data` container:
1.  **Alignment**: Maps the node identifiers from `node_ids.json` to a dense index space `[0, num_nodes - 1]`.
2.  **Adjacency Mapping**: Translates raw edges from `FraudGraph.edges` into a PyG `edge_index` Tensor of shape `[2, num_edges]` using the dense indexes.
3.  **Tensors Compilation**:
    *   `x`: Node indices tensor (`torch.arange(num_nodes)`).
    *   `node_text_feat`: The 768-dimensional node embeddings tensor.
    *   `class_node_text_feat`: The class label text embeddings matrix.
    *   `y`: Node label indices tensor. Unlabeled nodes are assigned `-1`.
4.  **Split Masks Generation (`split_generator.py`)**: Partitions nodes into `train_mask`, `val_mask`, and `test_mask` using deterministic hashing of node IDs.
5.  **Output**: Serializes the `Data` instance inside a python list `[data]` using `torch.save()` into `geometric_data_processed.pt` at `cache_data/<dataset>/processed/`.

---

## 3. Self-Supervised Pretraining Pipeline

The GFM pretraining pipeline (`pretrain.py`) utilizes a self-supervised Graph Contrastive Learning objective to train a robust GNN backbone without downstream labels.

### A. GNN Encoder Architecture (`model/encoder.py`)
The backbone encoder is a deep GraphSAGE model:
*   **Convolutions**: SUCCESSIVE message-passing layers using `MySAGEConv` (custom implementation extending PyG `MessagePassing`).
    
    $$\mathbf{h}_i^{(l+1)} = \mathbf{W}_1 \cdot \text{AGGREGATE}\left(\{\mathbf{h}_j^{(l)}, \forall j \in \mathcal{N}(i)\}\right) + \mathbf{W}_2 \cdot \mathbf{h}_i^{(l)}$$
    
    where the aggregate operation is a parametric mean of neighbor representations.
*   **Regularization**: After each message passing step, representations undergo `ReLU` (or `LeakyReLU`) activation, Batch Normalization, and Dropout.
*   **Pooling**: Global pooling is handled via mean neighborhood pooling (`NonParamPooling`) followed by a linear projection layer (`self.pooling_lin`).

### B. Graph Augmentations
For each minibatch, GGFM constructs two augmented views ($\mathbf{g}_1$ and $\mathbf{g}_2$) of the same subgraph:
*   **Feature Masking (`mask_feature`)**: Randomly masks elements in the node feature matrix with a drop probability `feat_p = 0.2`.
*   **Edge Perturbation (`dropout_adj`)**: Randomly drops connections from the adjacency matrix with a probability `edge_p = 0.2`.

### C. Pretraining Objectives (`model/pretrain_model.py`)
The pretraining loss function is a multi-task objective combining structure, semantics, and feature alignment:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{feat}} \mathcal{L}_{\text{feat}} + \lambda_{\text{topo}} \mathcal{L}_{\text{topo}} + \lambda_{\text{sem}} \mathcal{L}_{\text{sem}} + \lambda_{\text{align}} \mathcal{L}_{\text{align}}$$

1.  **Feature Reconstruction Loss ($\mathcal{L}_{\text{feat}}$)**: Passes representations through an MLP feature decoder and computes Mean Squared Error (MSE) against the original text embeddings.
2.  **Topological Reconstruction Loss ($\mathcal{L}_{\text{topo}}$)**: Uses an inner-product decoder to reconstruct adjacency. It optimizes binary cross-entropy on positive edges and negative sampled edges:
    
    $$\mathcal{L}_{\text{topo}} = -\log(\sigma(\mathbf{z}_u^\top \mathbf{z}_v)) - \log(1 - \sigma(\mathbf{z}_u^\top \mathbf{z}_w))$$
    
    where $(u, v)$ is a positive edge and $(u, w)$ is a negative sample.
3.  **Semantic Contrastive Loss ($\mathcal{L}_{\text{sem}}$)**: Employs an Exponential Moving Average (EMA) update on a target semantic encoder. The contrastive loss aligns the online encoder's projections ($\mathbf{h}_1$, $\mathbf{h}_2$) with the EMA encoder's outputs ($\mathbf{z}_1$, $\mathbf{z}_2$):
    
    $$\mathcal{L}_{\text{sem}} = \frac{1}{2} \left[ (1 - \text{cosine\_sim}(\mathbf{z}_1, \mathbf{h}_2)) + (1 - \text{cosine\_sim}(\mathbf{z}_2, \mathbf{h}_1)) \right]$$
    
4.  **Alignment Regularization Loss ($\mathcal{L}_{\text{align}}$)**: Computes KL Divergence between the batch-level average embedding distribution and a target uniform distribution to prevent representation collapse.

### D. Pretraining Optimization & Checkpoints
*   **Optimizer**: AdamW with learning rate `5e-5` and weight decay `1e-6`.
*   **Checkpoint**: Trained for 20 epochs. The GNN encoder weights are extracted and saved at `model/pretrain_model/.../encoder_20.pt`.

---

## 4. Downstream Fine-Tuning Pipeline

Downstream fine-tuning (`finetune.py`) loads the GGFM encoder weights and constructs a TaskModel for classification.

### A. Model Initialization
1.  **Pretrained Encoder Loading**: Initializes `Encoder` using the parameters from `encoder_20.pt`. The weights are loaded using `strict=True` to guarantee a 1-to-1 match.
2.  **Classifier Head Construction**: Instantiates a downstream model wrapper (`TaskModel`) combining the GNN encoder and a linear decoder:
    
    $$\mathbf{z}_i = \text{Encoder}(\mathbf{x}_i, \mathbf{A})$$
    
    $$\mathbf{z}_i' = \text{pooling\_lin}(\text{mean\_aggregate}(\mathbf{z}_i))$$
    
    $$\text{logits}_i = \mathbf{W}_{\text{clf}} \cdot \mathbf{z}_i' + \mathbf{b}$$
    
    where $\mathbf{W}_{\text{clf}}$ projects from `hidden_dim` (768) to the target dataset's class dimension.

### B. Minibatch Sampling & NeighborLoader
For large-scale fraud graphs (such as BUPT, Elliptic, and IBM AML), full-graph message passing is intractable. The pipeline uses `FallbackNeighborLoader` (wrapping PyG's `NeighborLoader`) to construct subgraphs:
*   **Training Batches**: Seeded by the training indices, sampling up to 10 neighbors per node across 2 layers. Batches are shuffled.
*   **Evaluation Batches**: Sampled sequentially across the entire graph to generate predictions for all nodes.

### C. Optimization and Training
*   **Objective**: CrossEntropyLoss computed on the logits of the seed nodes in each batch.
*   **Parameters**: Optimizes all layers (both encoder and classifier head) end-to-end.
*   **Optimizer & Scheduler**: AdamW (`lr=1e-4`, weight decay `1e-6`) paired with `CosineAnnealingLR` scheduler.
*   **Early Stopping**: Monitored on validation split Accuracy. Training stops if validation performance does not improve within a patience of `early_stop` (200) epochs.

---

## 5. Checkpoint Selection & Evaluation Metrics Pipeline

The evaluation pipeline is triggered natively inside the training loop of `finetune.py` whenever a new best validation performance is achieved.

```
                  +-----------------------------------------+
                  |         result['val'] > best_val        |
                  +--------------------+--------------------+
                                       |
                                       v
                  +--------------------+--------------------+
                  |    Save model to: best_model.pt         |
                  +--------------------+--------------------+
                                       |
                                       v
                  +--------------------+--------------------+
                  |  eval_node(..., return_predictions=True)|
                  +--------------------+--------------------+
                                       |
                                       v
                  +--------------------+--------------------+
                  |         utils/save_metrics.py           |
                  +--------------------+--------------------+
                                       |
                                       +
          +-------+-------+------------+------------+-------+-------+
          |               |                         |               |
          v               v                         v               v
    +-----+------+  +-----+------+            +-----+------+  +-----+------+
    |metrics.json|  |metrics.csv |            |predictions.  |  |  confusion_|
    |            |  |            |            |     csv      |  |  matrix.png|
    +------------+  +------------+            +------------+  +------------+
```

### A. Execution Flow
1.  When validation accuracy exceeds the historical best, the current model state is saved to `model/finetune_model/<dataset>/best_model.pt`.
2.  The pipeline invokes the GNN's evaluation function once using the model in memory:
    
    `eval_res = eval_node(model=task_model, data=data, split=split, params=params, return_predictions=True)`
    
3.  `eval_node` performs a forward pass, detaches the tensors, moves them to CPU, and returns `y_true` and `logits`.
4.  These outputs are passed to `save_metrics()`, which applies the `test_mask` to filter test node indexes.

### B. Artifact Equations & Formats
All metrics are computed on the test nodes using `scikit-learn`:

*   **Accuracy**:
    
    $$\text{Accuracy} = \frac{1}{|T_{\text{test}}|} \sum_{i \in T_{\text{test}}} \mathbb{I}(\hat{y}_i = y_i)$$
    
*   **Precision (Macro)**: Unweighted average of precision per class:
    
    $$\text{Precision}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^C \frac{TP_c}{TP_c + FP_c}$$
    
*   **Recall (Macro)**: Unweighted average of recall per class:
    
    $$\text{Recall}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^C \frac{TP_c}{TP_c + FN_c}$$
    
*   **F1-Score (Macro)**: Unweighted average of F1 per class:
    
    $$\text{F1}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^C 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
    
*   **ROC-AUC (Binary Classification)**: Computes the Area Under the Receiver Operating Characteristic curve. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
*   **Average Precision (Binary Classification)**: Computes Precision-Recall Area Under Curve (PR-AUC) using the positive class probability:
    
    $$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$
    
    where $P_n$ and $R_n$ are the precision and recall at the $n$-th threshold.

### C. File Output Specifications
1.  **`metrics.json`**: Saves a JSON dictionary containing the dataset name, best epoch, best validation accuracy, number of classes, test size, and all computed classification metrics.
2.  **`metrics.csv`**: A single-row CSV sheet with the same headers and values as `metrics.json` for easy aggregation.
3.  **`classification_report.txt`**: Standard scikit-learn text report showing precision, recall, and F1-score for each class.
4.  **`predictions.csv`**:
    *   *Columns*: `node_id`, `ground_truth`, `prediction`, `prob_class0`, `prob_class1`, ...
    *   *Node ID mapping*: Extracted using `torch.where(test_mask)[0]`.
    *   *Probabilities*: Softmax activations computed over raw logits.
5.  **`confusion_matrix.png`**: Matplotlib diagram showing true vs. predicted counts annotated inside cell heatmaps.

---

## 6. Developer's Dataset Integration Guide

Follow this step-by-step developer guide to integrate a completely new fraud dataset (e.g., `MyFraudData`) into this repository.

### Step 1: Preprocessing & Graph Construction (Starting from Raw CSVs)

If you only have raw CSV files (for example, `nodes.csv` containing accounts/entities with labels and features, and `edges.csv` containing transactions/communications), you can compile them into the universal `FraudGraph` format. 

Create a converter script `custom_pretrain/converters/convert_myfraud.py` using the following implementation structure:

```python
import os
import pandas as pd
import pickle
from custom_pretrain.schema.graph_schema import Node, Edge, FraudGraph

def load_from_csv(nodes_csv_path, edges_csv_path):
    # Initialize the empty graph object
    graph = FraudGraph(name="MyFraudData")
    
    # 1. Load Node CSV (contains fields like account_id, label, and attributes)
    df_nodes = pd.read_csv(nodes_csv_path)
    print(f"Loaded {len(df_nodes)} nodes from CSV.")
    
    # Parse each row into a Node instance
    for _, row in df_nodes.iterrows():
        node_id = str(row["account_id"])
        label = int(row["label"]) if not pd.isna(row["label"]) else None
        
        # Collect numerical features into a dictionary
        features = {
            "balance": float(row["balance"]),
            "age_days": float(row["age_days"]),
        }
        
        node = Node(
            id=node_id,
            type="bank_account",
            features=features,
            text="",
            label=label
        )
        graph.add_node(node)
        
    # 2. Load Edge CSV (contains source, destination, and properties)
    df_edges = pd.read_csv(edges_csv_path)
    print(f"Loaded {len(df_edges)} edges from CSV.")
    
    # Parse each row into an Edge instance
    for _, row in df_edges.iterrows():
        src_id = str(row["source_account"])
        dst_id = str(row["destination_account"])
        
        # Verify edge nodes exist in node set, create placeholder if missing
        if src_id not in graph.nodes:
            graph.add_node(Node(id=src_id, type="bank_account", features={}, text="", label=None))
        if dst_id not in graph.nodes:
            graph.add_node(Node(id=dst_id, type="bank_account", features={}, text="", label=None))
            
        edge = Edge(
            src=src_id,
            dst=dst_id,
            edge_type="transaction",
            weight=float(row.get("amount", 1.0)),
            timestamp=str(row.get("timestamp", "")),
            features={}
        )
        graph.add_edge(edge)
        
    # 3. Compute Degree Properties dynamically
    in_deg = {nid: 0 for nid in graph.nodes}
    out_deg = {nid: 0 for nid in graph.nodes}
    for edge in graph.edges:
        out_deg[edge.src] += 1
        in_deg[edge.dst] += 1
        
    for nid, node in graph.nodes.items():
        node.features["in_degree"] = in_deg[nid]
        node.features["out_degree"] = out_deg[nid]
        node.features["total_degree"] = in_deg[nid] + out_deg[nid]
        
    # Validate structure and save to serialized pickle
    graph.validate()
    
    output_path = "custom_pretrain/cache_output/MyFraudData_graph.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
        
    print(f"Successfully serialized FraudGraph to {output_path}")

if __name__ == "__main__":
    load_from_csv("raw_data/nodes.csv", "raw_data/edges.csv")
```

### Step 2: Semantic Text Generation
1. Open `custom_pretrain/text_generation/node_describer.py`. Add a describer function for your dataset:
   ```python
   def describe_myfraud_node(node) -> str:
       # Compile your node's features into a natural language sentence
       return f"Bank account {node.id}. Features: ... Degree: {node.features['total_degree']}"
   ```
2. Register this function in `get_node_description()`.
3. Generate class descriptions in `custom_pretrain/text_generation/class_describer.py` (e.g., `"benign"` vs `"fraud"`).
4. Run `dataset_text_builder.py` to generate description JSON files in `custom_pretrain/text_cache/MyFraudData/`:
   - `node_text.json`
   - `class_text.json`

### Step 3: Embeddings Generation
1. Compute node embeddings using the SentenceTransformer script:
   ```bash
   python custom_pretrain/multi_gpu_embed.py --dataset MyFraudData --text_file custom_pretrain/text_cache/MyFraudData/node_text.json --gpu 0
   ```
2. Generate class embeddings by running:
   ```bash
   python custom_pretrain/packaging/prepare_class_embeddings.py --dataset MyFraudData
   ```
   This outputs `node_embeddings.npy`, `node_ids.json`, and `class_embeddings.npy` inside `custom_pretrain/embeddings/MyFraudData/`.

### Step 4: Compiling `geometric_data_processed.pt`
1. Run the universal converter to align embeddings, generate split masks, and package the graph into a PyG Data container:
   ```bash
   python custom_pretrain/packaging/universal_converter.py --dataset MyFraudData
   ```
2. Verify that the processed file is created at:
   `cache_data/MyFraudData/processed/geometric_data_processed.pt`

### Step 5: Repository Integration & Downstream Fine-Tuning
1. Register `MyFraudData` in `data/finetune_data.py`:
   - Add `"MyFraudData"` to `citation_datasets` (lines 15-20).
2. Configure dataset hyperparameters in `config/base.yaml`:
   ```yaml
     MyFraudData:
       normalize: "batch"
       sft_lr: 0.000001
       sft_epochs: 100
       lr: 0.0001
       decay: 0.000001
   ```
3. Run the fine-tuning execution script on Kaggle:
   ```bash
   python finetune.py --dataset MyFraudData --pt_data fraud --pt_epochs 20 --pt_lr 5e-5 --use_params
   ```
4. Once completed, evaluation reports and checkpoints will automatically be generated in:
   `model/finetune_model/MyFraudData/`

---

## 7. Developer's Guide: Downstream Fine-Tuning Only (Skipping Pretraining)

If you already have a pretrained Graph Foundation Encoder checkpoint (e.g., `encoder_20.pt` generated from joint fraud pretraining) and want to apply it directly to a downstream fraud dataset without running the self-supervised pretraining phase, follow this guide.

### Step 1: Pretrained Encoder Checkpoint Placement
`finetune.py` resolves the path to the pretrained GNN encoder based on your CLI arguments. You must place your pretrained checkpoint in the constructed subdirectory structure matching the hyperparameters used during its pretraining.

1. Create the expected subdirectory inside your local repository or Kaggle working directory:
   ```bash
   mkdir -p model/pretrain_model/lr_5e-05_hidden_768_layer_2_backbone_sage_fp_0.2_ep_0.2_alignreg_10.0_pt_data_fraud
   ```
2. Save your pretrained encoder weight file (e.g., `encoder_20.pt` containing the state dict of the `Encoder` layers) inside this folder:
   ```text
   model/pretrain_model/lr_5e-05_hidden_768_layer_2_backbone_sage_fp_0.2_ep_0.2_alignreg_10.0_pt_data_fraud/encoder_20.pt
   ```

### Step 2: Downstream Dataset Preparation
Place the packaged, processed PyTorch Geometric graph dataset in the `cache_data/` directory:
1. Ensure the dataset folder name matches the dataset identifier (e.g., `BUPT`, `Elliptic`, `IBM_AML`, or your custom `MyFraudData`).
2. The file MUST be serialized as a list of length 1 containing the PyG `Data` object, saved at:
   ```text
   cache_data/<DATASET_NAME>/processed/geometric_data_processed.pt
   ```
3. The `Data` object must contain:
   * `x`: Node indices tensor of shape `[num_nodes]` (typically `arange(num_nodes)`).
   * `node_text_feat`: Float tensor of shape `[num_nodes, 768]` containing the sentence transformer node embeddings.
   * `edge_index`: Long tensor of shape `[2, num_edges]` containing graph topology.
   * `y`: Labels tensor of shape `[num_nodes]` (`0` = benign, `1` = fraud, `-1` = unlabeled).
   * `train_mask`, `val_mask`, `test_mask`: Boolean split masks of shape `[num_nodes]`.

### Step 3: Hyperparameter Configuration
1. Register your dataset and set its learning rates, weight decays, epochs, and normalizations inside `config/base.yaml` under the `node:` section:
   ```yaml
     MyFraudData:
       normalize: "batch"
       sft_lr: 0.000001
       sft_epochs: 100
       lr: 0.0001
       decay: 0.000001
   ```
2. Update dataset domain mappings in `data/finetune_data.py`:
   - Append your dataset name to `citation_datasets` (which handles node classification tasks).

### Step 4: Run the Fine-Tuning Command
Run the downstream fine-tuning execution script. Pass the exact pretraining arguments to ensure the loader finds the correct pretrained directory path, and set `--use_params` to pull parameters from the YAML configuration:
```bash
python finetune.py \
  --dataset MyFraudData \
  --pt_data fraud \
  --pt_epochs 20 \
  --pt_lr 5e-5 \
  --use_params
```

*   `--pt_data fraud`: Specifies the pretraining dataset name used in the folder naming.
*   `--pt_epochs 20`: Tells the model loader to look for the file `encoder_20.pt`.
*   `--pt_lr 5e-5`: Matches the pretraining learning rate folder prefix.
*   `--use_params`: Instructs the script to load training hyperparameters from `config/base.yaml`.

### Step 5: Check Saved Artifacts
The training script will run for up to `epochs` (1000) with early stopping on validation performance. Once a new best validation performance is achieved, the script automatically generates evaluation artifacts inside `model/finetune_model/<DATASET_NAME>/`:
*   `best_model.pt`: The downstream model state dict.
*   `metrics.json` & `metrics.csv`: Performance reports (Accuracy, Precision, Recall, F1, and ROC-AUC/PR-AUC for binary tasks).
*   `classification_report.txt`: Tabular view of per-class metrics.
*   `predictions.csv`: Log of raw probabilities and predictions for each node in the test mask.
*   `confusion_matrix.png`: Heatmap visual of true positives, false positives, true negatives, and false negatives.

