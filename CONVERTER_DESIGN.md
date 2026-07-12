# Universal FraudGraph to GIT Converter Design

This document details the software design for the universal converter script that transforms standardized `FraudGraph` cache outputs into PyTorch Geometric `Data` objects compatible with the GIT repository.

---

## 1. Input and Output Files

### Inputs:
* **Graph Structure**: `custom_pretrain/cache_output/<dataset>_graph.pkl` (pickled `FraudGraph` containing graph topology, labels, and original features).
* **Node Embeddings**: `custom_pretrain/embeddings/<dataset>/node_embeddings.npy` (NumPy float array of shape `[num_nodes, 768]`).
* **Node Mapping**: `custom_pretrain/embeddings/<dataset>/node_ids.json` (JSON list specifying the exact ordering of Node IDs in the embedding array).
* **Class Descriptions**: `custom_pretrain/text_cache/<dataset>/class_text.json` (JSON dictionary mapping class labels to text descriptions).
* **Edge Descriptions (Optional)**: `custom_pretrain/text_cache/<dataset>/edge_text.json` (JSON list of edge descriptions).

### Outputs:
* **GIT PyG File**: `cache_data/<dataset>/processed/geometric_data_processed.pt` (Serialized PyTorch list containing a single `torch_geometric.data.Data` object).

---

## 2. Graph Construction Mechanics

### Node Ordering & Alignment:
To guarantee that the node embeddings in `node_embeddings.npy` align perfectly with the graph nodes in PyTorch Geometric:
1. Load `node_ids.json`, which contains a list of Node IDs: `[id_0, id_1, ..., id_N-1]`.
2. Map each Node ID to its array index: `node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}`.
3. The index in this list dictates the index of the node in the PyG graph: `node_0` in PyG will represent `id_0`, `node_1` will represent `id_1`, and so on.
4. Load `node_embeddings.npy` as a `torch.float32` tensor. Since the nodes are ordered matching the JSON, the feature matrix `data.node_text_feat` is simply the loaded embedding tensor.

### Edge Ordering & Adjacency (`edge_index`):
1. Iterate over the edges in the `FraudGraph.edges` list.
2. For each edge:
   * Look up `src_idx = node_id_to_index[edge.src]`.
   * Look up `dst_idx = node_id_to_index[edge.dst]`.
3. Construct the `edge_index` Long tensor of shape `[2, num_edges]` where column `k` contains `[src_idx, dst_idx]`.

### Edge Features (`edge_text_feat` / `edge_attr`):
* If edge descriptions exist and the task requires edge features, initialize a SentenceTransformer (`BAAI/bge-base-en-v1.5`) and encode the list of edge descriptions in `edge_text.json` on the fly to generate a `[num_edges, 768]` Float tensor.
* Duplicate this tensor in both `data.edge_text_feat` and `data.edge_attr` to ensure compatibility with both custom task scripts and standard PyG convolutions.

---

## 3. Labels and Mask Splits

### Class Labels (`y`):
* Create a Long tensor `y` of shape `[num_nodes]`.
* For each node ID at index `i` in `node_ids`:
  * If the node has a label in the `FraudGraph`: `y[i] = int(graph.nodes[node_id].label)`.
  * If the node is unlabeled (e.g., in Elliptic++): set `y[i] = -1` (or a placeholder `0`). These nodes will be excluded from all masks.

### Class Node Embeddings (`class_node_text_feat`):
* Load class descriptions from `class_text.json`.
* Encode these descriptions using a SentenceTransformer to produce a Float tensor of shape `[num_classes, 768]`.

### Split Strategies (`train_mask`, `val_mask`, `test_mask`):
We define two modular splitting strategies:
1. **Chronological split (Temporal)**:
   * If nodes contain a `timestep` in `node.features`, sort the labeled nodes chronologically by timestep.
   * Allocate the first 60% of nodes to `train_mask`, the next 20% to `val_mask`, and the remaining 20% to `test_mask`.
2. **Random split**:
   * If no `timestep` is available, randomly assign labeled nodes to splits using a seeded random state (`random_state=42` for reproducibility) with a 60/20/20 ratio.
* Unlabeled nodes (`label is None` or `y[i] == -1`) are assigned `False` in all three masks.

---

## 4. PyTorch Geometric Serialization

To match the single-graph loading pipeline of the GIT repository (where data is loaded using `torch.load(path)[0]`), the `Data` object is serialized inside a list of length 1:

```python
import torch
from torch_geometric.data import Data

data = Data(
    x=torch.arange(num_nodes, dtype=torch.long),
    edge_index=edge_index,
    node_text_feat=node_text_feat,
    class_node_text_feat=class_node_text_feat,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
)

# Optional edge attributes
if edge_text_feat is not None:
    data.edge_text_feat = edge_text_feat
    data.edge_attr = edge_text_feat

# Save inside a list
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save([data], output_path)
```

---

## 5. Dataset-Agnostic Extension

The converter will expose a public function:
`convert_dataset(dataset_name: str, cache_subdir: str, temporal_split: bool)`

By referencing the directories matching `dataset_name`, the script automatically maps and processes the keys, labels, and embeddings, enabling any future dataset (such as PaySim or AMLSim) to be processed seamlessly without modifying the core logic.
