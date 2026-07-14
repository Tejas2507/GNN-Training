# GIT Dataset Compatibility Report

This report provides the final verification of the PyTorch Geometric (`torch_geometric.data.Data`) structure expected by the GIT codebase, based on official repository code analysis and the processed dataset requirements.

---

## 1. Field-by-Field Verification (Official Dataset Structure)

Based on the loading and training pipeline of the GIT repository, a processed dataset (such as `ogbn-arxiv`) contains the following attributes when deserialized:

| Field | Python Type | Dtype | Shape | Mandatory / Optional | Usage Point | Equivalent in Our Pipeline |
|---|---|---|---|---|---|---|
| `x` | `torch.Tensor` | `torch.int64` | `[num_nodes]` | **Mandatory** | `pretrain.py:33`, `NeighborLoader` batching | Arange placeholder: `torch.arange(num_nodes)` |
| `edge_index` | `torch.Tensor` | `torch.int64` | `[2, num_edges]` | **Mandatory** | GNN convolutions (`model/encoder.py`) | Topology adjacency matrix (`graph.edges`) |
| `node_text_feat` | `torch.Tensor` | `torch.float32` | `[num_nodes, 768]` | **Mandatory** | GNN inputs (`task/node.py`, `pretrain.py`) | BGE node embeddings (`node_embeddings.npy`) |
| `class_node_text_feat` | `torch.Tensor` | `torch.float32` | `[num_classes, 768]` | **Mandatory** | SFT loss / targets (`task/node.py`, `task/edge.py`) | BGE class embeddings (`class_text.json`) |
| `y` | `torch.Tensor` | `torch.int64` | `[num_nodes]` | **Mandatory** | Classification targets (`task/node.py`) | Node target labels (`node.label`) |
| `train_mask` | `torch.Tensor` | `torch.bool` | `[num_nodes]` | **Mandatory** | Split evaluation (`task/node.py`, `utils/split.py`) | Generated splits (60% chronological/random) |
| `val_mask` | `torch.Tensor` | `torch.bool` | `[num_nodes]` | **Mandatory** | Split evaluation (`task/node.py`, `utils/split.py`) | Generated splits (20% chronological/random) |
| `test_mask` | `torch.Tensor` | `torch.bool` | `[num_nodes]` | **Mandatory** | Split evaluation (`task/node.py`, `utils/split.py`) | Generated splits (20% chronological/random) |
| `edge_text_feat` | `torch.Tensor` | `torch.float32` | `[num_edges, 768]` | Optional | Edge classification tasks (`task/edge.py`) | BGE edge embeddings (`edge_text.json`) |
| `edge_attr` | `torch.Tensor` | `torch.float32` | `[num_edges, 768]` | Optional | Convolution message passing (`model/encoder.py`) | BGE edge embeddings (`edge_text.json`) |

---

## 2. Deep Dive Verifications

### Task 4: Verification of `data.x`
* **Semantic Meaning**: `data.x` in the loaded PyG `Data` object is a **1D tensor of node indices** (i.e. `[0, 1, 2, ..., num_nodes - 1]`), NOT raw feature vectors or token IDs.
* **Why**: The GNN node feature matrix is stored in `data.node_text_feat`. During pretraining, the `NeighborLoader` samples mini-batches and returns subgraphs where `data.x` holds the indices of the sampled nodes. The features are then looked up on-the-fly in the training loop (`pretrain.py:33`):
  `x = data.node_text_feat[data.x].to(device)`
* **Can we set `x = torch.arange(num_nodes)`?** **Yes.** This is explicitly verified by the preprocessing step in `data/pretrain_data.py` (lines 147-148):
  ```python
  if dataset_name in citation_datasets + ecommerce_datasets + kg_datasets + temporal_datasets:
      data.x = torch.arange(data.num_nodes)
  ```

### Task 5: Verification of `class_node_text_feat`
* **Label Mapping**: Target integer class labels in `data.y` are mapped to text embeddings in the shared LLM space using direct indexing:
  `y = data.class_node_text_feat[data.y.squeeze()]` (found in `task/node.py:21` and `task/edge.py:22`).
* **Ordering**: The order of rows in `class_node_text_feat` must correspond **exactly** to the integer label values. E.g., row `i` contains the embedding of category `i`.
* **Dimensions**: The embedding dimension (second dimension of the tensor) must exactly match `node_text_feat` (768 for BGE-base) because the model projects node representations to this same dimension and computes Mean Squared Error (MSE) loss:
  `loss = F.mse_loss(y_pred, y)`

### Task 6: Verification of `edge_attr`
* **Message Passing Consumption**: `MySAGEConv` consumes `edge_attr` directly in its `message` function (`model/encoder.py:160-165`):
  ```python
  def message(self, x_j, edge_attr=None):
      if edge_attr is not None:
          return (x_j + edge_attr).relu()
      else:
          return x_j
  ```
* **Shape Constraints**: Because `edge_attr` is added directly to neighbor node features `x_j`, it must have the same dimension as `x_j` in the respective convolution layer. In multi-layer GNNs, the feature dimension changes from `input_dim` (768) in the first layer to `hidden_dim` (e.g., 128 or 256) in subsequent layers. 
* **Omission**: If using edge features, we must duplicate the edge embeddings in both `data.edge_text_feat` (which is read by tasks) and `data.edge_attr` (which PyG uses during convolutions).

### Task 7: Verification of Serialization
* **Exact Object Structure**: The saved file `geometric_data_processed.pt` must contain a **list containing a single `Data` object**, i.e., `[data]`.
* **Why**: Loaded in `data/finetune_data.py:67` using:
  `data = torch.load(path)[0]`
  If we save it simply as `torch.save(data, path)`, indexing it with `[0]` will fail or yield incorrect slices of attributes.

### Task 8: Directed vs. Undirected Graphs
* **Conversion**: The GIT loading pipeline always converts graphs to undirected networks using the PyG transform `ToUndirected()` immediately after loading (`data/finetune_data.py:68`):
  `data = ToUndirected()(data)`
* **Implication**: Any directed edges saved in our dataset will be automatically symmetrized at runtime. Therefore, we do not need to convert the graphs before saving them.

---

## 3. Pre-Implementation Checklist

Before implementing the converter, ensure:
- [x] Node embeddings are generated with BGE-base and stored matching the node order in `node_ids.json`.
- [x] Class descriptions are formulated and mapped to class embeddings on the fly.
- [x] Split masks are defined (chronological split using `timestep` where available, otherwise seeded random split).
- [x] Node labels (`data.y`) are structured as a 1D long tensor.
- [x] Unlabeled nodes (in Elliptic++) are assigned `-1` and excluded from all train/val/test masks.
- [x] The final object is serialized inside a python list `[data]` using `torch.save()`.
