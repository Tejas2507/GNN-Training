# GIT Dataset Compatibility Audit Report

This report presents a thorough compatibility audit of the packaged GFM datasets against the original GIT repository code, tracing data loading and verification flows to ensure seamless execution during pretraining and fine-tuning on Kaggle.

---

## 1. Complete Loading & Execution Call Graph

When running `pretrain.py`, the dataset loading and transformation flow behaves as follows:

```text
pretrain.py (entrypoint)
    ↓  calls run(params)
pretrain.py::run
    ↓  calls unified_data(params)
data/pretrain_data.py::unified_data
    ↓  loops over pretrain_datasets list (including 'BUPT')
    ↓  calls get_data(params)
data/finetune_data.py::get_data
    ↓  routes based on dataset registry lists
    ↓  calls single_graph(params)
data/finetune_data.py::single_graph
    ↓  resolves path to 'geometric_data_processed.pt'
    ↓  calls torch.load(path, map_location="cpu", weights_only=False)
    ↓  returns loaded list, extracts [0] to yield PyG Data object
    ↓  calls ToUndirected()(data) to symmetrize edges
    ↓  returns data to get_data() and back to unified_data()
data/pretrain_data.py::unified_data (continued)
    ↓  calls preprocess(data)
        ↓  assigns data.x = torch.arange(num_nodes)
    ↓  calls VirtualNodeAugmentor.augment(data, task='node')
        ↓  appends num_nodes virtual nodes to data.x
        ↓  pads data.node_text_feat with a row of zeros
        ↓  connects original nodes to virtual nodes in data.edge_index
    ↓  calls postprocess(data)
        ↓  keeps only ['x', 'edge_index', 'node_text_feat']
        ↓  nullifies all other attributes (train_mask, class_node_text_feat, y, etc.)
    ↓  calls preprocess_data_dict(...) to shift indices for disjoint batching
    ↓  calls Batch.from_data_list(...) to group graphs
    ↓  returns unified batch to run()
pretrain.py::run (continued)
    ↓  calls get_pt_loader(pretrain_data, train_nodes, params)
utils/loader.py::get_pt_loader
    ↓  instantiates torch_geometric.loader.NeighborLoader
pretrain.py::run (continued)
    ↓  calls pretrain(model, loader, optimizer, ...)
pretrain.py::pretrain (loop)
    ↓  iterates for data in loader:
        ↓  extracts: x = data.node_text_feat[data.x].to(device)
        ↓  extracts: edge_index = data.edge_index.to(device)
        ↓  passes to model(graph, aug_graph1, aug_graph2, ...)
```

---

## 2. Dataset Expectations & Attribute Checklist

Below is the complete checklist of attributes accessed on the PyG `Data` object by the GNN model, splitters, loaders, and task scripts:

| Attribute | Mandatory / Optional | Expected Dtype | Expected Shape | Where Consumed |
|---|---|---|---|---|
| `x` | **Mandatory** | `torch.int64` | `[num_nodes]` | `pretrain.py:33`, `data/pretrain_data.py:92` |
| `edge_index` | **Mandatory** | `torch.int64` | `[2, num_edges]` | GNN Convolutions (`model/encoder.py`), `task/node.py` |
| `node_text_feat` | **Mandatory** | `torch.float32` | `[num_nodes, 768]` | GNN Inputs (`task/node.py`, `pretrain.py:33`) |
| `class_node_text_feat` | **Mandatory** (for node SFT) | `torch.float32` | `[num_classes, 768]` | Target mapping SFT (`task/node.py:21`, `task/edge.py:22`) |
| `y` | **Mandatory** (for fine-tuning) | `torch.int64` | `[num_nodes]` | Classification labels (`task/node.py:69`, `utils/split.py`) |
| `train_mask` | **Mandatory** (for fine-tuning) | `torch.bool` | `[num_nodes]` | Downstream split selection (`task/node.py:66`, `utils/split.py`) |
| `val_mask` | **Mandatory** (for fine-tuning) | `torch.bool` | `[num_nodes]` | Downstream split selection (`task/node.py`, `utils/split.py`) |
| `test_mask` | **Mandatory** (for fine-tuning) | `torch.bool` | `[num_nodes]` | Downstream split selection (`task/node.py`, `utils/split.py`) |
| `edge_text_feat` | Optional | `torch.float32` | `[num_edges, 768]` | Edge tasks (`task/edge.py:72`) |
| `edge_attr` | Optional | `torch.float32` | `[num_edges, 768]` | Edge convolutions (`model/encoder.py`) |

---

## 3. Comparison Audit

Our packaged dataset contains:
* `x` (Shape `[num_nodes]`, `torch.int64` sequential indices)
* `edge_index` (Shape `[2, num_edges]`, `torch.int64` topology indices)
* `node_text_feat` (Shape `[num_nodes, 768]`, `torch.float32` embeddings)
* `class_node_text_feat` (Shape `[num_classes, 768]`, `torch.float32` embeddings)
* `y` (Shape `[num_nodes]`, `torch.int64` labels with unlabeled as `-1`)
* `train_mask`, `val_mask`, `test_mask` (Shape `[num_nodes]`, `torch.bool` splits)

### Verdict:
* **Attributes Satisfied**: `x`, `edge_index`, `node_text_feat`, `class_node_text_feat`, `y`, `train_mask`, `val_mask`, `test_mask` are perfectly populated, matching expected shapes and datatypes.
* **Missing Attributes**: None of the mandatory attributes are missing. `edge_attr` and `edge_text_feat` are omitted as they are optional for node classification.
* **Extra Attributes**: None.
* **Mismatches**: None. The float tensors are `torch.float32` and long tensors are `torch.int64`.

---

## 4. Required Compatibility Patches

To successfully load and run our datasets in the GIT repository, the following minor patches must be applied:

### A. PyTorch 2.6 Deserialization Patch
By default, recent PyTorch versions enable secure loading (`weights_only=True`), which causes `torch.load` to fail when deserializing PyTorch Geometric custom classes.
* **Fix**: In `data/finetune_data.py`, modify line 67 to allow unpickling:
  ```python
  data = torch.load(path, map_location="cpu", weights_only=False)[0]
  ```

### B. Dataset Registration Patch
Since our dataset names (`BUPT`, `Elliptic`, `IBM_AML`) are not known to the original loading function, they will trigger a `ValueError` during data retrieval.
* **Fix**: Register the datasets in the citation list inside `data/finetune_data.py`:
  ```python
  citation_datasets = ['arxiv', 'cora', 'citeseer', 'pubmed', 'arxiv23', 'dblp', 'BUPT', 'Elliptic', 'IBM_AML']
  ```
  This maps them to the citation domain, uses the `node` task, and routes them to the `single_graph` loader automatically.

---

## 5. Verification of Pretraining Readiness

Our packaged BUPT dataset is **100% compliant** with the GIT loading pipeline. Since `postprocess()` discards everything except `x`, `edge_index`, and `node_text_feat` during pretraining, any downstream tasks (e.g. edge classification features) will not impact pretraining stability.

We are fully ready to register the datasets, package IBM AML and Elliptic++, and begin GFM pretraining.
