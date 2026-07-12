import os
import sys
import pickle
import json
import argparse
import numpy as np
import torch

try:
    from torch_geometric.data import Data
except ImportError:
    print("Warning: torch_geometric not installed. Running conversion will require it.")

# Add custom_pretrain root directory to import split_generator and graph_schema
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from custom_pretrain.packaging.split_generator import generate_masks

def convert_dataset(dataset_name, cache_output_dir, embeddings_dir, output_dir):
    """
    Universal converter: packages pickled FraudGraph and numpy embeddings into GIT compliant format.
    """
    print("=" * 60)
    print(f"Universal Converter starting for dataset: {dataset_name}")
    print("=" * 60)
    
    # 1. Load FraudGraph
    graph_filename_map = {
        "IBM_AML": "IBM_AML_graph.pkl",
        "BUPT": "BUPT_graph.pkl",
        "Elliptic": "Elliptic_graph.pkl"
    }
    
    graph_filename = graph_filename_map.get(dataset_name, f"{dataset_name}_graph.pkl")
    graph_path = os.path.join(cache_output_dir, graph_filename)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"FraudGraph pickle file not found at: {graph_path}")
        
    print(f"Loading FraudGraph from: {graph_path}...")
    with open(graph_path, "rb") as f:
        # Import schema inside to unpickle correctly
        from custom_pretrain.schema.graph_schema import Node, Edge, FraudGraph
        graph = pickle.load(f)
        
    print(f"Loaded FraudGraph: '{graph.name}' with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    
    # 2. Load Node Embeddings and IDs mapping
    dataset_embed_dir = os.path.join(embeddings_dir, dataset_name)
    npy_path = os.path.join(dataset_embed_dir, "node_embeddings.npy")
    json_path = os.path.join(dataset_embed_dir, "node_ids.json")
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Node embeddings file not found at: {npy_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Node IDs mapping file not found at: {json_path}")
        
    print(f"Loading Node Embeddings from: {npy_path}...")
    node_embeddings = np.load(npy_path)
    
    print(f"Loading Node IDs mapping from: {json_path}...")
    with open(json_path, "r") as f:
        node_ids = json.load(f)
        
    # Verification
    if len(node_ids) != node_embeddings.shape[0]:
        raise ValueError(f"CRITICAL: len(node_ids) ({len(node_ids)}) != node_embeddings rows ({node_embeddings.shape[0]})!")
        
    print("Verifying that all node IDs exist in the FraudGraph...")
    missing_nodes = [nid for nid in node_ids if nid not in graph.nodes]
    if missing_nodes:
        raise ValueError(f"CRITICAL: {len(missing_nodes)} node IDs from embeddings do not exist in FraudGraph! (Example: {missing_nodes[:5]})")
    print("Verification passed! All node IDs successfully aligned.")
    
    num_nodes = len(node_ids)
    
    # 3. Construct data.x
    x = torch.arange(num_nodes, dtype=torch.long)
    
    # 4. Construct data.node_text_feat
    node_text_feat = torch.tensor(node_embeddings, dtype=torch.float32)
    
    # 5. Construct edge_index
    # Build deterministic node ID to integer index mapping
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    print("Constructing edge list indices...")
    edge_index_list = []
    for edge in graph.edges:
        if edge.src in node_id_to_index and edge.dst in node_id_to_index:
            src_idx = node_id_to_index[edge.src]
            dst_idx = node_id_to_index[edge.dst]
            edge_index_list.append([src_idx, dst_idx])
            
    num_edges = len(edge_index_list)
    if num_edges == 0:
        print("Warning: Constructed edge list has 0 edges! Check if node IDs in edges align with node_ids.json.")
        
    # Convert to Long tensor shape [2, num_edges]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    print(f"Constructed edge_index of shape {list(edge_index.shape)}")
    
    # 6. Construct data.y
    y_list = []
    unlabeled_count = 0
    for node_id in node_ids:
        node = graph.nodes[node_id]
        if node.label is None or node.label == -1:
            y_list.append(-1)
            unlabeled_count += 1
        else:
            y_list.append(int(node.label))
            
    y = torch.tensor(y_list, dtype=torch.long)
    print(f"Constructed labels tensor of shape {list(y.shape)} ({unlabeled_count} unlabeled nodes assigned -1).")
    
    # 7. Construct class_node_text_feat
    class_npy_path = os.path.join(dataset_embed_dir, "class_embeddings.npy")
    if not os.path.exists(class_npy_path):
        raise FileNotFoundError(f"Class embeddings not found at: {class_npy_path}. Run prepare_class_embeddings.py first.")
        
    print(f"Loading Class Embeddings from: {class_npy_path}...")
    class_embeddings = np.load(class_npy_path)
    class_node_text_feat = torch.tensor(class_embeddings, dtype=torch.float32)
    print(f"Loaded class_node_text_feat of shape {list(class_node_text_feat.shape)}")
    
    # 8. Construct edge_attr & edge_text_feat (Optional)
    edge_npy_path = os.path.join(dataset_embed_dir, "edge_embeddings.npy")
    edge_text_feat = None
    if os.path.exists(edge_npy_path):
        print(f"Loading Edge Embeddings from: {edge_npy_path}...")
        edge_embeddings = np.load(edge_npy_path)
        edge_text_feat = torch.tensor(edge_embeddings, dtype=torch.float32)
        print(f"Loaded edge_text_feat of shape {list(edge_text_feat.shape)}")
    else:
        print("Edge embeddings not found. Omiting edge_attr / edge_text_feat (standard for node tasks).")
        
    # 9. Generate split masks
    train_mask, val_mask, test_mask = generate_masks(graph, node_ids)
    
    # Assemble PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        node_text_feat=node_text_feat,
        class_node_text_feat=class_node_text_feat,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    if edge_text_feat is not None:
        data.edge_text_feat = edge_text_feat
        data.edge_attr = edge_text_feat
        
    # 11. Serialization: Save as [data]
    output_path = os.path.join(output_dir, dataset_name, "processed", "geometric_data_processed.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving packaged dataset to: {output_path}...")
    torch.save([data], output_path)
    print("Serialization completed.")
    
    # Immediate reload validation
    print("Verifying saved file by reloading...")
    reloaded_list = torch.load(output_path, map_location="cpu")
    if not isinstance(reloaded_list, list) or len(reloaded_list) == 0:
        raise ValueError("CRITICAL: Saved file does not load as a non-empty python list!")
        
    reloaded_data = reloaded_list[0]
    print("Reload successful!")
    print(f"  Reloaded type: {type(reloaded_data)}")
    print(f"  Reloaded keys: {list(reloaded_data.keys())}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal FraudGraph to GIT PyG Dataset Converter")
    parser.add_argument("--dataset", type=str, choices=["IBM_AML", "BUPT", "Elliptic"], required=True,
                        help="Dataset name to package")
    parser.add_argument("--cache_output_dir", type=str, default="custom_pretrain/cache_output",
                        help="Path to cache_output directory")
    parser.add_argument("--embeddings_dir", type=str, default="custom_pretrain/embeddings",
                        help="Path to embeddings root directory")
    parser.add_argument("--output_dir", type=str, default="cache_data",
                        help="Root directory for generated datasets (e.g. cache_data/)")
                        
    args = parser.parse_args()
    
    try:
        convert_dataset(args.dataset, args.cache_output_dir, args.embeddings_dir, args.output_dir)
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
