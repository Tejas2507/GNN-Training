import os
import sys
import pickle
from collections import Counter

# Ensure custom_pretrain path is in system path to import graph_schema
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from custom_pretrain.schema.graph_schema import Node, Edge, FraudGraph


def get_file_size_mb(filepath):
    """Returns file size in MB."""
    if not os.path.exists(filepath):
        return "N/A"
    size_bytes = os.path.getsize(filepath)
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def compare():
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_output")
    files = {
        "IBM AML": "IBM_AML_graph.pkl",
        "BUPT": "BUPT_graph.pkl",
        "Elliptic++": "Elliptic_graph.pkl"
    }
    
    print("\n" + "="*80)
    print("UNIVERSAL FRAUD GRAPH COMPARISON REPORT")
    print("="*80)
    
    for dataset_name, filename in files.items():
        filepath = os.path.join(cache_dir, filename)
        if not os.path.exists(filepath):
            print(f"\n[WARNING] Pickle file for {dataset_name} not found at {filepath}")
            continue
            
        print(f"\nLoading {dataset_name} graph...")
        with open(filepath, "rb") as f:
            graph = pickle.load(f)
            
        # Compute stats
        n_nodes = graph.num_nodes()
        n_edges = graph.num_edges()
        
        # Labels
        labels = [node.label for node in graph.nodes.values() if node.label is not None]
        label_dist = dict(Counter(labels))
        
        # Fraud Nodes
        fraud_nodes = sum(1 for node in graph.nodes.values() if node.label == 1)
        total_labeled_nodes = sum(1 for node in graph.nodes.values() if node.label is not None)
        fraud_pct = (fraud_nodes / total_labeled_nodes * 100) if total_labeled_nodes > 0 else 0.0
        
        # Feature Dimension
        feat_dim = 0
        if n_nodes > 0:
            first_node = next(iter(graph.nodes.values()))
            feat_dim = len(first_node.features)
            
        # Types
        node_types = dict(Counter(node.type for node in graph.nodes.values()))
        edge_types = dict(Counter(edge.edge_type for edge in graph.edges))
        
        # Average Degree
        avg_degree = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0
        
        # Disk/Memory Usage
        disk_size = get_file_size_mb(filepath)
        
        # Validation Status
        print(f"Validating {dataset_name}...")
        validation_status = "PASSED" if graph.validate() else "FAILED"
        
        print("\n" + "-"*40)
        print(f"Dataset Name      : {dataset_name}")
        print(f"File Size on Disk : {disk_size}")
        print(f"Nodes             : {n_nodes}")
        print(f"Edges             : {n_edges}")
        print(f"Average Degree    : {avg_degree:.4f}")
        print(f"Node Types        : {node_types}")
        print(f"Edge Types        : {edge_types}")
        print(f"Classes           : {label_dist} (unlabelled: {n_nodes - total_labeled_nodes})")
        print(f"Fraud Nodes       : {fraud_nodes} / {total_labeled_nodes} labeled")
        print(f"Fraud %           : {fraud_pct:.4f}%")
        print(f"Feature Dimension : {feat_dim}")
        print(f"Validation Status : {validation_status}")
        print("-"*40)
        
        # Free memory before loading next dataset
        del graph
        import gc
        gc.collect()
        
    print("="*80 + "\n")


if __name__ == "__main__":
    compare()
