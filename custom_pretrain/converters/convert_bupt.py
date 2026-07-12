import os
import sys
import pickle
import datetime
import gc
import pandas as pd
import numpy as np

# Ensure custom_pretrain path is in system path to import graph_schema
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from custom_pretrain.schema.graph_schema import Node, Edge, FraudGraph


def load_graph(dataset_path: str) -> FraudGraph:
    """
    Converts raw BUPT dataset into a FraudGraph object.
    
    Args:
        dataset_path (str): Path to the BUPT dataset folder containing raw files.
        
    Returns:
        FraudGraph: The constructed and validated universal graph representation.
    """
    features_path = os.path.join(dataset_path, "TF.features")
    edgelist_path = os.path.join(dataset_path, "TF.edgelist")
    labels_path = os.path.join(dataset_path, "TF.labels")
    
    for p in [features_path, edgelist_path, labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing BUPT dataset file: {p}")
            
    print(f"Loading BUPT raw features from: {features_path}")
    
    # Instantiate FraudGraph
    graph = FraudGraph(name="BUPT")
    
    # 1. Load Node Features and Create Nodes
    # Format: node_id followed by 39 space-separated float features
    df_feat = pd.read_csv(features_path, sep=' ', header=None)
    num_rows, num_cols = df_feat.shape
    print(f"Loaded {num_rows} node features rows with {num_cols} columns.")
    
    node_ids = df_feat[0].astype(str).tolist()
    # Features (columns 1 to end)
    features_matrix = df_feat.iloc[:, 1:].values.astype(float)
    
    # Clear df_feat to save memory
    del df_feat
    gc.collect()
    
    print("Constructing BUPT nodes...")
    for i, nid in enumerate(node_ids):
        feat_dict = {f"feature_{j}": float(features_matrix[i, j]) for j in range(num_cols - 1)}
        node = Node(
            id=nid,
            type="phone_number",
            features=feat_dict,
            text="",
            label=None  # Assigned from TF.labels below
        )
        graph.add_node(node)
        
    del features_matrix
    gc.collect()
    
    # 2. Load and Apply Node Labels
    print(f"Loading BUPT labels from: {labels_path}")
    df_labels = pd.read_csv(labels_path, sep=' ', header=None, names=["node_id", "label"])
    df_labels["node_id"] = df_labels["node_id"].astype(str)
    
    labels_dict = dict(zip(df_labels["node_id"], df_labels["label"]))
    del df_labels
    gc.collect()
    
    # Apply labels to graph nodes
    for nid, label in labels_dict.items():
        if nid in graph.nodes:
            graph.nodes[nid].label = int(label)
        else:
            # If label contains nodes not in features, we can create them with empty/zero features
            # to preserve topology and labels, or ignore.
            # To keep original graph topology and preserve all labels exactly, we add the node:
            node = Node(
                id=nid,
                type="phone_number",
                features={},
                text="",
                label=int(label)
            )
            graph.add_node(node)
            
    # 3. Load and Add Edges
    print(f"Loading BUPT edgelist from: {edgelist_path}")
    df_edges = pd.read_csv(edgelist_path, sep=' ', header=None, names=["src", "dst"])
    df_edges["src"] = df_edges["src"].astype(str)
    df_edges["dst"] = df_edges["dst"].astype(str)
    
    src_list = df_edges["src"].tolist()
    dst_list = df_edges["dst"].tolist()
    
    del df_edges
    gc.collect()
    
    # Ensure all edge nodes exist in graph.nodes (unlabelled/unfeatured nodes might be added here)
    for i in range(len(src_list)):
        s_id = src_list[i]
        d_id = dst_list[i]
        
        if s_id not in graph.nodes:
            graph.add_node(Node(id=s_id, type="phone_number", features={}, text="", label=None))
        if d_id not in graph.nodes:
            graph.add_node(Node(id=d_id, type="phone_number", features={}, text="", label=None))
            
        edge = Edge(
            src=s_id,
            dst=d_id,
            edge_type="call_or_sms",
            weight=1.0,
            timestamp=None,
            features={}
        )
        graph.add_edge(edge)
        
    del src_list, dst_list
    gc.collect()
    
    # Compute degree features and add them to node.features (matching IBM AML structure)
    print("Computing graph degrees...")
    in_deg = {nid: 0 for nid in graph.nodes}
    out_deg = {nid: 0 for nid in graph.nodes}
    
    for edge in graph.edges:
        out_deg[edge.src] += 1
        in_deg[edge.dst] += 1
        
    # Set classes count based on unique labels found
    unique_labels = {node.label for node in graph.nodes.values() if node.label is not None}
    graph.num_classes = len(unique_labels)
    
    for nid, node in graph.nodes.items():
        node.features["in_degree"] = in_deg[nid]
        node.features["out_degree"] = out_deg[nid]
        node.features["total_degree"] = in_deg[nid] + out_deg[nid]
        
    # Metadata
    graph.metadata = {
        "dataset_name": "BUPT",
        "source_file": "TF.features, TF.edgelist, TF.labels",
        "number_of_transactions": graph.num_edges(),
        "number_of_accounts": graph.num_nodes(),
        "creation_timestamp": datetime.datetime.now().isoformat()
    }
    
    # Validate and Summarize
    graph.validate()
    graph.summary()
    
    # Export Graph to Pickle
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache_output", "BUPT_graph.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving FraudGraph to: {output_path}...")
    with open(output_path, "wb") as f_out:
        pickle.dump(graph, f_out)
    print("BUPT conversion completed.")
    
    return graph


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert BUPT Dataset to FraudGraph Schema")
    parser.add_argument("--path", required=True, help="Path to BUPT raw dataset folder")
    args = parser.parse_args()
    
    load_graph(args.path)
