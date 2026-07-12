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
    Converts raw Elliptic++ dataset into a FraudGraph object.
    
    Args:
        dataset_path (str): Path to the Elliptic++ dataset folder containing raw files.
        
    Returns:
        FraudGraph: The constructed and validated universal graph representation.
    """
    classes_path = os.path.join(dataset_path, "elliptic_txs_classes.csv")
    edgelist_path = os.path.join(dataset_path, "elliptic_txs_edgelist.csv")
    features_path = os.path.join(dataset_path, "elliptic_txs_features.csv")
    
    for p in [classes_path, edgelist_path, features_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing Elliptic++ dataset file: {p}")
            
    print(f"Loading Elliptic++ classes from: {classes_path}")
    df_classes = pd.read_csv(classes_path)
    df_classes["txId"] = df_classes["txId"].astype(str)
    
    # Class mapping: '1' -> 1 (illicit), '2' -> 0 (licit), 'unknown' -> None
    class_map = {'1': 1, '2': 0, 'unknown': None}
    df_classes["mapped_label"] = df_classes["class"].astype(str).map(class_map)
    
    labels_dict = dict(zip(df_classes["txId"], df_classes["mapped_label"]))
    del df_classes
    gc.collect()
    
    print(f"Loading Elliptic++ features from: {features_path}...")
    df_feat = pd.read_csv(features_path, header=None)
    n_rows, n_cols = df_feat.shape
    print(f"Loaded {n_rows} features rows with {n_cols} columns.")
    
    node_ids = df_feat[0].astype(str).tolist()
    timesteps = df_feat[1].astype(int).tolist()
    features_matrix = df_feat.iloc[:, 2:].values.astype(float)
    
    # Keep track of timestep for each node to assign as edge timestamp later
    node_timesteps = dict(zip(node_ids, timesteps))
    
    del df_feat
    gc.collect()
    
    graph = FraudGraph(name="Elliptic++")
    
    print("Constructing Elliptic++ transaction nodes...")
    for i, nid in enumerate(node_ids):
        feat_dict = {f"feature_{j}": float(features_matrix[i, j]) for j in range(n_cols - 2)}
        feat_dict["timestep"] = int(timesteps[i])
        
        label_val = labels_dict.get(nid, None)
        if label_val is not None and not pd.isna(label_val):
            label_val = int(label_val)
        else:
            label_val = None
            
        node = Node(
            id=nid,
            type="transaction",
            features=feat_dict,
            text="",
            label=label_val
        )
        graph.add_node(node)
        
    del features_matrix, node_ids, timesteps
    gc.collect()
    
    print(f"Loading Elliptic++ edgelist from: {edgelist_path}")
    df_edges = pd.read_csv(edgelist_path)
    df_edges["txId1"] = df_edges["txId1"].astype(str)
    df_edges["txId2"] = df_edges["txId2"].astype(str)
    
    src_list = df_edges["txId1"].tolist()
    dst_list = df_edges["txId2"].tolist()
    
    del df_edges
    gc.collect()
    
    print("Adding edges to graph...")
    for i in range(len(src_list)):
        s_id = src_list[i]
        d_id = dst_list[i]
        
        # Ensure nodes exist
        if s_id not in graph.nodes:
            # Reconstruct missing nodes
            graph.add_node(Node(id=s_id, type="transaction", features={"timestep": None}, text="", label=None))
        if d_id not in graph.nodes:
            graph.add_node(Node(id=d_id, type="transaction", features={"timestep": None}, text="", label=None))
            
        # Get source timestamp
        src_ts = node_timesteps.get(s_id, None)
        edge_ts_str = str(src_ts) if src_ts is not None else None
        
        edge = Edge(
            src=s_id,
            dst=d_id,
            edge_type="flow",
            weight=1.0,
            timestamp=edge_ts_str,
            features={}
        )
        graph.add_edge(edge)
        
    del src_list, dst_list, node_timesteps
    gc.collect()
    
    # Compute degree features and add them to node.features
    print("Computing graph degrees...")
    in_deg = {nid: 0 for nid in graph.nodes}
    out_deg = {nid: 0 for nid in graph.nodes}
    
    for edge in graph.edges:
        out_deg[edge.src] += 1
        in_deg[edge.dst] += 1
        
    for nid, node in graph.nodes.items():
        node.features["in_degree"] = in_deg[nid]
        node.features["out_degree"] = out_deg[nid]
        node.features["total_degree"] = in_deg[nid] + out_deg[nid]
        
    # Unique non-None labels counts
    unique_labels = {node.label for node in graph.nodes.values() if node.label is not None}
    graph.num_classes = len(unique_labels)
    
    # Metadata
    graph.metadata = {
        "dataset_name": "Elliptic++",
        "source_file": "elliptic_txs_classes.csv, elliptic_txs_edgelist.csv, elliptic_txs_features.csv",
        "number_of_transactions": graph.num_edges(),
        "number_of_accounts": graph.num_nodes(),
        "creation_timestamp": datetime.datetime.now().isoformat()
    }
    
    # Validate and Summarize
    graph.validate()
    graph.summary()
    
    # Export Graph to Pickle
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache_output", "Elliptic_graph.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving FraudGraph to: {output_path}...")
    with open(output_path, "wb") as f_out:
        pickle.dump(graph, f_out)
    print("Elliptic++ conversion completed.")
    
    return graph


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Elliptic++ Dataset to FraudGraph Schema")
    parser.add_argument("--path", required=True, help="Path to Elliptic++ raw dataset folder")
    args = parser.parse_args()
    
    load_graph(args.path)
