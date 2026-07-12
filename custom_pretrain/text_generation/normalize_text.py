import os
import sys
import pickle
import json
import gc
import numpy as np
import pandas as pd

# Ensure custom_pretrain path is in system path to import graph_schema
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from custom_pretrain.schema.graph_schema import Node, Edge, FraudGraph


def compute_vector_stats(feat_dict):
    """Computes stats from node features (keys matching feature_*)."""
    vals = [float(feat_dict[k]) for k in feat_dict if k.startswith("feature_")]
    if not vals:
        return 0.0, 0.0, 0.0, 0.0, 0.0, []
    arr = np.array(vals)
    mean_val = np.mean(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    std_val = np.std(arr)
    l2_norm = np.linalg.norm(arr)
    first_5 = list(arr[:5])
    return mean_val, min_val, max_val, std_val, l2_norm, first_5


def describe_ibm_node_normalized(node) -> str:
    features = node.features
    lifetime_sec = features.get("account_lifetime", 0)
    lifetime_days = lifetime_sec / (24 * 3600)
    
    in_trans = features.get("incoming_transaction_count", 0)
    out_trans = features.get("outgoing_transaction_count", 0)
    total_trans = features.get("total_transaction_count", 0)
    in_amt = features.get("incoming_amount_total", 0.0)
    out_amt = features.get("outgoing_amount_total", 0.0)
    fraud_ratio = features.get("fraud_ratio", 0.0)
    in_deg = features.get("in_degree", 0)
    out_deg = features.get("out_degree", 0)
    
    desc = (
        f"Bank account {node.id}. "
        f"Incoming transactions: {in_trans}. "
        f"Outgoing transactions: {out_trans}. "
        f"Total count: {total_trans}. "
        f"Incoming amount: ${in_amt:.2f}. "
        f"Outgoing amount: ${out_amt:.2f}. "
        f"Fraud ratio: {fraud_ratio:.2f}. "
        f"In degree: {in_deg}. "
        f"Out degree: {out_deg}. "
        f"Account lifetime: {lifetime_days:.1f} days."
    )
    return desc


def describe_bupt_node_normalized(node) -> str:
    features = node.features
    in_deg = features.get("in_degree", 0)
    out_deg = features.get("out_degree", 0)
    total_deg = features.get("total_degree", 0)
    
    mean_val, min_val, max_val, std_val, _, _ = compute_vector_stats(features)
    class_str = "unknown" if node.label is None else str(node.label)
    
    desc = (
        f"Phone number node. "
        f"Incoming communications: {in_deg}. "
        f"Outgoing communications: {out_deg}. "
        f"Total degree: {total_deg}. "
        f"Numerical features summary: dimension = 39, mean = {mean_val:.4f}, min = {min_val:.4f}, max = {max_val:.4f}, std = {std_val:.4f}. "
        f"Predicted class: {class_str}."
    )
    return desc


def describe_elliptic_node_normalized(node) -> str:
    features = node.features
    timestep = features.get("timestep", "unknown")
    in_deg = features.get("in_degree", 0)
    out_deg = features.get("out_degree", 0)
    
    mean_val, min_val, max_val, std_val, l2_norm, first_5 = compute_vector_stats(features)
    
    f5_str = ", ".join([f"{val:.4f}" for val in first_5])
    
    desc = (
        f"Bitcoin transaction node. "
        f"Time step: {timestep}. "
        f"In degree: {in_deg}. "
        f"Out degree: {out_deg}. "
        f"Numerical features summary: dimension = 165, mean = {mean_val:.4f}, min = {min_val:.4f}, max = {max_val:.4f}, std = {std_val:.4f}, L2 norm = {l2_norm:.4f}. "
        f"First 5 features: [{f5_str}]."
    )
    return desc


def describe_ibm_edge_normalized(edge) -> str:
    amount = edge.features.get("amount_received", edge.weight)
    currency = edge.features.get("receiving_currency", "USD")
    payment_type = edge.features.get("payment_format", "transfer")
    is_laundering = edge.features.get("is_laundering", 0)
    return f"Transfer of {amount:.2f} {currency} using {payment_type}. Laundering: {is_laundering}."


def describe_bupt_edge_normalized(edge) -> str:
    return "Communication event."


def describe_elliptic_edge_normalized(edge) -> str:
    timestep = edge.timestamp
    t_str = f" at time step {timestep}" if timestep else ""
    return f"Transaction flow{t_str}."


def get_normalized_node_description(node, dataset_name: str) -> str:
    dn_lower = dataset_name.lower().replace("-", "").replace("_", "").replace("+", "")
    if "ibm" in dn_lower:
        return describe_ibm_node_normalized(node)
    elif "bupt" in dn_lower:
        return describe_bupt_node_normalized(node)
    elif "elliptic" in dn_lower:
        return describe_elliptic_node_normalized(node)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def get_normalized_edge_description(edge, dataset_name: str) -> str:
    dn_lower = dataset_name.lower().replace("-", "").replace("_", "").replace("+", "")
    if "ibm" in dn_lower:
        return describe_ibm_edge_normalized(edge)
    elif "bupt" in dn_lower:
        return describe_bupt_edge_normalized(edge)
    elif "elliptic" in dn_lower:
        return describe_elliptic_edge_normalized(edge)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def process_dataset(dataset_name: str, pickle_filename: str, output_subdir: str):
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache_output")
    text_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "text_cache", output_subdir)
    
    pickle_path = os.path.join(cache_dir, pickle_filename)
    if not os.path.exists(pickle_path):
        print(f"Skipping {dataset_name} (pickle file not found)")
        return
        
    print(f"\nProcessing normalized text generation for {dataset_name}...")
    with open(pickle_path, "rb") as f:
        graph = pickle.load(f)
        
    # Generate & Save Node descriptions
    print("Normalizing node descriptions...")
    node_text = {}
    node_lengths = []
    node_word_counts = []
    
    for nid, node in graph.nodes.items():
        desc = get_normalized_node_description(node, dataset_name)
        node_text[nid] = desc
        node_lengths.append(len(desc))
        node_word_counts.append(len(desc.split()))
        
    # Assert word limits
    max_words = max(node_word_counts)
    assert max_words <= 512, f"Error: Node description word count {max_words} exceeds 512 limit!"
    
    node_output_path = os.path.join(text_cache_dir, "node_text.json")
    with open(node_output_path, "w") as f_out:
        json.dump(node_text, f_out, indent=2)
    print(f"Saved normalized node descriptions to {node_output_path}")
    del node_text
    gc.collect()
    
    # Generate & Save Edge descriptions
    print("Normalizing edge descriptions...")
    edge_text = []
    for edge in graph.edges:
        desc = get_normalized_edge_description(edge, dataset_name)
        edge_text.append(desc)
        
    edge_output_path = os.path.join(text_cache_dir, "edge_text.json")
    with open(edge_output_path, "w") as f_out:
        json.dump(edge_text, f_out, indent=2)
    print(f"Saved normalized edge descriptions to {edge_output_path}")
    del edge_text
    gc.collect()
    
    # Compute Validation Stats
    node_lengths = np.array(node_lengths)
    node_word_counts = np.array(node_word_counts)
    
    print("\n" + "="*50)
    print(f"NORMALIZED VALIDATION STATISTICS: {dataset_name}")
    print("="*50)
    print(f"Number of nodes                 : {len(graph.nodes)}")
    print(f"Number of edges                 : {graph.num_edges()}")
    print(f"Average node description length : {np.mean(node_lengths):.2f} chars")
    print(f"Median node description length  : {np.median(node_lengths):.1f} chars")
    print(f"95th percentile node length     : {np.percentile(node_lengths, 95):.1f} chars")
    print(f"Minimum node length             : {np.min(node_lengths)} chars")
    print(f"Maximum node length             : {np.max(node_lengths)} chars")
    print(f"Average node word count         : {np.mean(node_word_counts):.2f} words")
    print(f"Maximum node word count         : {np.max(node_word_counts)} words (limit: 512)")
    print("="*50 + "\n")
    
    del graph
    gc.collect()


def main():
    datasets = [
        ("IBM AML", "IBM_AML_graph.pkl", "IBM_AML"),
        ("BUPT", "BUPT_graph.pkl", "BUPT"),
        ("Elliptic++", "Elliptic_graph.pkl", "Elliptic")
    ]
    
    for name, filename, subdir in datasets:
        process_dataset(name, filename, subdir)


if __name__ == "__main__":
    main()
