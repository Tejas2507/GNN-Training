import os
import sys
import pickle
import datetime
import gc
from collections import Counter
import pandas as pd
import numpy as np

# Ensure custom_pretrain path is in system path to import graph_schema
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from custom_pretrain.schema.graph_schema import Node, Edge, FraudGraph


def parse_amount(val):
    """Parses amount string or number to float."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    val_str = str(val).replace('$', '').replace(',', '').strip()
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def load_graph(dataset_path: str) -> FraudGraph:
    """
    Converts the raw IBM AML HI-Small_Trans.csv dataset into a FraudGraph.
    
    Args:
        dataset_path (str): Path to HI-Small_Trans.csv.
        
    Returns:
        FraudGraph: The constructed, validated, and optimized fraud graph.
    """
    # Resolve target file path
    if os.path.isdir(dataset_path):
        target_file = os.path.join(dataset_path, "HI-Small_Trans.csv")
    else:
        target_file = dataset_path

    if not os.path.exists(target_file):
        raise FileNotFoundError(f"IBM AML CSV file not found at: {target_file}")
        
    print(f"Loading raw transaction dataset from: {target_file}")
    
    # Step 1: Read the complete CSV using optimized dtypes
    dtype_dict = {
        'From Bank': str,
        'To Bank': str,
        'Account': str,
        'Account.1': str,
        'Payment Format': 'category',
        'Receiving Currency': 'category',
        'Payment Currency': 'category',
        'Is Laundering': 'int8'
    }
    
    df = pd.read_csv(target_file, dtype=dtype_dict)
    n_rows, n_cols = df.shape
    cols = list(df.columns)
    
    # Auto-detect label & account columns (should match Step 1 results)
    timestamp_col = "Timestamp"
    label_col = "Is Laundering"
    src_account_col = "Account"
    dst_account_col = "Account.1"
    src_bank_col = "From Bank"
    dst_bank_col = "To Bank"
    amount_received_col = "Amount Received"
    amount_paid_col = "Amount Paid"
    payment_currency_col = "Payment Currency"
    receiving_currency_col = "Receiving Currency"
    payment_format_col = "Payment Format"
    
    print("\n" + "="*50)
    print("IBM AML Dataset Preprocessing - Step 1 Results")
    print("="*50)
    print(f"Dataset File            : {os.path.basename(target_file)}")
    print(f"Number of Rows          : {n_rows}")
    print(f"Number of Columns       : {n_cols}")
    print(f"Column Names            : {cols}")
    print(f"Detected Label Column   : {label_col}")
    print(f"Detected Account Columns: Source='{src_account_col}', Destination='{dst_account_col}'")
    print("="*50 + "\n")
    
    # Cast amount columns safely to numeric
    df[amount_received_col] = pd.to_numeric(df[amount_received_col], errors='coerce').fillna(0.0)
    df[amount_paid_col] = pd.to_numeric(df[amount_paid_col], errors='coerce').fillna(0.0)
    
    # Create unique Node IDs: "{BankID}_{AccountID}"
    df['src_id'] = df[src_bank_col].astype(str) + "_" + df[src_account_col].astype(str)
    df['dst_id'] = df[dst_bank_col].astype(str) + "_" + df[dst_account_col].astype(str)
    
    # Parse Timestamps to Datetime for lifetime calculations
    print("Parsing timestamps...")
    df['dt'] = pd.to_datetime(df[timestamp_col], format='%Y/%m/%d %H:%M')
    
    # Step 4: Pre-aggregate Node Features using Vectorized Pandas operations
    print("Pre-aggregating node features via groupby...")
    out_stats = df.groupby('src_id').agg(
        count=('src_id', 'size'),
        amount_total=(amount_paid_col, 'sum'),
        amount_mean=(amount_paid_col, 'mean'),
        laundering_count=(label_col, 'sum'),
        first_ts=('dt', 'min'),
        last_ts=('dt', 'max')
    )

    in_stats = df.groupby('dst_id').agg(
        count=('dst_id', 'size'),
        amount_total=(amount_received_col, 'sum'),
        amount_mean=(amount_received_col, 'mean'),
        laundering_count=(label_col, 'sum'),
        first_ts=('dt', 'min'),
        last_ts=('dt', 'max')
    )
    
    all_accounts = out_stats.index.union(in_stats.index)
    
    # Reindex to align all accounts
    out_df = out_stats.reindex(all_accounts)
    in_df = in_stats.reindex(all_accounts)
    
    # Fill numeric columns with 0
    numeric_cols = ['count', 'amount_total', 'amount_mean', 'laundering_count']
    out_df[numeric_cols] = out_df[numeric_cols].fillna(0)
    in_df[numeric_cols] = in_df[numeric_cols].fillna(0)
    
    # Vectorized computation of features
    in_cnt = in_df['count'].values
    out_cnt = out_df['count'].values
    total_cnt = in_cnt + out_cnt
    
    in_sum = in_df['amount_total'].values
    out_sum = out_df['amount_total'].values
    in_mean = in_df['amount_mean'].values
    out_mean = out_df['amount_mean'].values
    
    in_fraud = in_df['laundering_count'].values
    out_fraud = out_df['laundering_count'].values
    total_fraud = in_fraud + out_fraud
    
    fraud_ratio = total_fraud / np.maximum(total_cnt, 1)
    
    first_ts_series = pd.concat([out_df['first_ts'], in_df['first_ts']], axis=1).min(axis=1)
    last_ts_series = pd.concat([out_df['last_ts'], in_df['last_ts']], axis=1).max(axis=1)
    account_lifetime = (last_ts_series - first_ts_series).dt.total_seconds().fillna(0.0).values
    
    # Step 5: Graph Structural Features (in/out/total degree matches incoming/outgoing/total transaction counts)
    in_degree = in_cnt
    out_degree = out_cnt
    total_degree = total_cnt
    
    # Instantiate FraudGraph
    graph = FraudGraph(name="IBM AML")
    
    # Add Nodes
    print("Constructing and adding nodes to graph...")
    account_ids = all_accounts.astype(str).tolist()
    first_ts_strs = first_ts_series.dt.strftime('%Y/%m/%d %H:%M').fillna("").tolist()
    last_ts_strs = last_ts_series.dt.strftime('%Y/%m/%d %H:%M').fillna("").tolist()
    
    # Garbage collect unused series before looping to optimize memory
    del out_stats, in_stats, out_df, in_df, first_ts_series, last_ts_series
    gc.collect()
    
    for i, acc_id in enumerate(account_ids):
        f_ts = first_ts_strs[i] if first_ts_strs[i] != "" else None
        l_ts = last_ts_strs[i] if last_ts_strs[i] != "" else None
        
        node_features = {
            "incoming_transaction_count": int(in_cnt[i]),
            "outgoing_transaction_count": int(out_cnt[i]),
            "total_transaction_count": int(total_cnt[i]),
            "incoming_amount_total": float(in_sum[i]),
            "outgoing_amount_total": float(out_sum[i]),
            "incoming_amount_mean": float(in_mean[i]),
            "outgoing_amount_mean": float(out_mean[i]),
            "incoming_laundering_count": int(in_fraud[i]),
            "outgoing_laundering_count": int(out_fraud[i]),
            "total_laundering_count": int(total_fraud[i]),
            "fraud_ratio": float(fraud_ratio[i]),
            "first_timestamp": f_ts,
            "last_timestamp": l_ts,
            "account_lifetime": float(account_lifetime[i]),
            "in_degree": int(in_degree[i]),
            "out_degree": int(out_degree[i]),
            "total_degree": int(total_degree[i]),
            "fraud_count": int(total_fraud[i]),
            "transaction_count": int(total_cnt[i])
        }
        
        node = Node(
            id=acc_id,
            type="bank_account",
            features=node_features,
            text="",
            label=1 if total_fraud[i] > 0 else 0
        )
        graph.add_node(node)
        
    # Free memory arrays
    del in_cnt, out_cnt, total_cnt, in_sum, out_sum, in_mean, out_mean
    del in_fraud, out_fraud, total_fraud, fraud_ratio, account_lifetime
    gc.collect()
    
    # Step 3: Edge Construction (using memory efficient zips)
    print("Preparing edge features lists...")
    src_ids = df['src_id'].astype(str).tolist()
    dst_ids = df['dst_id'].astype(str).tolist()
    timestamps = df[timestamp_col].astype(str).tolist()
    amounts_paid = df[amount_paid_col].astype(float).tolist()
    amounts_received = df[amount_received_col].astype(float).tolist()
    pay_currencies = df[payment_currency_col].astype(str).tolist()
    rec_currencies = df[receiving_currency_col].astype(str).tolist()
    formats = df[payment_format_col].astype(str).tolist()
    is_launderings = df[label_col].astype(int).tolist()
    from_banks = df[src_bank_col].astype(str).tolist()
    to_banks = df[dst_bank_col].astype(str).tolist()
    
    # Free dataframe completely
    del df
    gc.collect()
    
    print("Adding edges to graph...")
    for i in range(len(src_ids)):
        edge_features = {
            "amount_paid": amounts_paid[i],
            "amount_received": amounts_received[i],
            "payment_currency": pay_currencies[i],
            "receiving_currency": rec_currencies[i],
            "payment_format": formats[i],
            "is_laundering": is_launderings[i],
            "from_bank": from_banks[i],
            "to_bank": to_banks[i]
        }
        edge = Edge(
            src=src_ids[i],
            dst=dst_ids[i],
            edge_type="bank_transfer",
            weight=amounts_received[i],
            timestamp=timestamps[i],
            features=edge_features
        )
        graph.add_edge(edge)
        
    # Free edges lists
    del src_ids, dst_ids, timestamps, amounts_paid, amounts_received
    del pay_currencies, rec_currencies, formats, is_launderings, from_banks, to_banks
    gc.collect()
    
    # Step 6: Labels Diagnostics
    fraud_nodes_count = sum(1 for node in graph.nodes.values() if node.label == 1)
    total_nodes = len(graph.nodes)
    benign_nodes_count = total_nodes - fraud_nodes_count
    fraud_pct = (fraud_nodes_count / total_nodes * 100) if total_nodes > 0 else 0.0
    
    graph.num_classes = 2
    
    # Metadata
    graph.metadata = {
        "dataset_name": "IBM AML",
        "source_file": os.path.basename(target_file),
        "number_of_transactions": graph.num_edges(),
        "number_of_accounts": total_nodes,
        "creation_timestamp": datetime.datetime.now().isoformat()
    }
    
    # Step 7: Validation
    graph.validate()
    
    # Custom step 7 diagnostic print
    print("\n" + "="*50)
    print("IBM AML Preprocessing - Step 7 Validation")
    print("="*50)
    print(f"Number of nodes : {total_nodes}")
    print(f"Number of edges : {graph.num_edges()}")
    print(f"Fraud nodes     : {fraud_nodes_count}")
    print(f"Benign nodes    : {benign_nodes_count}")
    print(f"Fraud %         : {fraud_pct:.4f}%")
    print(f"Average degree  : {2.0 * graph.num_edges() / total_nodes:.4f}")
    print(f"Node type counts: {{'bank_account': {total_nodes}}}")
    print(f"Edge type counts: {{'bank_transfer': {graph.num_edges()}}}")
    print("="*50 + "\n")
    
    # Step 8: Export
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache_output", "IBM_AML_graph.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving FraudGraph to pickle file: {output_path}...")
    with open(output_path, "wb") as f_out:
        pickle.dump(graph, f_out)
    print("Save complete.")
    
    # Step 9: Verification
    print("\n" + "="*50)
    print("IBM AML Preprocessing - Step 9 Verification")
    print("="*50)
    
    # First 5 nodes
    first_5_node_ids = list(graph.nodes.keys())[:5]
    print(f"First 5 Node IDs: {first_5_node_ids}")
    
    # First 5 edges
    first_5_edges = graph.edges[:5]
    print("\nFirst 5 Edges:")
    for edge in first_5_edges:
        print(f"  {edge.src} ---> {edge.dst} (weight: {edge.weight}, type: {edge.edge_type})")
        
    # First 5 node feature dicts
    print("\nFirst 5 Node Feature Dictionaries:")
    for nid in first_5_node_ids:
        print(f"  Node {nid}: {graph.nodes[nid].features}")
        
    # Label distribution
    labels = [node.label for node in graph.nodes.values()]
    print(f"\nLabel Distribution: {dict(Counter(labels))}")
    
    # Fraud ratio distribution
    fraud_ratios = [node.features["fraud_ratio"] for node in graph.nodes.values()]
    # Print simple stats of fraud ratios
    fraud_ratio_non_zero = [r for r in fraud_ratios if r > 0]
    print(f"Total nodes with non-zero fraud ratio: {len(fraud_ratio_non_zero)}")
    if fraud_ratio_non_zero:
        print(f"Max fraud ratio: {max(fraud_ratio_non_zero):.4f}")
        print(f"Mean non-zero fraud ratio: {np.mean(fraud_ratio_non_zero):.4f}")
    print("="*50 + "\n")
    
    return graph


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert IBM AML HI-Small_Trans.csv to FraudGraph")
    parser.add_argument("--path", required=True, help="Path to HI-Small_Trans.csv file")
    args = parser.parse_args()
    
    load_graph(args.path)
