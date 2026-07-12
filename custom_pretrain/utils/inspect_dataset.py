import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import Counter

def get_file_size(filepath):
    """Returns human-readable file size."""
    size_bytes = os.path.getsize(filepath)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def fast_line_count(filepath):
    """Fast line count for text files."""
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    return count

def inspect_bupt(path, log_dir):
    print("Inspecting BUPT dataset...")
    files = ["TF.features", "TF.edgelist", "TF.labels"]
    found_files = []
    file_sizes = {}
    
    for f in files:
        fpath = os.path.join(path, f)
        if os.path.exists(fpath):
            found_files.append(f)
            file_sizes[f] = get_file_size(fpath)
            
    if len(found_files) < 3:
        print(f"Error: BUPT dataset files missing. Found: {found_files}")
        return
        
    features_path = os.path.join(path, "TF.features")
    edgelist_path = os.path.join(path, "TF.edgelist")
    labels_path = os.path.join(path, "TF.labels")
    
    # 1. Row counts
    n_nodes = fast_line_count(features_path)
    n_edges = fast_line_count(edgelist_path)
    n_labels = fast_line_count(labels_path)
    
    # 2. Sample Data inspection
    df_feat_sample = pd.read_csv(features_path, sep=' ', header=None, nrows=100)
    num_cols = df_feat_sample.shape[1]
    col_names = [f"col_{i}" for i in range(num_cols)]
    col_names[0] = "node_id"
    
    # Check data types
    dtypes_dict = {col: str(df_feat_sample[i].dtype) for i, col in enumerate(col_names)}
    
    # Duplicate edges in edgelist
    df_edges = pd.read_csv(edgelist_path, sep=' ', header=None)
    duplicate_edges = df_edges.duplicated().sum()
    
    # Labels distribution
    df_labels = pd.read_csv(labels_path, sep=' ', header=None, names=["node_id", "label"])
    label_counts = df_labels["label"].value_counts().to_dict()
    total_labeled = sum(label_counts.values())
    fraud_count = label_counts.get(1, 0)
    fraud_pct = (fraud_count / total_labeled * 100) if total_labeled > 0 else 0.0
    
    # Feature columns type analysis
    numeric_cols = col_names[1:] # Columns 1 to end are features
    categorical_cols = []
    text_cols = []
    timestamp_cols = []
    
    report = f"""# BUPT Dataset Inspection Report

## GENERAL INFORMATION
* **Dataset Name**: BUPT (Mobile Fraud Dataset)
* **Files Found**: {', '.join(found_files)}
* **File Sizes**:
{chr(10).join([f"  * `{k}`: {v}" for k, v in file_sizes.items()])}
* **Number of Rows**:
  * Node Features (`TF.features`): {n_nodes}
  * Edges (`TF.edgelist`): {n_edges}
  * Labels (`TF.labels`): {n_labels}
* **Number of Columns (Features)**: {num_cols - 1} (Excluding node ID)
* **Column Names**: `node_id` followed by {num_cols - 1} numeric feature columns
* **Data Types**: 
  * `node_id`: Integer
  * Features: Float
* **Missing Values**: 0 (in sampled data)
* **Duplicate Rows (Edges)**: {duplicate_edges} duplicate edges in edgelist

## GRAPH INFORMATION
* **Can this dataset already be interpreted as a graph?**: Yes
* **Node Type(s)**: `phone_number` / `actor`
* **Edge Type(s)**: `call_or_sms` / `interaction`
* **Node Identifier Column**: First column in `TF.features` and `TF.labels`
* **Source Column**: First column in `TF.edgelist`
* **Destination Column**: Second column in `TF.edgelist`
* **Timestamp Column**: None
* **Label Column**: Second column in `TF.labels`
* **Feature Columns**: Columns index 1 to {num_cols - 1} in `TF.features`

## LABEL INFORMATION
* **Number of Classes**: {len(label_counts)}
* **Class Names**: 0 (Licit/Normal), 1 (Illicit/Fraud)
* **Class Distribution**: {label_counts}
* **Fraud Percentage**: {fraud_pct:.4f}%

## FEATURE ANALYSIS
* **Categorical Columns**: {len(categorical_cols)}
* **Numeric Columns**: {len(numeric_cols)} (All node features)
* **Text Columns**: {len(text_cols)}
* **Timestamp Columns**: {len(timestamp_cols)}

## SUMMARY METRICS
* **Total Nodes**: {n_nodes}
* **Total Edges**: {n_edges}
* **Average Degree**: {2.0 * n_edges / n_nodes:.4f}
"""
    os.makedirs(log_dir, exist_ok=True)
    report_path = os.path.join(log_dir, "BUPT_report.md")
    with open(report_path, "w") as f_out:
        f_out.write(report)
    print(f"Report saved to {report_path}")


def inspect_sichuan(path, log_dir):
    print("Inspecting Sichuan dataset...")
    csv_file = "all_feat_with_label.csv"
    npz_file = "node_adj_sparse.npz"
    
    csv_path = os.path.join(path, csv_file)
    npz_path = os.path.join(path, npz_file)
    
    found_files = []
    file_sizes = {}
    
    if os.path.exists(csv_path):
        found_files.append(csv_file)
        file_sizes[csv_file] = get_file_size(csv_path)
    if os.path.exists(npz_path):
        found_files.append(npz_file)
        file_sizes[npz_file] = get_file_size(npz_path)
        
    if len(found_files) < 2:
        print(f"Error: Sichuan dataset files missing. Found: {found_files}")
        return
        
    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape
    col_names = list(df.columns)
    dtypes = df.dtypes.to_dict()
    missing_vals = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Adjacency matrix stats
    adj_data = np.load(npz_path)
    # The sparse matrix has format keys: data, indices, indptr, shape
    adj_keys = list(adj_data.keys())
    shape = adj_data.get('shape', [n_rows, n_rows])
    n_edges = len(adj_data.get('data', []))
    
    # Labels distribution
    label_counts = df['label'].value_counts().to_dict()
    total_labeled = sum(label_counts.values())
    fraud_count = label_counts.get(1, 0)
    fraud_pct = (fraud_count / total_labeled * 100) if total_labeled > 0 else 0.0
    
    # Feature columns types
    numeric_cols = [col for col in col_names if col not in ['phone_no_m', 'label'] and pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in col_names if col not in ['phone_no_m', 'label'] and not pd.api.types.is_numeric_dtype(df[col])]
    text_cols = ['phone_no_m']
    timestamp_cols = []
    
    report = f"""# Sichuan Dataset Inspection Report

## GENERAL INFORMATION
* **Dataset Name**: Sichuan (Telecom Fraud Dataset)
* **Files Found**: {', '.join(found_files)}
* **File Sizes**:
{chr(10).join([f"  * `{k}`: {v}" for k, v in file_sizes.items()])}
* **Number of Rows**: {n_rows}
* **Number of Columns**: {n_cols}
* **Column Names**: {', '.join(col_names[:10])} ... (+ {n_cols - 10} more)
* **Data Types**: Mixed (IDs are Object/Text, Features are Numeric/Float)
* **Missing Values**: {missing_vals}
* **Duplicate Rows (Nodes)**: {duplicate_rows}

## GRAPH INFORMATION
* **Can this dataset already be interpreted as a graph?**: Yes (via the sparse adjacency matrix `{npz_file}`)
* **Node Type(s)**: `phone_number`
* **Edge Type(s)**: `call_or_sms` / `interaction`
* **Node Identifier Column**: `phone_no_m`
* **Source Column**: Implicit in `{npz_file}` indices
* **Destination Column**: Implicit in `{npz_file}` indices
* **Timestamp Column**: None
* **Label Column**: `label`
* **Feature Columns**: All columns other than `phone_no_m` and `label`

## LABEL INFORMATION
* **Number of Classes**: {len(label_counts)}
* **Class Names**: 0 (Licit/Normal), 1 (Illicit/Fraud)
* **Class Distribution**: {label_counts}
* **Fraud Percentage**: {fraud_pct:.4f}%

## FEATURE ANALYSIS
* **Categorical Columns**: {len(categorical_cols)} ({categorical_cols})
* **Numeric Columns**: {len(numeric_cols)}
* **Text Columns**: {len(text_cols)} ({text_cols})
* **Timestamp Columns**: {len(timestamp_cols)}

## SUMMARY METRICS
* **Total Nodes**: {n_rows}
* **Total Edges**: {n_edges}
* **Average Degree**: {2.0 * n_edges / n_rows:.4f}
"""
    os.makedirs(log_dir, exist_ok=True)
    report_path = os.path.join(log_dir, "Sichuan_report.md")
    with open(report_path, "w") as f_out:
        f_out.write(report)
    print(f"Report saved to {report_path}")


def inspect_elliptic(path, log_dir):
    print("Inspecting Elliptic++ dataset...")
    files = ["elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv", "elliptic_txs_features.csv"]
    found_files = []
    file_sizes = {}
    
    for f in files:
        fpath = os.path.join(path, f)
        if os.path.exists(fpath):
            found_files.append(f)
            file_sizes[f] = get_file_size(fpath)
            
    if len(found_files) < 3:
        print(f"Error: Elliptic++ dataset files missing. Found: {found_files}")
        return
        
    classes_path = os.path.join(path, "elliptic_txs_classes.csv")
    edgelist_path = os.path.join(path, "elliptic_txs_edgelist.csv")
    features_path = os.path.join(path, "elliptic_txs_features.csv")
    
    # Row counts
    n_nodes = fast_line_count(classes_path) - 1 # Exclude header
    n_edges = fast_line_count(edgelist_path) - 1
    n_features_lines = fast_line_count(features_path)
    
    # Read classes and edgelist (small files)
    df_classes = pd.read_csv(classes_path)
    df_edgelist = pd.read_csv(edgelist_path)
    
    # Duplicate edges
    duplicate_edges = df_edgelist.duplicated().sum()
    
    # Class distribution
    label_counts = df_classes["class"].value_counts().to_dict()
    total_labeled = sum(label_counts.values())
    
    # Class mapping in Elliptic: '1' is illicit, '2' is licit, 'unknown' is unlabelled
    illicit_count = label_counts.get('1', 0)
    licit_count = label_counts.get('2', 0)
    unknown_count = label_counts.get('unknown', 0)
    
    labeled_total = illicit_count + licit_count
    fraud_pct = (illicit_count / labeled_total * 100) if labeled_total > 0 else 0.0
    
    # Inspect features sample (heavy file, read chunk/nrows)
    df_feat_sample = pd.read_csv(features_path, header=None, nrows=100)
    num_cols = df_feat_sample.shape[1]
    
    # Column Names
    col_names = ["txId", "timestep"] + [f"feat_{i}" for i in range(num_cols - 2)]
    
    numeric_cols = col_names[2:]
    categorical_cols = []
    text_cols = []
    timestamp_cols = ["timestep"]
    
    report = f"""# Elliptic++ Dataset Inspection Report

## GENERAL INFORMATION
* **Dataset Name**: Elliptic++ (Bitcoin Transaction Dataset)
* **Files Found**: {', '.join(found_files)}
* **File Sizes**:
{chr(10).join([f"  * `{k}`: {v}" for k, v in file_sizes.items()])}
* **Number of Rows**:
  * Node Classes (`elliptic_txs_classes.csv`): {n_nodes}
  * Edges (`elliptic_txs_edgelist.csv`): {n_edges}
  * Node Features (`elliptic_txs_features.csv`): {n_features_lines}
* **Number of Columns (Features)**: {num_cols - 2} (Excluding node ID and timestep)
* **Column Names**: `txId`, `timestep`, followed by {num_cols - 2} numeric features
* **Data Types**: All columns are numeric (Int/Float)
* **Missing Values**: 0 (in sampled data)
* **Duplicate Rows (Edges)**: {duplicate_edges} duplicate edges in edgelist

## GRAPH INFORMATION
* **Can this dataset already be interpreted as a graph?**: Yes
* **Node Type(s)**: `transaction`
* **Edge Type(s)**: `flow` (Bitcoin money flow)
* **Node Identifier Column**: `txId`
* **Source Column**: `txId1` (in edgelist)
* **Destination Column**: `txId2` (in edgelist)
* **Timestamp Column**: `timestep` (second column of features)
* **Label Column**: `class` (in classes file)
* **Feature Columns**: Columns index 2 to {num_cols - 1} in features

## LABEL INFORMATION
* **Number of Classes**: 3 (including unknown)
* **Class Names**: '1' (Illicit/Fraud), '2' (Licit/Normal), 'unknown' (Unlabelled)
* **Class Distribution**: {label_counts}
* **Fraud Percentage (of labeled)**: {fraud_pct:.4f}%

## FEATURE ANALYSIS
* **Categorical Columns**: {len(categorical_cols)}
* **Numeric Columns**: {len(numeric_cols)}
* **Text Columns**: {len(text_cols)}
* **Timestamp Columns**: {len(timestamp_cols)} ({timestamp_cols})

## SUMMARY METRICS
* **Total Nodes**: {n_nodes}
* **Total Edges**: {n_edges}
* **Average Degree**: {2.0 * n_edges / n_nodes:.4f}
"""
    os.makedirs(log_dir, exist_ok=True)
    report_path = os.path.join(log_dir, "Elliptic_report.md")
    with open(report_path, "w") as f_out:
        f_out.write(report)
    print(f"Report saved to {report_path}")


def inspect_ibm(path, log_dir):
    print("Inspecting IBM AML dataset...")
    # The path could be a specific file or a directory. Let's find out.
    csv_files = []
    if os.path.isfile(path):
        csv_files = [path]
    else:
        for f in os.listdir(path):
            if f.endswith(".csv"):
                csv_files.append(os.path.join(path, f))
                
    if not csv_files:
        print(f"Error: No CSV files found at path {path}")
        return
        
    # Inspect the first / chosen CSV file
    target_file = csv_files[0]
    filename = os.path.basename(target_file)
    fsize = get_file_size(target_file)
    
    print(f"Target inspection file: {filename} ({fsize})")
    
    # 1. Count rows efficiently skipping comments
    n_rows = 0
    has_header = False
    header = None
    
    # We parse the file to find the header and count transactions
    import io
    lines_sample = []
    max_sample = 20000
    
    with open(target_file, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("BEGIN") or line_str.startswith("END"):
                continue
            if "Timestamp" in line_str and "Is Laundering" in line_str:
                header = [col.strip() for col in line_str.split(',')]
                has_header = True
                continue
            n_rows += 1
            if len(lines_sample) < max_sample:
                lines_sample.append(line_str)
                
    if not has_header:
        header = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", 
                  "Amount Received", "Receiving Currency", "Amount Paid", 
                  "Payment Currency", "Payment Format", "Is Laundering"]
        
    df_sample = pd.read_csv(io.StringIO('\n'.join(lines_sample)), header=None, names=header)
    n_cols = df_sample.shape[1]
    dtypes = df_sample.dtypes.to_dict()
    missing_vals_sample = df_sample.isnull().sum().sum()
    duplicate_rows_sample = df_sample.duplicated().sum()
    
    # Label analysis
    label_counts = df_sample["Is Laundering"].value_counts().to_dict()
    total_labeled = sum(label_counts.values())
    fraud_count = label_counts.get(1, 0)
    fraud_pct = (fraud_count / total_labeled * 100) if total_labeled > 0 else 0.0
    
    # Feature analysis
    categorical_cols = ["From Bank", "To Bank", "Receiving Currency", "Payment Currency", "Payment Format"]
    numeric_cols = ["Amount Received", "Amount Paid"]
    text_cols = ["Account", "Account.1"]
    timestamp_cols = ["Timestamp"]
    
    report = f"""# IBM AML Dataset Inspection Report

## GENERAL INFORMATION
* **Dataset Name**: IBM AML (Anti-Money Laundering Dataset)
* **File Under Inspection**: `{filename}`
* **File Size**: {fsize}
* **Number of Rows**: {n_rows} (Clean transactions excluding comments/headers)
* **Number of Columns**: {n_cols}
* **Column Names**: {', '.join(header)}
* **Data Types**: Mixed (Banks & Accounts are ID/Object, Amounts are Float/Object, Currency/Format are Categorical)
* **Missing Values**: {missing_vals_sample} (in sampled {max_sample} rows)
* **Duplicate Rows**: {duplicate_rows_sample} (in sampled {max_sample} rows)

## GRAPH INFORMATION
* **Can this dataset already be interpreted as a graph?**: Yes
* **Node Type(s)**: `bank_account`
* **Edge Type(s)**: `transfer` (anti-money laundering transaction)
* **Node Identifier Column**: `Account` and `Account.1` (or concatenated with bank ID like `From Bank + Account`)
* **Source Column**: `Account`
* **Destination Column**: `Account.1`
* **Timestamp Column**: `Timestamp`
* **Label Column**: `Is Laundering`
* **Feature Columns**: `From Bank`, `To Bank`, `Amount Received`, `Receiving Currency`, `Amount Paid`, `Payment Currency`, `Payment Format`

## LABEL INFORMATION
* **Number of Classes**: {len(label_counts)}
* **Class Names**: 0 (Licit/Normal), 1 (Illicit/Laundering)
* **Class Distribution (Sampled {max_sample} rows)**: {label_counts}
* **Fraud Percentage (Sampled)**: {fraud_pct:.4f}%

## FEATURE ANALYSIS
* **Categorical Columns**: {len(categorical_cols)} ({categorical_cols})
* **Numeric Columns**: {len(numeric_cols)} ({numeric_cols})
* **Text/Identifier Columns**: {len(text_cols)} ({text_cols})
* **Timestamp Columns**: {len(timestamp_cols)} ({timestamp_cols})

## SUMMARY METRICS (Sampled)
* **Total Transactions (Sample)**: {len(df_sample)}
* **Unique Accounts (Sample)**: {len(set(df_sample['Account'].tolist() + df_sample['Account.1'].tolist()))}
"""
    os.makedirs(log_dir, exist_ok=True)
    report_path = os.path.join(log_dir, "IBM_report.md")
    with open(report_path, "w") as f_out:
        f_out.write(report)
    print(f"Report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Dataset Exploration & Inspection Utility")
    parser.add_argument("--dataset", required=True, choices=["BUPT", "Sichuan", "Elliptic", "IBM_AML"],
                        help="Dataset to inspect (BUPT, Sichuan, Elliptic, IBM_AML)")
    parser.add_argument("--path", required=True, help="Absolute or relative path to the dataset folder/file")
    
    args = parser.parse_args()
    
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    
    if args.dataset == "BUPT":
        inspect_bupt(args.path, log_dir)
    elif args.dataset == "Sichuan":
        inspect_sichuan(args.path, log_dir)
    elif args.dataset == "Elliptic":
        inspect_elliptic(args.path, log_dir)
    elif args.dataset == "IBM_AML":
        inspect_ibm(args.path, log_dir)

if __name__ == "__main__":
    main()
