import os
import sys
import argparse
import torch

try:
    from torch_geometric.data import Data
except ImportError:
    print("Warning: torch_geometric not installed. Running verification will require it.")

def verify_dataset(dataset_name, output_dir):
    """
    Exhaustive verification of a generated processed dataset.
    """
    pt_path = os.path.join(output_dir, dataset_name, "processed", "geometric_data_processed.pt")
    
    print("=" * 60)
    print(f"VERIFYING DATASET: {dataset_name}")
    print(f"File Path: {pt_path}")
    print("=" * 60)
    
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Processed dataset file not found at: {pt_path}")
        
    # Step 1: Load exactly like GIT loads it
    loaded_list = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded_list, list) or len(loaded_list) == 0:
        raise ValueError(f"FAIL: Loaded object from {pt_path} is not a non-empty python list!")
        
    data = loaded_list[0]
    
    # Track verification status
    warnings = []
    failures = []
    
    # Helper to check attribute existence, shape, and type
    def check_tensor(name, expected_dtype, expected_ndim, shape_constraints=None, optional=False):
        if not hasattr(data, name) or getattr(data, name) is None:
            if optional:
                print(f"  [OPTIONAL] {name} is not present. Skipped.")
                return None
            else:
                failures.append(f"Attribute '{name}' is missing.")
                return None
                
        val = getattr(data, name)
        if not isinstance(val, torch.Tensor):
            failures.append(f"Attribute '{name}' is of type {type(val)}, not torch.Tensor.")
            return None
            
        print(f"  [CHECK] {name:22} | Dtype: {str(val.dtype):12} | Shape: {list(val.shape)}")
        
        # Verify Dtype
        if val.dtype != expected_dtype:
            failures.append(f"Attribute '{name}' has dtype {val.dtype}, expected {expected_dtype}.")
            
        # Verify Dimension
        if val.ndim != expected_ndim:
            failures.append(f"Attribute '{name}' has {val.ndim} dimensions, expected {expected_ndim}.")
            
        # Verify NaNs
        if torch.isnan(val).any():
            failures.append(f"Attribute '{name}' contains NaN values!")
            
        # Shape constraints verification
        if shape_constraints:
            for dim_idx, constraint in enumerate(shape_constraints):
                if constraint is not None and val.shape[dim_idx] != constraint:
                    failures.append(f"Attribute '{name}' shape at dim {dim_idx} is {val.shape[dim_idx]}, expected {constraint}.")
                    
        return val

    # Verify high-level statistics first
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else None
    num_edges = data.num_edges if hasattr(data, 'num_edges') else None
    
    # 2. Check each attribute
    
    # x: Node indices
    x_val = check_tensor("x", torch.int64, 1)
    if x_val is not None:
        num_nodes = x_val.shape[0]
        # Verify it contains sequential indices 0 to num_nodes - 1
        expected_x = torch.arange(num_nodes, dtype=torch.long)
        if not torch.equal(x_val, expected_x):
            failures.append("x tensor does not contain sequential indices from 0 to num_nodes-1.")
            
    # node_text_feat
    node_feats = check_tensor("node_text_feat", torch.float32, 2, [num_nodes, 768])
    
    # class_node_text_feat
    class_feats = check_tensor("class_node_text_feat", torch.float32, 2, [None, 768])
    num_classes = class_feats.shape[0] if class_feats is not None else 0
    
    # y: Labels
    y_val = check_tensor("y", torch.int64, 1, [num_nodes])
    if y_val is not None and num_classes > 0:
        # Check label range: labels should be either -1 (unlabeled) or [0, num_classes-1]
        valid_mask = (y_val == -1) | ((y_val >= 0) & (y_val < num_classes))
        if not valid_mask.all():
            out_of_bounds = y_val[~valid_mask]
            failures.append(f"Labels tensor y contains invalid labels out of range [-1, {num_classes-1}]. (Found example values: {out_of_bounds[:5].tolist()})")
            
    # edge_index
    edge_idx_val = check_tensor("edge_index", torch.int64, 2, [2, None])
    if edge_idx_val is not None:
        num_edges = edge_idx_val.shape[1]
        # Verify indices are within [0, num_nodes - 1]
        if num_nodes:
            if edge_idx_val.numel() > 0:
                min_idx = edge_idx_val.min().item()
                max_idx = edge_idx_val.max().item()
                if min_idx < 0 or max_idx >= num_nodes:
                    failures.append(f"edge_index contains indices out of bounds [0, {num_nodes-1}]. Min found: {min_idx}, Max found: {max_idx}")
            else:
                warnings.append("edge_index has 0 edges.")
                
    # train_mask, val_mask, test_mask
    tr_mask = check_tensor("train_mask", torch.bool, 1, [num_nodes])
    va_mask = check_tensor("val_mask", torch.bool, 1, [num_nodes])
    te_mask = check_tensor("test_mask", torch.bool, 1, [num_nodes])
    
    # Split overlap checks
    if tr_mask is not None and va_mask is not None and te_mask is not None:
        overlap_tr_va = (tr_mask & va_mask).any().item()
        overlap_tr_te = (tr_mask & te_mask).any().item()
        overlap_va_te = (va_mask & te_mask).any().item()
        
        if overlap_tr_va:
            failures.append("Train and Val masks overlap!")
        if overlap_tr_te:
            failures.append("Train and Test masks overlap!")
        if overlap_va_te:
            failures.append("Val and Test masks overlap!")
            
        # Check that masked nodes are indeed labeled (y != -1)
        if y_val is not None:
            if y_val[tr_mask].eq(-1).any().item():
                failures.append("train_mask contains nodes with label -1 (unlabeled).")
            if y_val[va_mask].eq(-1).any().item():
                failures.append("val_mask contains nodes with label -1 (unlabeled).")
            if y_val[te_mask].eq(-1).any().item():
                failures.append("test_mask contains nodes with label -1 (unlabeled).")
                
        # Print counts
        print(f"  Mask Counts: Train={tr_mask.sum().item()} | Val={va_mask.sum().item()} | Test={te_mask.sum().item()}")

    # Optional Edge features
    edge_text_feats = check_tensor("edge_text_feat", torch.float32, 2, [num_edges, 768], optional=True)
    edge_attrs = check_tensor("edge_attr", torch.float32, 2, [num_edges, 768], optional=True)
    
    if (edge_text_feats is None) != (edge_attrs is None):
        warnings.append("One of edge_text_feat or edge_attr is present but the other is missing.")

    # 3. Overall validation verdict
    print("-" * 60)
    print("VERIFICATION RESULTS:")
    if warnings:
        print(f"  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        print("\n  VERDICT: FAILED ❌ (Dataset is not compatible with GIT)")
        sys.exit(1)
    else:
        print("\n  VERDICT: PASSED ✅ (Dataset is 100% compatible with GIT)")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIT Processed Dataset Verification Utility")
    parser.add_argument("--dataset", type=str, choices=["IBM_AML", "BUPT", "Elliptic"], required=True,
                        help="Dataset name to verify")
    parser.add_argument("--output_dir", type=str, default="cache_data",
                        help="Root directory where processed datasets are saved")
                        
    args = parser.parse_args()
    verify_dataset(args.dataset, args.output_dir)
