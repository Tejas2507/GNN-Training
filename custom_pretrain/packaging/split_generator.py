import random
import torch

def generate_masks(graph, node_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Generates deterministic train/val/test split masks for the graph.
    Supports Chronological split (if 'timestep' is available in node features) 
    and Random split (seeded fallback).
    
    Args:
        graph (FraudGraph): The FraudGraph object.
        node_ids (list): List of node IDs in the order they appear in the embeddings.
        train_ratio (float): Ratio of nodes for training.
        val_ratio (float): Ratio of nodes for validation.
        test_ratio (float): Ratio of nodes for testing.
        seed (int): Seed for reproducibility.
        
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            Boolean masks (train_mask, val_mask, test_mask) of shape [num_nodes].
    """
    num_nodes = len(node_ids)
    
    # Initialize all masks to False
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # 1. Identify labeled nodes and collect timesteps
    labeled_indices = []
    timesteps = []
    
    for idx, node_id in enumerate(node_ids):
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        
        # Labeled nodes are those where label is not None and not -1
        if node.label is not None and node.label != -1:
            labeled_indices.append(idx)
            # Check for timestep in features
            t = node.features.get("timestep") if isinstance(node.features, dict) else None
            timesteps.append(t)
            
    num_labeled = len(labeled_indices)
    if num_labeled == 0:
        print("Warning: No labeled nodes found in the graph. All masks will be False.")
        return train_mask, val_mask, test_mask
        
    # Check if we should use chronological split
    valid_timesteps = [t for t in timesteps if t is not None]
    use_chronological = False
    
    # If at least 90% of labeled nodes have timesteps, use chronological split
    if len(valid_timesteps) >= 0.9 * num_labeled:
        # Check if there are at least some unique timesteps to sort by
        unique_t = set(valid_timesteps)
        if len(unique_t) > 1:
            use_chronological = True
            
    # Calculate split sizes
    train_size = int(train_ratio * num_labeled)
    val_size = int(val_ratio * num_labeled)
    test_size = num_labeled - train_size - val_size
    
    print(f"Split Generator Summary:")
    print(f"  Total Nodes: {num_nodes}")
    print(f"  Labeled Nodes: {num_labeled}")
    print(f"  Split Mode: {'Chronological' if use_chronological else 'Random'}")
    print(f"  Split Sizes: Train={train_size}, Val={val_size}, Test={test_size}")
    
    if use_chronological:
        # Pair indices with their timesteps
        # Fill missing timesteps with 0 if any (though at least 90% exist)
        pairs = []
        for i, idx in enumerate(labeled_indices):
            t = timesteps[i]
            t_val = float(t) if (t is not None and str(t).replace('.','',1).isdigit()) else 0.0
            pairs.append((t_val, idx))
            
        # Sort chronologically by timestep, secondary sort by index to preserve determinism
        pairs.sort(key=lambda x: (x[0], x[1]))
        sorted_labeled_indices = [x[1] for x in pairs]
        
        train_indices = sorted_labeled_indices[:train_size]
        val_indices = sorted_labeled_indices[train_size:train_size + val_size]
        test_indices = sorted_labeled_indices[train_size + val_size:]
    else:
        # Random split
        # Use a local random instance with a fixed seed for reproducibility
        rng = random.Random(seed)
        shuffled_indices = list(labeled_indices)
        rng.shuffle(shuffled_indices)
        
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size + val_size]
        test_indices = shuffled_indices[train_size + val_size:]
        
    # 2. Populate the boolean masks
    for idx in train_indices:
        train_mask[idx] = True
    for idx in val_indices:
        val_mask[idx] = True
    for idx in test_indices:
        test_mask[idx] = True
        
    return train_mask, val_mask, test_mask
