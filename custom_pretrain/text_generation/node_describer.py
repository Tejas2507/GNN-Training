def describe_ibm_node(node) -> str:
    """Generates natural language description for an IBM AML bank account node."""
    features = node.features
    lifetime_sec = features.get("account_lifetime", 0)
    lifetime_days = lifetime_sec / (24 * 3600)
    
    in_trans = features.get("incoming_transaction_count", 0)
    out_trans = features.get("outgoing_transaction_count", 0)
    in_amt = features.get("incoming_amount_total", 0.0)
    out_amt = features.get("outgoing_amount_total", 0.0)
    total_transferred = in_amt + out_amt
    fraud_ratio = features.get("fraud_ratio", 0.0)
    in_deg = features.get("in_degree", 0)
    out_deg = features.get("out_degree", 0)
    
    desc = (
        f"Bank account {node.id}.\n"
        f"Incoming transactions: {in_trans}.\n"
        f"Outgoing transactions: {out_trans}.\n"
        f"Total transferred: ${total_transferred:.2f}.\n"
        f"Fraud ratio: {fraud_ratio:.2f}.\n"
        f"Account lifetime: {lifetime_days:.1f} days.\n"
        f"In degree: {in_deg}.\n"
        f"Out degree: {out_deg}."
    )
    return desc


def describe_bupt_node(node) -> str:
    """Generates natural language description for a BUPT phone number node."""
    features = node.features
    in_deg = features.get("in_degree", 0)
    out_deg = features.get("out_degree", 0)
    
    feat_strs = []
    # Collect numerical features sorted by feature index
    sorted_feat_keys = sorted([k for k in features.keys() if k.startswith("feature_")], 
                              key=lambda x: int(x.split('_')[1]))
    for k in sorted_feat_keys:
        feat_strs.append(f"{k} = {features[k]:.4f}")
        
    class_str = "unknown" if node.label is None else str(node.label)
    
    desc = (
        f"Phone number node.\n"
        f"Incoming communications: {in_deg}.\n"
        f"Outgoing communications: {out_deg}.\n"
        + "\n".join(feat_strs) + "\n"
        f"Predicted class: {class_str}."
    )
    return desc


def describe_elliptic_node(node) -> str:
    """Generates natural language description for an Elliptic++ bitcoin transaction node."""
    features = node.features
    timestep = features.get("timestep", "unknown")
    in_deg = features.get("in_degree", 0)
    out_deg = features.get("out_degree", 0)
    
    feat_strs = []
    sorted_feat_keys = sorted([k for k in features.keys() if k.startswith("feature_")], 
                              key=lambda x: int(x.split('_')[1]))
    for k in sorted_feat_keys:
        feat_strs.append(f"{k} = {features[k]:.4f}")
        
    desc = (
        f"Bitcoin transaction.\n"
        f"Time step: {timestep}.\n"
        + "\n".join(feat_strs) + "\n"
        f"Incoming edges: {in_deg}.\n"
        f"Outgoing edges: {out_deg}."
    )
    return desc


def get_node_description(node, dataset_name: str) -> str:
    """
    Dispatcher function to generate node descriptions based on dataset type.
    """
    dn_lower = dataset_name.lower().replace("-", "").replace("_", "").replace("+", "")
    if "ibm" in dn_lower:
        return describe_ibm_node(node)
    elif "bupt" in dn_lower:
        return describe_bupt_node(node)
    elif "elliptic" in dn_lower:
        return describe_elliptic_node(node)
    else:
        raise ValueError(f"Unknown dataset name for node description: {dataset_name}")
