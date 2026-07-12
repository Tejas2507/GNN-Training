def describe_ibm_edge(edge) -> str:
    """Generates natural language description for an IBM AML bank transfer edge."""
    amount = edge.features.get("amount_received", edge.weight)
    currency = edge.features.get("receiving_currency", "USD")
    payment_format = edge.features.get("payment_format", "transfer")
    return f"Bank transfer of {amount:.2f} {currency} using {payment_format}."


def describe_bupt_edge(edge) -> str:
    """Generates natural language description for a BUPT communication edge."""
    return "Communication between two phone numbers."


def describe_elliptic_edge(edge) -> str:
    """Generates natural language description for an Elliptic++ transaction flow edge."""
    return "Bitcoin transaction flow."


def get_edge_description(edge, dataset_name: str) -> str:
    """
    Dispatcher function to generate edge descriptions based on dataset type.
    """
    dn_lower = dataset_name.lower().replace("-", "").replace("_", "").replace("+", "")
    if "ibm" in dn_lower:
        return describe_ibm_edge(edge)
    elif "bupt" in dn_lower:
        return describe_bupt_edge(edge)
    elif "elliptic" in dn_lower:
        return describe_elliptic_edge(edge)
    else:
        raise ValueError(f"Unknown dataset name for edge description: {dataset_name}")
