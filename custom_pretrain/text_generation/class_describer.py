def describe_ibm_class(class_label: int) -> str:
    """Generates description for an IBM AML class label."""
    if class_label == 0:
        return "Legitimate bank account."
    elif class_label == 1:
        return "Money laundering account."
    else:
        return f"Unknown class label {class_label}."


def describe_elliptic_class(class_label: int) -> str:
    """Generates description for an Elliptic++ class label."""
    if class_label == 0:
        return "Legitimate bitcoin transaction."
    elif class_label == 1:
        return "Illicit bitcoin transaction."
    else:
        return f"Unknown class label {class_label}."


def describe_bupt_class(class_label: int) -> str:
    """Generates description for a BUPT class label."""
    # BUPT classes: 0, 1, 2, 3
    if class_label in [0, 1, 2, 3]:
        return f"Telecommunication class {class_label}."
    else:
        return f"Unknown class label {class_label}."


def get_class_description(class_label: int, dataset_name: str) -> str:
    """
    Dispatcher function to generate class descriptions based on dataset type.
    """
    dn_lower = dataset_name.lower().replace("-", "").replace("_", "").replace("+", "")
    if "ibm" in dn_lower:
        return describe_ibm_class(class_label)
    elif "bupt" in dn_lower:
        return describe_bupt_class(class_label)
    elif "elliptic" in dn_lower:
        return describe_elliptic_class(class_label)
    else:
        raise ValueError(f"Unknown dataset name for class description: {dataset_name}")
