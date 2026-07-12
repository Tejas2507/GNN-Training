from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set
from collections import Counter

@dataclass
class Node:
    """
    Represents a node in the fraud graph.

    Attributes:
        id (str): Unique identifier for the node.
        type (str): Type of the node (e.g., 'phone_number', 'bank_account', 'bitcoin_wallet').
        features (dict): Dictionary containing numerical or categorical attributes of the node.
        text (str): Human-readable text description of the node. Defaults to an empty string.
        label (Optional[int]): Fraud class label if available (e.g., 0 for licit, 1 for illicit). Defaults to None.
    """
    id: str
    type: str
    features: dict = field(default_factory=dict)
    text: str = ""
    label: Optional[int] = None


@dataclass
class Edge:
    """
    Represents a directed or undirected edge in the fraud graph.

    Attributes:
        src (str): Source node identifier.
        dst (str): Destination node identifier.
        edge_type (str): Type of the edge (e.g., 'call', 'sms', 'transfer', 'bitcoin_transaction').
        weight (float): Edge weight or transaction amount. Defaults to 1.0.
        timestamp (Optional[str]): Timestamp of the edge/transaction. Defaults to None.
        features (dict): Dictionary containing numerical or categorical attributes of the edge.
    """
    src: str
    dst: str
    edge_type: str
    weight: float = 1.0
    timestamp: Optional[str] = None
    features: dict = field(default_factory=dict)


@dataclass
class FraudGraph:
    """
    Represents a universal fraud graph containing nodes and edges.

    Attributes:
        name (str): Name of the dataset/graph.
        nodes (Dict[str, Node]): Dictionary mapping node IDs to Node objects.
        edges (List[Edge]): List of Edge objects in the graph.
        num_classes (int): Number of unique fraud classes. Defaults to 0.
        metadata (dict): Metadata dictionary for additional dataset-level info.
    """
    name: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    num_classes: int = 0
    metadata: dict = field(default_factory=dict)
    
    # Internal set to track if duplicate nodes were attempted to be added
    _duplicate_node_ids: Set[str] = field(default_factory=set, init=False, repr=False)

    def add_node(self, node: Node) -> None:
        """
        Adds a node to the graph. Keeps track of duplicate additions for validation.
        """
        if node.id in self.nodes:
            self._duplicate_node_ids.add(node.id)
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """
        Adds an edge to the graph.
        """
        self.edges.append(edge)

    def num_nodes(self) -> int:
        """
        Returns the number of unique nodes in the graph.
        """
        return len(self.nodes)

    def num_edges(self) -> int:
        """
        Returns the number of edges in the graph.
        """
        return len(self.edges)

    def summary(self) -> None:
        """
        Prints a detailed summary of the graph structure and statistics.
        """
        node_types = Counter(node.type for node in self.nodes.values())
        edge_types = Counter(edge.edge_type for edge in self.edges)
        labels = Counter(node.label for node in self.nodes.values() if node.label is not None)
        
        n_nodes = self.num_nodes()
        n_edges = self.num_edges()
        avg_degree = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0

        print("=========================================")
        print("Graph Summary")
        print("=========================================")
        print(f"Dataset        : {self.name}")
        print(f"Nodes          : {n_nodes}")
        print(f"Edges          : {n_edges}")
        print(f"Node Types     : {dict(node_types)}")
        print(f"Edge Types     : {dict(edge_types)}")
        print(f"Classes        : {dict(labels)}")
        print(f"Average Degree : {avg_degree:.4f}")
        print("=========================================")

    def validate(self) -> bool:
        """
        Validates the graph's integrity and prints diagnostics.
        
        Checks for:
        - Duplicate node IDs (attempted additions)
        - Edges pointing to missing nodes
        - Nodes missing labels (prints a warning count)
        - Class distribution
        - Node types distribution
        - Edge types distribution
        - Average degree
        - Isolated nodes count
        
        Returns:
            bool: True if the graph is valid (no missing nodes or critical duplicate ID errors), False otherwise.
        """
        is_valid = True
        print("=========================================")
        print(f"Validating FraudGraph: {self.name}")
        print("=========================================")

        # 1. Check duplicate node IDs
        if self._duplicate_node_ids:
            print(f"[WARNING] Duplicate node addition attempts detected for IDs: {list(self._duplicate_node_ids)[:10]}")
            # Attempted duplicate node ID additions is a warning but does not fail validation unless strict duplicates are forbidden.
            # Here we keep it as True but warn.

        # 2. Check edges pointing to missing nodes
        missing_nodes = set()
        for edge in self.edges:
            if edge.src not in self.nodes:
                missing_nodes.add(edge.src)
            if edge.dst not in self.nodes:
                missing_nodes.add(edge.dst)
        
        if missing_nodes:
            print(f"[ERROR] Found {len(missing_nodes)} edges pointing to missing nodes!")
            print(f"Sample missing node IDs: {list(missing_nodes)[:10]}")
            is_valid = False
        else:
            print("[OK] All edges point to existing nodes.")

        # 3. Check missing labels
        nodes_with_missing_labels = [nid for nid, node in self.nodes.items() if node.label is None]
        if nodes_with_missing_labels:
            print(f"[INFO] {len(nodes_with_missing_labels)} nodes are missing labels (label is None).")
        else:
            print("[OK] All nodes have labels.")

        # 4. Print distributions
        node_types = Counter(node.type for node in self.nodes.values())
        print(f"Node Types Distribution: {dict(node_types)}")

        edge_types = Counter(edge.edge_type for edge in self.edges)
        print(f"Edge Types Distribution: {dict(edge_types)}")

        labels = Counter(node.label for node in self.nodes.values() if node.label is not None)
        print(f"Class Distribution: {dict(labels)}")

        # 5. Average Degree
        n_nodes = self.num_nodes()
        n_edges = self.num_edges()
        avg_degree = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0
        print(f"Average Degree: {avg_degree:.4f}")

        # 6. Isolated nodes
        active_nodes = set()
        for edge in self.edges:
            active_nodes.add(edge.src)
            active_nodes.add(edge.dst)
        
        isolated_nodes = set(self.nodes.keys()) - active_nodes
        print(f"Isolated Nodes Count: {len(isolated_nodes)}")
        if isolated_nodes:
            print(f"Sample isolated node IDs: {list(isolated_nodes)[:10]}")

        print("-----------------------------------------")
        if is_valid:
            print("[RESULT] Graph validation PASSED.")
        else:
            print("[RESULT] Graph validation FAILED.")
        print("=========================================")

        return is_valid
