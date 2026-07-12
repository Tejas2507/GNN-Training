import os
import sys
import pickle
import json
import gc

# Ensure custom_pretrain path is in system path to import describers & graph_schema
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from custom_pretrain.schema.graph_schema import Node, Edge, FraudGraph
from custom_pretrain.text_generation.node_describer import get_node_description
from custom_pretrain.text_generation.edge_describer import get_edge_description
from custom_pretrain.text_generation.class_describer import get_class_description


def process_dataset(dataset_name: str, pickle_filename: str, output_subdir: str):
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache_output")
    text_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "text_cache", output_subdir)
    os.makedirs(text_cache_dir, exist_ok=True)
    
    pickle_path = os.path.join(cache_dir, pickle_filename)
    if not os.path.exists(pickle_path):
        print(f"Skipping {dataset_name} (pickle file not found at {pickle_path})")
        return
        
    print(f"\nProcessing text generation for {dataset_name}...")
    with open(pickle_path, "rb") as f:
        graph = pickle.load(f)
        
    # Stats trackers
    all_lengths = []
    longest_text = ""
    shortest_text = None
    
    # Helper to update length stats
    def track_text(text):
        nonlocal longest_text, shortest_text
        if not text:
            return
        length = len(text)
        all_lengths.append(length)
        if length > len(longest_text):
            longest_text = text
        if shortest_text is None or length < len(shortest_text):
            shortest_text = text

    # 1. Generate Class descriptions
    print("Generating class descriptions...")
    class_text = {}
    # Find all unique class labels
    labels = {node.label for node in graph.nodes.values() if node.label is not None}
    for label in sorted(labels):
        desc = get_class_description(label, dataset_name)
        class_text[str(label)] = desc
        track_text(desc)
        
    class_output_path = os.path.join(text_cache_dir, "class_text.json")
    with open(class_output_path, "w") as f_out:
        json.dump(class_text, f_out, indent=2)
    print(f"Saved class descriptions to {class_output_path} (count: {len(class_text)})")
    del class_text
    gc.collect()

    # 2. Generate Node descriptions
    print("Generating node descriptions...")
    node_text = {}
    for nid, node in graph.nodes.items():
        desc = get_node_description(node, dataset_name)
        node_text[nid] = desc
        track_text(desc)
        
    node_output_path = os.path.join(text_cache_dir, "node_text.json")
    with open(node_output_path, "w") as f_out:
        json.dump(node_text, f_out, indent=2)
    n_nodes_gen = len(node_text)
    print(f"Saved node descriptions to {node_output_path} (count: {n_nodes_gen})")
    del node_text
    gc.collect()

    # 3. Generate Edge descriptions
    print("Generating edge descriptions...")
    edge_text = []
    for edge in graph.edges:
        desc = get_edge_description(edge, dataset_name)
        edge_text.append(desc)
        track_text(desc)
        
    edge_output_path = os.path.join(text_cache_dir, "edge_text.json")
    with open(edge_output_path, "w") as f_out:
        json.dump(edge_text, f_out, indent=2)
    n_edges_gen = len(edge_text)
    print(f"Saved edge descriptions to {edge_output_path} (count: {n_edges_gen})")
    del edge_text
    gc.collect()

    # Print Validation Statistics
    avg_len = sum(all_lengths) / len(all_lengths) if all_lengths else 0
    shortest_len = len(shortest_text) if shortest_text else 0
    longest_len = len(longest_text)
    
    print("\n" + "="*50)
    print(f"TEXT GENERATION VALIDATION STATISTICS: {dataset_name}")
    print("="*50)
    print(f"Number of generated node descriptions  : {n_nodes_gen}")
    print(f"Number of generated edge descriptions  : {n_edges_gen}")
    print(f"Number of generated class descriptions  : {len(labels)}")
    print(f"Average text length                    : {avg_len:.2f} characters")
    print(f"Shortest text length                   : {shortest_len} characters")
    if shortest_text:
        print(f"Shortest text snippet                  : \"{shortest_text.replace(chr(10), ' ')}\"")
    print(f"Longest text length                    : {longest_len} characters")
    print("="*50 + "\n")
    
    # Completely clean up graph memory
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
