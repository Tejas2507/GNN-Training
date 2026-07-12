import os
import json
import argparse
import numpy as np
import torch
import pickle

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Please run: pip install sentence-transformers")

def get_num_edges_from_graph(dataset_name, cache_output_dir):
    """Loads the FraudGraph pickle file and returns the number of edges."""
    graph_path = os.path.join(cache_output_dir, f"{dataset_name}_graph.pkl")
    if not os.path.exists(graph_path):
        # Fallback to loading BUPT_graph.pkl or similar
        graph_path = os.path.join(cache_output_dir, f"{dataset_name.replace('_', '')}_graph.pkl")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"FraudGraph pickle file not found at: {graph_path}")
            
    print(f"Loading FraudGraph metadata from: {graph_path} to count edges...")
    with open(graph_path, "rb") as f:
        # Import schema inside to unpickle correctly
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from custom_pretrain.schema.graph_schema import FraudGraph
        graph = pickle.load(f)
        return len(graph.edges)

def prepare_edge_embeddings(dataset_name, text_cache_dir, embeddings_dir, cache_output_dir, dummy=False, limit=None, device=None):
    """
    Reads edge_text.json (or uses dummy generation) and saves edge_embeddings.npy.
    """
    dataset_text_dir = os.path.join(text_cache_dir, dataset_name)
    dataset_embed_dir = os.path.join(embeddings_dir, dataset_name)
    
    edge_text_path = os.path.join(dataset_text_dir, "edge_text.json")
    output_path = os.path.join(dataset_embed_dir, "edge_embeddings.npy")
    
    # Check total edges first
    try:
        num_edges = get_num_edges_from_graph(dataset_name, cache_output_dir)
    except Exception as e:
        print(f"Warning: Could not load graph to count edges ({e}). Will attempt to infer from JSON length.")
        num_edges = None

    if dummy:
        if num_edges is None:
            # Attempt to count JSON lines/entries without loading everything in memory
            with open(edge_text_path, "r") as f:
                edge_text_list = json.load(f)
                num_edges = len(edge_text_list)
                
        print(f"Generating DUMMY zero edge embeddings for {num_edges} edges...")
        embeddings = np.zeros((num_edges, 768), dtype=np.float32)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, embeddings)
        print(f"Saved dummy edge embeddings of shape {embeddings.shape} to: {output_path}")
        print("=" * 60)
        return

    if not os.path.exists(edge_text_path):
        raise FileNotFoundError(f"Edge text file not found at: {edge_text_path}")
        
    print(f"Loading edge texts from: {edge_text_path}...")
    with open(edge_text_path, "r") as f:
        texts = json.load(f)
        
    if num_edges is not None and len(texts) != num_edges:
        print(f"Warning: JSON contains {len(texts)} texts but graph has {num_edges} edges. Aligning with JSON length.")
        num_edges = len(texts)
    else:
        num_edges = len(texts)

    if limit is not None:
        print(f"Limiting encoding to the first {limit} edges. The rest will be zero padded.")
        texts_to_encode = texts[:limit]
    else:
        texts_to_encode = texts

    # Initialize SentenceTransformer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer ('BAAI/bge-base-en-v1.5') on device: {device}...")
    
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    
    print(f"Encoding {len(texts_to_encode)} edge descriptions...")
    encoded = model.encode(texts_to_encode, batch_size=256, show_progress_bar=True)
    
    if limit is not None and limit < num_edges:
        # Pad remaining with zeros
        embeddings = np.zeros((num_edges, 768), dtype=np.float32)
        embeddings[:limit] = encoded
    else:
        embeddings = np.array(encoded, dtype=np.float32)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved edge embeddings of shape {embeddings.shape} to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Edge Description Embeddings")
    parser.add_argument("--dataset", type=str, choices=["IBM_AML", "BUPT", "Elliptic", "all"], default="all",
                        help="Dataset name or 'all' to run on all datasets")
    parser.add_argument("--text_cache_dir", type=str, default="custom_pretrain/text_cache",
                        help="Path to text_cache root directory")
    parser.add_argument("--embeddings_dir", type=str, default="custom_pretrain/embeddings",
                        help="Path to embeddings root directory")
    parser.add_argument("--cache_output_dir", type=str, default="custom_pretrain/cache_output",
                        help="Path to cache_output directory")
    parser.add_argument("--dummy", action="store_true",
                        help="Generate dummy zero embeddings instead of calling SentenceTransformer (extremely fast)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of edges to encode (remaining are zero padded)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (cuda, cpu, mps, etc.)")
                        
    args = parser.parse_args()
    
    datasets_to_run = ["IBM_AML", "BUPT", "Elliptic"] if args.dataset == "all" else [args.dataset]
    
    for dset in datasets_to_run:
        print(f"Processing edge embeddings for dataset: {dset}")
        try:
            # For IBM_AML, if not dummy and no limit is specified, warn that it will take long
            if dset == "IBM_AML" and not args.dummy and args.limit is None:
                print("WARNING: IBM_AML has 5 million edges. This will take a long time to encode.")
                print("If you want a quick run, use --dummy or --limit 1000")
                
            prepare_edge_embeddings(
                dset, args.text_cache_dir, args.embeddings_dir, 
                args.cache_output_dir, args.dummy, args.limit, args.device
            )
        except Exception as e:
            print(f"Error processing {dset}: {e}")
            import traceback
            traceback.print_exc()
