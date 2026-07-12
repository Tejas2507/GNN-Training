import os
import json
import argparse
import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Please run: pip install sentence-transformers")

def prepare_class_embeddings(dataset_name, text_cache_dir, embeddings_dir, device=None):
    """
    Reads class_text.json, encodes the descriptions, and saves class_embeddings.npy.
    """
    dataset_text_dir = os.path.join(text_cache_dir, dataset_name)
    dataset_embed_dir = os.path.join(embeddings_dir, dataset_name)
    
    class_text_path = os.path.join(dataset_text_dir, "class_text.json")
    output_path = os.path.join(dataset_embed_dir, "class_embeddings.npy")
    
    if not os.path.exists(class_text_path):
        raise FileNotFoundError(f"Class text file not found at: {class_text_path}")
        
    print(f"Loading class texts from: {class_text_path}")
    with open(class_text_path, "r") as f:
        class_text_dict = json.load(f)
        
    # Sort keys as integers to preserve index ordering (0, 1, 2, ...)
    sorted_keys = sorted(class_text_dict.keys(), key=lambda x: int(x))
    texts = [class_text_dict[k] for k in sorted_keys]
    
    print(f"Detected {len(texts)} classes:")
    for k in sorted_keys:
        print(f"  Class {k}: '{class_text_dict[k]}'")
        
    # Initialize SentenceTransformer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer ('BAAI/bge-base-en-v1.5') on device: {device}...")
    
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    
    print("Encoding class descriptions...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved class embeddings of shape {embeddings.shape} to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Class Description Embeddings")
    parser.add_argument("--dataset", type=str, choices=["IBM_AML", "BUPT", "Elliptic", "all"], default="all",
                        help="Dataset name or 'all' to run on all datasets")
    parser.add_argument("--text_cache_dir", type=str, default="custom_pretrain/text_cache",
                        help="Path to text_cache root directory")
    parser.add_argument("--embeddings_dir", type=str, default="custom_pretrain/embeddings",
                        help="Path to embeddings root directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (cuda, cpu, mps, etc.)")
                        
    args = parser.parse_args()
    
    datasets_to_run = ["IBM_AML", "BUPT", "Elliptic"] if args.dataset == "all" else [args.dataset]
    
    for dset in datasets_to_run:
        print(f"Processing class embeddings for dataset: {dset}")
        try:
            prepare_class_embeddings(dset, args.text_cache_dir, args.embeddings_dir, args.device)
        except Exception as e:
            print(f"Error processing {dset}: {e}")
            print("Please ensure sentence-transformers is installed and text files exist.")
