
import os
import json
import argparse
import numpy as np
import torch

from pathlib import Path
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--text_file", required=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

print("=" * 80)
print(f"Dataset : {args.dataset}")
print(f"GPU     : {args.gpu}")
print("=" * 80)

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)

# Short descriptions -> don't tokenize to 512
model.max_seq_length = 64

model.eval()
model.half()

EMBED_DIM = model.get_embedding_dimension()

ROOT = Path("custom_pretrain/embeddings")
OUT = ROOT / args.dataset
OUT.mkdir(parents=True, exist_ok=True)

emb_file = OUT / "node_embeddings.npy"
ids_file = OUT / "node_ids.json"
progress_file = OUT / "progress.json"

print("Loading text...")

with open(args.text_file, "r") as f:
    node_text = json.load(f)

node_ids = list(node_text.keys())
texts = list(node_text.values())

N = len(texts)

print(f"Nodes : {N:,}")

if not ids_file.exists():
    with open(ids_file, "w") as f:
        json.dump(node_ids, f)

start = 0

if progress_file.exists():
    with open(progress_file) as f:
        start = json.load(f)["completed"]

    print(f"Resuming from {start:,}")

emb = np.memmap(
    emb_file,
    dtype=np.float32,
    mode="r+" if emb_file.exists() else "w+",
    shape=(N, EMBED_DIM),
)

print("Embedding...")

for i in tqdm(range(start, N, args.batch_size)):

    batch = texts[i:i + args.batch_size]

    with torch.inference_mode():

        vec = model.encode(
            batch,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    emb[i:i + len(batch)] = vec
    emb.flush()

    with open(progress_file, "w") as f:
        json.dump(
            {
                "completed": i + len(batch),
                "total": N,
                "model": "BAAI/bge-base-en-v1.5",
            },
            f,
        )

print(f"\nFinished {args.dataset}")
