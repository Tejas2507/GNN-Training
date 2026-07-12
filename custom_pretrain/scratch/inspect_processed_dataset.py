import os
import zipfile
import urllib.request
import torch

# Ensure torch_geometric is available
try:
    import torch_geometric as pyg
    from torch_geometric.data import Data
except ImportError:
    print("Installing torch-geometric...")
    os.system("pip install torch-geometric -q")
    import torch_geometric as pyg
    from torch_geometric.data import Data

def download_and_extract():
    zip_path = "ogbn_arxiv.zip"
    pt_path = "geometric_data_processed.pt"
    
    if not os.path.exists(pt_path):
        if not os.path.exists(zip_path):
            print("Downloading ogbn_arxiv.zip from Hugging Face...")
            url = "https://huggingface.co/datasets/zkchen/OGBArxiv/resolve/main/ogbn_arxiv.zip"
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete.")
            
        print("Extracting geometric_data_processed.pt from zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            member = "content/arxiv_node/ogbn_arxiv/processed/geometric_data_processed.pt"
            zip_ref.extract(member, path="extracted_temp")
            os.rename(os.path.join("extracted_temp", member), pt_path)
            print("Extraction complete.")
            
    return pt_path

def inspect_and_report(pt_path, report_path):
    report_lines = []
    def log(msg):
        print(msg)
        report_lines.append(msg)
        
    log("=" * 60)
    log("OFFICIAL DATASET INSPECTION REPORT")
    log("=" * 60)
    
    # Check torch.load structure
    loaded_obj = torch.load(pt_path, map_location="cpu")
    log(f"1. Structure returned by torch.load('{pt_path}'):")
    log(f"   Python type: {type(loaded_obj)}")
    if isinstance(loaded_obj, list):
        log(f"   Structure is a list of length: {len(loaded_obj)}")
        data = loaded_obj[0]
    elif isinstance(loaded_obj, tuple):
        log(f"   Structure is a tuple of length: {len(loaded_obj)}")
        data = loaded_obj[0]
    else:
        log(f"   Structure is direct Data object")
        data = loaded_obj
        
    log(f"   Extracted object type: {type(data)}")
    log(f"   Is instance of pyg.data.Data: {isinstance(data, Data)}")
    log("-" * 60)
    
    log("2. Fields and Tensors Breakdown:")
    log(f"   Keys/attributes: {list(data.keys())}")
    log("-" * 60)
    
    # Force num_nodes and num_edges calculation if not set
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else None
    num_edges = data.num_edges if hasattr(data, 'num_edges') else None
    
    for key in sorted(data.keys()):
        val = getattr(data, key)
        log(f"Attribute Name: {key}")
        log(f"  Type: {type(val)}")
        
        if isinstance(val, torch.Tensor):
            log(f"  Dtype: {val.dtype}")
            log(f"  Shape: {list(val.shape)}")
            
            # Min/Max if numeric
            if val.numel() > 0 and val.dtype in [torch.float32, torch.float64, torch.int32, torch.int64, torch.uint8, torch.bool]:
                if val.dtype == torch.bool:
                    log(f"  Min/Max: {val.any().item()} / {val.all().item()}")
                else:
                    log(f"  Min/Max: {val.min().item()} / {val.max().item()}")
                    
            # First few values
            if val.ndim == 0:
                log(f"  First values: {val.item()}")
            elif val.ndim == 1:
                log(f"  First values: {val[:5].tolist()}")
            elif val.ndim == 2:
                log(f"  First values (top 3 rows):")
                for row in val[:3].tolist():
                    # For long float vectors, show only first 5 elements of each row
                    log(f"    {row[:5]}... (length {len(row)})")
            else:
                log(f"  First values: {val[:3]}")
                
            # If we don't have num_nodes or num_edges yet, we can infer from shapes
            if num_nodes is None and key == 'x':
                num_nodes = val.shape[0]
            elif num_nodes is None and key == 'node_text_feat':
                num_nodes = val.shape[0]
            elif num_edges is None and key == 'edge_index':
                num_edges = val.shape[1]
        else:
            try:
                log(f"  Length: {len(val)}")
            except:
                pass
            log(f"  First values: {str(val)[:200]}")
            
        log("-" * 60)
        
    # Print high level stats
    log("3. Graph Statistics:")
    log(f"   Total Nodes: {num_nodes}")
    log(f"   Total Edges: {num_edges}")
    log("=" * 60)
    
    # Save report
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    pt_path = download_and_extract()
    inspect_and_report(pt_path, "processed_dataset_report.txt")
