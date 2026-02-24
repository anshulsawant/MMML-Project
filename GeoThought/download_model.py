import argparse
from huggingface_hub import snapshot_download
import os

def download_model(model_id, local_dir):
    print(f"Downloading model: {model_id} to {local_dir}")
    try:
        snapshot_download(repo_id=model_id, local_dir=local_dir)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model from Hugging Face")
    parser.add_argument("--model_id", type=str, default="xinlingdedeng/InternVL3-8B-10834", help="Hugging Face model ID")
    parser.add_argument("--local_dir", type=str, default="./models/InternVL3-8B-10834", help="Local directory to save the model")
    
    args = parser.parse_args()
    
    # Ensure directory exists
    os.makedirs(args.local_dir, exist_ok=True)
    
    download_model(args.model_id, args.local_dir)
