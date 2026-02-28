import os
import json
import pandas as pd
from PIL import Image
import io

def extract_samples():
    from datasets import load_dataset
    
    parquet_path = "GeoThought/playground/data/geo_thought/Geo-Thought-6K.parquet"
    out_dir = "GeoThought/playground/data/"
    img_dir = os.path.join(out_dir, "images", "samples")
    jsonl_path = os.path.join(out_dir, "test_question.jsonl")
    
    os.makedirs(img_dir, exist_ok=True)
    
    print(f"Reading dataset...")
    # Read the parquet directly using HuggingFace datasets
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    
    with open(jsonl_path, "w") as f:
        # Iterate over all exact examples natively stored in the HuggingFace GeoThought split
        for idx in range(len(dataset)):
            item = dataset[idx]
            
            img_filename = f"sample_{idx}.jpg"
            img_filepath = os.path.join(img_dir, img_filename)
            
            # The 'images' column contains the PIL image lists
            # The row structure is: images[0] is the PIL element
            image_list = item["images"]
            pil_img = image_list[0] if isinstance(image_list, list) else image_list
            
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img.save(img_filepath, "JPEG")
            
            rel_img_path = os.path.join("images", "samples", img_filename)
            
            f.write(json.dumps({
                "image": rel_img_path,
                "text": item["problem"]
            }) + "\n")
            
            print(f"Extracted {rel_img_path}")
            
    print(f"\nWrote {len(dataset)} total examples to {jsonl_path}")

if __name__ == "__main__":
    extract_samples()
