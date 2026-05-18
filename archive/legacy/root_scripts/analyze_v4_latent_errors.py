import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import yaml
import sys

# Ensure MMML-Project is in path if running from within it
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.latent_euclid import LatentEuclid

def main():
    # 1. Load V4 Eval json
    with open("/workspace/MMML-Project/data/eval_v4_projection_and_unfrozen_layers.json", "r") as f:
        v4_eval = json.load(f)
        
    print(f"Loaded {len(v4_eval)} V4 evaluations.")
    
    # 2. Load Config
    with open("/workspace/MMML-Project/training/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    
    # Initialize the full LatentEuclid wrapper
    latent_euclid = LatentEuclid(
        base_model_id=config["model"]["base_model_id"],
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"]
    ).to(device)
    
    # Load V2 weights because V4 used the perfectly frozen V2 vision alignment
    weight_path = "/workspace/checkpoints/v2_huber_mean_pooled/x_encoder_best.pt"
    print(f"Loading X-Encoder weights from {weight_path}")
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        latent_euclid.load_state_dict(state_dict["model_state_dict"])
    else:
        latent_euclid.load_state_dict(state_dict)
    latent_euclid.eval()
    
    import numpy as np
    from PIL import Image
    
    # Build image_path to index mapping
    image_to_idx = {}
    image_to_question = {}
    with open("/workspace/MMML-Project/data/geothoughts_verified.jsonl", "r") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            # Normalize path securely using basename
            img_path_key = os.path.basename(item["image_path"])
            image_to_idx[img_path_key] = idx
            image_to_question[img_path_key] = item["question"]
            
    correct_mses = []
    correct_coss = []
    failed_mses = []
    failed_coss = []
    
    with torch.no_grad():
        for item in tqdm(v4_eval, desc="Computing Prediction Latent Errors"):
            raw_img_path = item["image"] # e.g. "data/playground/..."
            img_path = "/workspace/MMML-Project/" + raw_img_path
            is_correct = item["is_correct"]
            
            img_path_key = os.path.basename(raw_img_path)
            if img_path_key not in image_to_idx:
                continue
                
            idx = image_to_idx[img_path_key]
            
            # Load True Target Tensor
            target_path = f"/workspace/target_tensors/target_tensors_v2_huber_mean_pooled/problem_{idx}_targets.pt"
            
            if not os.path.exists(target_path):
                continue
                
            target_tensor = torch.load(target_path, map_location=device, weights_only=True) # [K, 3584]
            target_tensor = target_tensor.unsqueeze(0) # [1, K, 3584]
            
            # Run LatentEuclid to get Predicted Latents
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                continue
            
            # We append the exact same prompt sequence it was trained on
            question = image_to_question[img_path_key]
            thought_string = "".join([f"<thought_{i+1}>" for i in range(config["model"]["k_steps"])])
            full_prompt = question + " " + thought_string
            msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": full_prompt}]}]
            text_prompt = latent_euclid.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inputs = latent_euclid.processor(text=[text_prompt], images=[img], padding=True, return_tensors="pt").to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_latents = latent_euclid(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw")
                ) # [1, K, 3584]
            
            # Compute MSE and Cosine
            mse = F.mse_loss(pred_latents.float(), target_tensor.float()).item()
            cos = F.cosine_similarity(pred_latents.float(), target_tensor.float(), dim=2).mean().item()
            
            if is_correct:
                correct_mses.append(mse)
                correct_coss.append(cos)
            else:
                failed_mses.append(mse)
                failed_coss.append(cos)
                
    print("\n" + "="*50)
    print("=== Latent Vector Prediction Error Analysis ===")
    print("="*50)
    print(f"Correct Samples ({len(correct_mses)}):")
    print(f"  Avg MSE Loss:   {np.mean(correct_mses):.4f}")
    print(f"  Avg Cosine Sim: {np.mean(correct_coss):.4f}")
    
    print(f"\nFailed Samples ({len(failed_mses)}):")
    print(f"  Avg MSE Loss:   {np.mean(failed_mses):.4f}")
    print(f"  Avg Cosine Sim: {np.mean(failed_coss):.4f}")
    print("\nConclusion: If the distance metrics between Failed and Correct samples are statically identical, the fault lies entirely with the Language Model decoding capacity, rather than the Image Processor generating visually faulty geometries.")

if __name__ == "__main__":
    main()
