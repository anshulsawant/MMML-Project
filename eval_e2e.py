import argparse
import yaml
import json
import torch
from PIL import Image
import math
import random
from tqdm import tqdm
import sys

from models.latent_euclid import LatentEuclid
from models.y_decoder_prefix import YDecoderPrefix

# Import robust evaluation metrics
sys.path.append('data')
from evaluate_generated import clean_gen_ans, safe_math_eval, normalize

def e2e_evaluate():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation for LatentEuclid")
    parser.add_argument("--config", type=str, default="training/config_decoder.yaml")
    parser.add_argument("--decoder_weights", type=str, required=True, help="Path to decoder_epoch_X.pt")
    parser.add_argument("--x_encoder_weights", type=str, default=None, help="Override path to X-encoder weights")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", type=float, default=0.9, help="Validation split index ratio (default: 0.9, i.e. last 10%)")
    parser.add_argument("--out", type=str, default="data/e2e_mismatches.json", help="Path to save mismatches")
    args = parser.parse_args()

    device = args.device

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("Loading X-Encoder...")
    x_encoder = LatentEuclid(
        base_model_id=config["model"]["base_model_id"],
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"]
    ).to(device)
    
    # Load VICReg weights
    weight_path = args.x_encoder_weights if args.x_encoder_weights else config["model"].get("x_encoder_weights")
    if weight_path and torch.cuda.is_available() or '{' not in weight_path:
        try:
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
            x_encoder.load_state_dict(state_dict["model_state_dict"])
            print(f"Loaded X-Encoder weights from {weight_path}")
        except Exception as e:
            print(f"Warning: Could not load X-encoder weights from {weight_path}: {e}. Evaluating with untrained projections.")
    x_encoder.eval()

    print("Loading Y-Decoder...")
    y_decoder = YDecoderPrefix(
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"]
    ).to(device)
    
    # Load projection head weights
    try:
        decoder_state_dict = torch.load(args.decoder_weights, map_location="cpu")
        
        # Backward compatibility: Check if the saved checkpoint is from before the MLP upgrade
        if "weight" in decoder_state_dict and "0.weight" not in decoder_state_dict:
            import torch.nn as nn
            print("Detected old-style single Linear layer checkpoint. Downgrading architecture for evaluation...")
            y_decoder.prefix_projection = nn.Linear(y_decoder.embedding_dim, y_decoder.embedding_dim).to(torch.bfloat16).to(device)
            
        y_decoder.prefix_projection.load_state_dict(decoder_state_dict)
        print(f"Loaded Y-Decoder weights from {args.decoder_weights}")
    except Exception as e:
        print(f"Warning: Could not load Y-decoder weights from {args.decoder_weights}: {e}")
    y_decoder.eval()

    print("Loading Validation Dataset...")
    full_data = []
    try:
        with open(config["data"]["jsonl_path"], 'r') as f:
            full_data = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading {config['data']['jsonl_path']}: {e}")
        return
        
    ground_truths = {}
    try:
        with open("data/ground_truths.json", 'r') as f:
            ground_truths = json.load(f)
    except Exception as e:
        print(f"Error loading ground_truths.json: {e}")
        return

    random.seed(42)  # Maintain identical split to validate_generation.py and training loops
    random.shuffle(full_data)
    split_idx = int(args.split * len(full_data))
    val_data = full_data[split_idx:]
    
    if args.limit:
        val_data = val_data[:args.limit]

    print(f"\n================ END-TO-END EVALUATION ({len(val_data)} samples) ================\n")
    
    correct = 0
    total = 0
    mismatches = []
    
    with torch.no_grad():
        for item in tqdm(val_data, desc="Evaluating"):
            img_path = item["image_path"]
            try:
                # Load image for processing
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Skipping {img_path} due to image load error: {e}")
                continue
                
            question = item["question"]
            true_answer_raw = ground_truths.get(img_path)
            
            if true_answer_raw is None:
                continue
            
            # Format X-encoder input - LatentEuclid natively expects the `<thought>` tokens appended 
            # to the end of the text sequence so it can extract the final hidden representations from those exact positions.
            thought_string = "".join([f"<thought_{i+1}>" for i in range(config["model"]["k_steps"])])
            full_text = question + " " + thought_string
            
            # Pass raw image and constructed text through the native VLM processor
            inputs = x_encoder.processor(text=[full_text], images=[image], padding=True, return_tensors="pt").to(device)
            
            # Get continuous vectors
            predicted_latents = x_encoder(
                input_ids=inputs.input_ids, 
                pixel_values=inputs.get("pixel_values"), 
                attention_mask=inputs.attention_mask,
                image_grid_thw=inputs.get("image_grid_thw")
            )
            
            # Format Y-decoder clean prompt
            # Auto-regressively generate the textual answer
            generated_text = y_decoder.generate(
                predicted_latents=predicted_latents, 
                text_prompts=[question.strip() + "\nAnswer: "],
                max_new_tokens=15
            )[0]
            
            pred_raw = clean_gen_ans(generated_text)
            
            gt_norm = normalize(true_answer_raw)
            pred_norm = normalize(pred_raw)
            
            is_correct = (gt_norm == pred_norm)
            
            # Math fallback test
            if not is_correct:
                gt_val = safe_math_eval(true_answer_raw)
                pred_val = safe_math_eval(pred_raw)
                if gt_val is not None and pred_val is not None:
                    is_correct = math.isclose(gt_val, pred_val, rel_tol=1e-3, abs_tol=0.06)
            
            total += 1
            if is_correct:
                correct += 1
            else:
                mismatches.append({
                    'image': img_path,
                    'gt_raw': str(true_answer_raw),
                    'pred_raw': pred_raw,
                    'gt_norm': gt_norm,
                    'pred_norm': pred_norm,
                    'model_generation': generated_text.strip()
                })
                
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"\n+++ RESULTS +++")
    print(f"Evaluated {total} samples.")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%")
    
    if mismatches:
        with open(args.out, 'w') as f:
            json.dump(mismatches, f, indent=2)
        print(f"Saved {len(mismatches)} mismatches to {args.out}")

if __name__ == "__main__":
    e2e_evaluate()
