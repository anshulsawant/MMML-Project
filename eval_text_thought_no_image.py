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
from evaluate_generated import clean_base_model_ans, safe_math_eval, normalize

def e2e_evaluate():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation for LatentEuclid")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--decoder_weights", type=str, default=None, help="Path to decoder_epoch_X.pt (defaults to best experiment checkpoint)")
    parser.add_argument("--x_encoder_weights", type=str, default=None, help="Override path to X-encoder weights")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generating answers.")
    parser.add_argument("--split", type=float, default=0.9, help="Validation split index ratio (default: 0.9, i.e. last 10%)")
    parser.add_argument("--out", type=str, default="data/e2e_mismatches.json", help="Path to save mismatches")
    parser.add_argument("--end_to_end", action="store_true", help="Use train_end_to_end configuration instead of train_decoder")
    parser.add_argument("--zero_thoughts", action="store_true", help="Ablation: Force all predicted_latents to pure zeros")
    parser.add_argument("--zero_thought_indices", type=int, nargs="+", help="Ablation: Force specific predicted_latents indices (0 to k_steps-1) to pure zeros")
    args = parser.parse_args()

    device = args.device

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    active_block = "train_end_to_end" if args.end_to_end else "train_decoder"
    experiment_name = args.experiment_name_override or config.get(active_block, {}).get("experiment_name") or config.get("experiment", {}).get("name", "default")
    
    if args.out == "data/e2e_mismatches.json":
        suffix = ""
        if args.zero_thoughts:
            suffix = "_zero_all_thoughts"
        elif args.zero_thought_indices:
            suffix = f"_zero_thoughts_{'_'.join(map(str, args.zero_thought_indices))}"
        args.out = f"data/eval_{experiment_name}{suffix}.json"
        
    # Auto-resolve latest decoder weights for this experiment if missing
    # Auto-resolve latest decoder weights for this experiment if missing
    import os
    if args.decoder_weights is None:
        ckpt_dir = f"/workspace/checkpoints/{experiment_name}/decoder"
        if os.path.exists(ckpt_dir):
            if os.path.exists(os.path.join(ckpt_dir, "decoder_best.pt")):
                args.decoder_weights = os.path.join(ckpt_dir, "decoder_best.pt")
            else:
                def parse_cp_name(f):
                    base = f.split('epoch_')[1].split('.pt')[0]
                    if "_step_" in base:
                        ep, st = base.split("_step_")
                        return (int(ep), int(st))
                    return (int(base), 0)
                    
                checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("decoder_epoch_") and f.endswith(".pt")]
                if checkpoints:
                    checkpoints.sort(key=parse_cp_name)
                    args.decoder_weights = os.path.join(ckpt_dir, checkpoints[-1])
        if args.decoder_weights is None:
            print(f"Error: No decoder weights provided and none found in {ckpt_dir}")
            sys.exit(1)

    print("Loading X-Encoder...")
    x_encoder = LatentEuclid(
        base_model_id=config["model"]["base_model_id"],
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"]
    ).to(device)
    
    # Load VICReg weights
    weight_path = args.x_encoder_weights
    
    # 1. Primary: check if e2e weights were saved in this experiment (Method B)
    if not weight_path:
        e2e_weight_path = f"/workspace/checkpoints/{experiment_name}/decoder/x_encoder_e2e_best.pt"
        if os.path.exists(e2e_weight_path):
            weight_path = e2e_weight_path

    # 2. Secondary: X-encoder specific folder (Phase 3)
    if not weight_path:
        local_encoder_path = f"/workspace/checkpoints/{experiment_name}/x_encoder_best.pt"
        if os.path.exists(local_encoder_path):
            weight_path = local_encoder_path
            
    # 3. Fallback: check explicit override in config (Method A)
    if not weight_path:
        override_path = config.get(active_block, {}).get("x_encoder_weights_override")
        if override_path and os.path.exists(override_path):
            weight_path = override_path
        
    if weight_path and os.path.exists(weight_path):
        try:
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in state_dict:
                x_encoder.load_state_dict(state_dict["model_state_dict"])
            else:
                x_encoder.load_state_dict(state_dict)
            print(f"Loaded X-Encoder weights from {weight_path}")
        except Exception as e:
            print(f"Warning: Could not load X-encoder weights from {weight_path}: {e}. Evaluating with untrained projections.")
    else:
        print(f"Warning: Could not find X-encoder weights at {weight_path}. Evaluating with untrained projections.")
    x_encoder.eval()

    print("Loading Y-Decoder...")
    y_decoder = YDecoderPrefix(
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"],
        unfreeze_layers=config.get(active_block, {}).get("unfreeze_layers", 0),
        use_projection_mlp=config.get(active_block, {}).get("use_projection_mlp", True)
    ).to(device)
    
    # Load projection head / decoder weights
    try:
        decoder_state_dict = torch.load(args.decoder_weights, map_location="cpu")
        
        is_named_params = any(k.startswith("prefix_projection.") or k.startswith("decoder.") for k in decoder_state_dict)
        is_old_linear = "weight" in decoder_state_dict and "0.weight" not in decoder_state_dict and not is_named_params
        
        if is_old_linear:
            import torch.nn as nn
            print("Detected old-style single Linear layer checkpoint. Downgrading architecture for evaluation...")
            y_decoder.prefix_projection = nn.Linear(y_decoder.embedding_dim, y_decoder.embedding_dim).to(torch.bfloat16).to(device)
            y_decoder.prefix_projection.load_state_dict(decoder_state_dict)
        elif is_named_params:
            y_decoder.load_state_dict(decoder_state_dict, strict=False)
        else:
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
    results = []
    
    # Chunk the data into batches
    batches = [val_data[i:i + args.batch_size] for i in range(0, len(val_data), args.batch_size)]
    
    with torch.no_grad():
        pbar = tqdm(batches, desc="Evaluating Batches")
        for batch in pbar:
            images = []
            valid_items = []
            
            # 1. Load images dynamically inside batch
            for item in batch:
                img_path = item["image_path"]
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                    valid_items.append(item)
                except Exception as e:
                    print(f"Skipping {img_path} due to image load error: {e}")
            
            if not valid_items:
                continue
                
            batch_messages = []
            questions = []
            true_answers = []
            
            # 2. Extract valid items
            for item, img in zip(valid_items, images):
                question = item["question"]
                true_answer_raw = ground_truths.get(item["image_path"])
                
                if true_answer_raw is None:
                    continue
                    
                thought_string = "".join([f"<thought_{i+1}>" for i in range(config["model"]["k_steps"])])
                full_text = question + " " + thought_string
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_text}
                    ]
                }]
                batch_messages.append(messages)
                questions.append(question)
                true_answers.append((item["image_path"], true_answer_raw))
                
            if not batch_messages:
                continue
                
            rendered_texts = [x_encoder.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
            
            # The processor automatically pads dynamic resolution images across the batch 
            inputs = x_encoder.processor(text=rendered_texts, padding=True, return_tensors="pt").to(device)
            
            predicted_latents = x_encoder(
                input_ids=inputs.input_ids, 
                pixel_values=inputs.get("pixel_values"), 
                attention_mask=inputs.attention_mask,
                image_grid_thw=inputs.get("image_grid_thw")
            )
            
            if args.zero_thoughts:
                predicted_latents = torch.zeros_like(predicted_latents)
            elif args.zero_thought_indices:
                for idx in args.zero_thought_indices:
                    if 0 <= idx < predicted_latents.shape[1]:
                        predicted_latents[:, idx, :] = 0.0
            
            # Format clean prompts for y-decoder, completely starving it of the image pixels
            # The decoder sees the text question, the <thought>s, but NOT the raw images!
            prompts = []
            for t in questions:
                clean_t = t
                for i in range(config["model"]["k_steps"]):
                    clean_t = clean_t.replace(f"<thought_{i+1}>", "")
                prompts.append(clean_t.strip() + "\nAnswer: ")
            
            # 2. Forward pass through Y-Decoder to generate answers
            generated_texts = y_decoder.generate(
                predicted_latents=predicted_latents, 
                text_prompts=prompts,
                max_new_tokens=15
            )
            
            # 3. Calculate metrics per generation
            for gen_text, (img_path, true_answer_raw), q in zip(generated_texts, true_answers, questions):
                pred_raw = clean_base_model_ans(gen_text)
                gt_norm = normalize(true_answer_raw)
                pred_norm = normalize(pred_raw)
                
                is_correct = (gt_norm == pred_norm)
                
                if not is_correct:
                    gt_val = safe_math_eval(true_answer_raw)
                    pred_val = safe_math_eval(pred_raw)
                    if gt_val is not None and pred_val is not None:
                        is_correct = math.isclose(gt_val, pred_val, rel_tol=1e-3, abs_tol=0.06)
                        
                total += 1
                if is_correct:
                    correct += 1
                    
                results.append({
                    'image': img_path,
                    'is_correct': is_correct,
                    'gt_raw': str(true_answer_raw),
                    'pred_raw': pred_raw,
                    'gt_norm': gt_norm,
                    'pred_norm': pred_norm,
                    'model_generation': gen_text.strip()
                })
                    
            if total > 0:
                acc_so_far = correct / total * 100
                pbar.set_postfix({"Correct": f"{correct}/{total}", "Accuracy": f"{acc_so_far:.2f}%"})
                
            # Periodically save results mid-run so they can be inspected if cancelled early
            if results:
                with open(args.out, 'w') as f:
                    json.dump(results, f, indent=2)
                
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"\n+++ RESULTS +++")
    print(f"Evaluated {total} samples.")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%")
    
    if results:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} evaluated samples to {args.out}")

if __name__ == "__main__":
    e2e_evaluate()
