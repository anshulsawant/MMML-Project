import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.evaluate_generated import clean_base_model_ans, normalize, safe_math_eval
from models.latent_euclid import LatentEuclid
from models.y_decoder_prefix import YDecoderPrefix


def e2e_evaluate():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation for LatentEuclid")
    parser.add_argument("--config", type=str, default="configs/v12_cod.yaml")
    parser.add_argument("--decoder_weights", type=str, default=None, help="Path to decoder_epoch_X.pt (defaults to best experiment checkpoint)")
    parser.add_argument("--x_encoder_weights", type=str, default=None, help="Override path to X-encoder weights")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generating answers.")
    parser.add_argument("--split", type=float, default=0.9, help="Validation split index ratio (default: 0.9, i.e. last 10%)")
    parser.add_argument("--seed", type=int, default=42, help="Seed used when falling back to a shuffled holdout split.")
    parser.add_argument("--out", type=str, default="data/e2e_mismatches.json", help="Path to save mismatches")
    parser.add_argument("--experiment_name_override", type=str, default=None, help="Override experiment name")
    parser.add_argument("--force_k_steps", type=int, default=None, help="Override dynamic halt to evaluate exactly K steps")
    args = parser.parse_args()

    device = args.device

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    active_block = "train_decoder"
    experiment_name = args.experiment_name_override or config.get(active_block, {}).get("experiment_name") or config.get("experiment", {}).get("name", "default")

    if args.out == "data/e2e_mismatches.json":
        args.out = f"data/eval_{experiment_name}.json"

    if args.decoder_weights is None:
        ckpt_dir = f"/workspace/checkpoints/{experiment_name}/decoder"
        if os.path.exists(ckpt_dir):
            best_checkpoint = os.path.join(ckpt_dir, "decoder_best.pt")
            if os.path.exists(best_checkpoint):
                args.decoder_weights = best_checkpoint
            else:
                def parse_cp_name(filename):
                    base = filename.split("epoch_")[1].split(".pt")[0]
                    if "_step_" in base:
                        epoch, step = base.split("_step_")
                        return (int(epoch), int(step))
                    return (int(base), 0)

                checkpoints = [name for name in os.listdir(ckpt_dir) if name.startswith("decoder_epoch_") and name.endswith(".pt")]
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
        max_thought_tokens=config["model"].get("max_thought_tokens", 30),
    ).to(device)

    weight_path = args.x_encoder_weights
    if not weight_path:
        e2e_weight_path = f"/workspace/checkpoints/{experiment_name}/decoder/x_encoder_e2e_best.pt"
        if os.path.exists(e2e_weight_path):
            weight_path = e2e_weight_path

    if not weight_path:
        local_encoder_path = f"/workspace/checkpoints/{experiment_name}/x_encoder_best.pt"
        if os.path.exists(local_encoder_path):
            weight_path = local_encoder_path

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
        except Exception as exc:
            print(f"Warning: Could not load X-encoder weights from {weight_path}: {exc}. Evaluating with untrained projections.")
    else:
        print(f"Warning: Could not find X-encoder weights at {weight_path}. Evaluating with untrained projections.")
    x_encoder.eval()

    print("Loading Y-Decoder...")
    y_decoder = YDecoderPrefix(
        target_model_id=config["model"]["decoder_base_model_id"],
        unfreeze_layers=config.get(active_block, {}).get("unfreeze_layers", 0),
        use_projection_mlp=config.get(active_block, {}).get("use_projection_mlp", True),
    ).to(device)

    try:
        decoder_state_dict = torch.load(args.decoder_weights, map_location="cpu")

        is_named_params = any(key.startswith("prefix_projection.") or key.startswith("decoder.") for key in decoder_state_dict)
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
    except Exception as exc:
        print(f"Warning: Could not load Y-decoder weights from {args.decoder_weights}: {exc}")
    y_decoder.eval()

    print("Loading Validation Dataset...")
    try:
        with open(config["data"]["jsonl_path"], "r") as f:
            full_data = [json.loads(line) for line in f]
    except Exception as exc:
        print(f"Error loading {config['data']['jsonl_path']}: {exc}")
        return

    try:
        with open("data/ground_truths.json", "r") as f:
            ground_truths = json.load(f)
    except Exception as exc:
        print(f"Error loading ground_truths.json: {exc}")
        return

    try:
        with open("data/v4_split_keys.json", "r") as vf:
            v4_splits = json.load(vf)
        v4_val_keys = set(v4_splits["val_keys"])
        val_data = [item for item in full_data if os.path.basename(item["image_path"]) in v4_val_keys]
        print(f"Loaded {len(val_data)} validation samples explicitly matching V4 split structure natively.")
    except Exception as exc:
        print(f"Fallback to linear slice: {exc}")
        random.seed(args.seed)
        random.shuffle(full_data)
        split_idx = int(args.split * len(full_data))
        val_data = full_data[split_idx:]

    if args.limit:
        val_data = val_data[:args.limit]

    print(f"\n================ END-TO-END EVALUATION ({len(val_data)} samples) ================\n")

    correct = 0
    total = 0
    results = []
    batches = [val_data[index : index + args.batch_size] for index in range(0, len(val_data), args.batch_size)]

    with torch.no_grad():
        pbar = tqdm(batches, desc="Evaluating Batches")
        for batch in pbar:
            batch_images = []
            batch_messages = []
            questions = []
            true_answers = []

            for item in batch:
                img_path = item["image_path"]
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as exc:
                    print(f"Skipping {img_path} due to image load error: {exc}")
                    continue

                true_answer_raw = ground_truths.get(img_path)
                if true_answer_raw is None:
                    continue
                question = item["question"]
                thought_string = "".join([f"<thought_{index + 1}>" for index in range(config["model"]["k_steps"])])
                full_text = question + " " + thought_string

                batch_images.append(image)
                batch_messages.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": full_text},
                        ],
                    }
                ])
                questions.append(question)
                true_answers.append((img_path, true_answer_raw))

            if not batch_messages:
                continue
            rendered_texts = [x_encoder.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in batch_messages]
            inputs = x_encoder.processor(text=rendered_texts, images=batch_images, padding=True, return_tensors="pt").to(device)
            predicted_latents = x_encoder(
                input_ids=inputs.input_ids,
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.attention_mask,
                image_grid_thw=inputs.get("image_grid_thw"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
            )

            prompts = [question.strip() + "\nAnswer: " for question in questions]
            generated_texts = y_decoder.generate(predicted_latents=predicted_latents, text_prompts=prompts, max_new_tokens=15)

            for gen_text, (img_path, true_answer_raw) in zip(generated_texts, true_answers):
                pred_raw = clean_base_model_ans(gen_text)
                gt_norm = normalize(true_answer_raw)
                pred_norm = normalize(pred_raw)

                is_correct = gt_norm == pred_norm
                if not is_correct:
                    gt_val = safe_math_eval(true_answer_raw)
                    pred_val = safe_math_eval(pred_raw)
                    if gt_val is not None and pred_val is not None:
                        is_correct = math.isclose(gt_val, pred_val, rel_tol=1e-3, abs_tol=0.06)

                total += 1
                if is_correct:
                    correct += 1

                results.append(
                    {
                        "image": img_path,
                        "is_correct": is_correct,
                        "gt_raw": str(true_answer_raw),
                        "pred_raw": pred_raw,
                        "gt_norm": gt_norm,
                        "pred_norm": pred_norm,
                        "model_generation": gen_text.strip(),
                    }
                )

            if total > 0:
                acc_so_far = correct / total * 100
                pbar.set_postfix({"Correct": f"{correct}/{total}", "Accuracy": f"{acc_so_far:.2f}%"})

            if results:
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=2)

    acc = correct / total * 100 if total > 0 else 0.0
    print("\n+++ RESULTS +++")
    print(f"Evaluated {total} samples.")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%")

    if results:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} evaluated samples to {args.out}")


if __name__ == "__main__":
    e2e_evaluate()