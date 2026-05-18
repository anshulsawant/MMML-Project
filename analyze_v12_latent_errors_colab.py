import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from huggingface_hub import hf_hub_download

# Ensure MMML-Project root is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.latent_euclid import LatentEuclid


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze latent errors for v12_manifold_anchor_cod (Colab-ready).")
    parser.add_argument("--config", type=str, default="configs/v12_cod.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--eval_json",
        type=str,
        default="data/eval_v12_manifold_anchor_cod.json",
        help="Evaluation JSON produced by eval/e2e.py",
    )
    parser.add_argument(
        "--dataset_jsonl",
        type=str,
        default="data/geothoughts_verified.jsonl",
        help="JSONL used for question/image index mapping",
    )
    parser.add_argument(
        "--target_tensors_dir",
        type=str,
        default="/content/target_tensors/target_tensors_v12_manifold_anchor_cod",
        help="Directory containing problem_{idx}_targets.pt",
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=".",
        help="Project root used to resolve relative image paths",
    )
    parser.add_argument("--hf_repo", type=str, default="shilinm/latent_euclid", help="HF model repo id")
    parser.add_argument(
        "--hf_weight_path",
        type=str,
        default="x_encoder/v12_manifold_anchor_cod/x_encoder_best.pt",
        help="Path in HF repo to x-encoder checkpoint",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/content/.cache/huggingface",
        help="HF cache dir",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional path to save aggregated latent MSE/cosine metrics as JSON.",
    )
    return parser.parse_args()


def resolve_image_path(raw_img_path: str, repo_root: str) -> str | None:
    candidates = [
        raw_img_path,
        os.path.join(repo_root, raw_img_path),
        os.path.join(repo_root, "data", raw_img_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def safe_mean(values):
    return float(np.mean(values)) if values else float("nan")


def main():
    args = parse_args()

    with open(args.eval_json, "r") as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} eval rows from {args.eval_json}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Downloading weights from HF: {args.hf_repo}/{args.hf_weight_path}")
    weight_path = hf_hub_download(
        repo_id=args.hf_repo,
        filename=args.hf_weight_path,
        repo_type="model",
        cache_dir=args.cache_dir,
    )
    print(f"Resolved checkpoint: {weight_path}")

    latent_euclid = LatentEuclid(
        base_model_id=config["model"]["base_model_id"],
        target_model_id=config["model"]["target_model_id"],
        k_steps=config["model"]["k_steps"],
    ).to(device)

    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        latent_euclid.load_state_dict(state_dict["model_state_dict"])
    else:
        latent_euclid.load_state_dict(state_dict)
    latent_euclid.eval()

    image_to_idx = {}
    image_to_question = {}
    with open(args.dataset_jsonl, "r") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            key = os.path.basename(item["image_path"])
            image_to_idx[key] = idx
            image_to_question[key] = item["question"]

    print(f"\n[DIAG] Dataset index built: {len(image_to_idx)} entries")
    sample_keys = list(image_to_idx.keys())[:3]
    print(f"[DIAG] Sample index keys: {sample_keys}")

    # Check target tensors dir
    if os.path.isdir(args.target_tensors_dir):
        pt_files = [f for f in os.listdir(args.target_tensors_dir) if f.endswith(".pt")]
        print(f"[DIAG] target_tensors_dir: {args.target_tensors_dir}  ({len(pt_files)} .pt files)")
        if pt_files:
            print(f"[DIAG] Sample tensor files: {sorted(pt_files)[:3]}")
    else:
        print(f"[DIAG] WARNING: target_tensors_dir does not exist: {args.target_tensors_dir}")

    # Probe the first eval item through every filter
    if eval_data:
        probe = eval_data[0]
        probe_raw = probe.get("image", "")
        probe_key = os.path.basename(probe_raw)
        probe_idx = image_to_idx.get(probe_key)
        probe_target = (
            os.path.join(args.target_tensors_dir, f"problem_{probe_idx}_targets.pt")
            if probe_idx is not None else None
        )
        probe_img = resolve_image_path(probe_raw, args.repo_root)
        print("\n[DIAG] Probe first eval item:")
        print(f"  raw image path : {probe_raw}")
        print(f"  basename key   : {probe_key}")
        print(f"  in index?      : {probe_idx is not None}  (idx={probe_idx})")
        print(f"  target path    : {probe_target}")
        print(f"  target exists? : {os.path.exists(probe_target) if probe_target else False}")
        print(f"  resolved image : {probe_img}")
        print(f"  image exists?  : {os.path.exists(probe_img) if probe_img else False}")
        print()

    correct_mses, correct_coss = [], []
    failed_mses, failed_coss = [], []
    skip_no_index = skip_no_tensor = skip_no_image = 0

    with torch.no_grad():
        for item in tqdm(eval_data, desc="Computing latent errors"):
            raw_img_path = item.get("image", "")
            is_correct = bool(item.get("is_correct", False))

            img_path_key = os.path.basename(raw_img_path)
            if img_path_key not in image_to_idx:
                skip_no_index += 1
                continue

            idx = image_to_idx[img_path_key]
            target_path = os.path.join(args.target_tensors_dir, f"problem_{idx}_targets.pt")
            if not os.path.exists(target_path):
                skip_no_tensor += 1
                continue

            resolved_img_path = resolve_image_path(raw_img_path, args.repo_root)
            if resolved_img_path is None:
                skip_no_image += 1
                continue

            try:
                img = Image.open(resolved_img_path).convert("RGB")
            except Exception:
                continue

            target_tensor = torch.load(target_path, map_location=device, weights_only=True).unsqueeze(0)

            question = image_to_question[img_path_key]
            thought_string = "".join([f"<thought_{i+1}>" for i in range(config["model"]["k_steps"])])
            full_prompt = question + " " + thought_string
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ]
            text_prompt = latent_euclid.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inputs = latent_euclid.processor(
                text=[text_prompt], images=[img], padding=True, return_tensors="pt"
            ).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_latents = latent_euclid(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                )

            mse = F.mse_loss(pred_latents.float(), target_tensor.float()).item()
            cos = F.cosine_similarity(pred_latents.float(), target_tensor.float(), dim=2).mean().item()

            if is_correct:
                correct_mses.append(mse)
                correct_coss.append(cos)
            else:
                failed_mses.append(mse)
                failed_coss.append(cos)

    print("\n" + "=" * 60)
    print("Latent Vector Prediction Error Analysis (v12_manifold_anchor_cod)")
    print("=" * 60)
    if skip_no_index + skip_no_tensor + skip_no_image > 0:
        print(f"[SKIPPED] not in dataset index: {skip_no_index}")
        print(f"[SKIPPED] target tensor missing: {skip_no_tensor}")
        print(f"[SKIPPED] image file not found:  {skip_no_image}")
        print()
    print(f"Correct samples: {len(correct_mses)}")
    print(f"  Avg MSE Loss:   {safe_mean(correct_mses):.6f}")
    print(f"  Avg Cosine Sim: {safe_mean(correct_coss):.6f}")
    print(f"Failed samples: {len(failed_mses)}")
    print(f"  Avg MSE Loss:   {safe_mean(failed_mses):.6f}")
    print(f"  Avg Cosine Sim: {safe_mean(failed_coss):.6f}")

    if args.summary_json:
        summary = {
            "correct": {
                "count": len(correct_mses),
                "avg_mse": safe_mean(correct_mses),
                "avg_cosine": safe_mean(correct_coss),
            },
            "failed": {
                "count": len(failed_mses),
                "avg_mse": safe_mean(failed_mses),
                "avg_cosine": safe_mean(failed_coss),
            },
            "skipped": {
                "no_index": skip_no_index,
                "no_tensor": skip_no_tensor,
                "no_image": skip_no_image,
            },
        }
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved latent summary to {args.summary_json}")



if __name__ == "__main__":
    main()
