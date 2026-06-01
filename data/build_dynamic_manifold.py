'''
Data Engineering: Dynamic Expert Target Manifold (Y-Encoder)

This distinct script functionally preserves backwards compatibility with `build_manifold.py`
while implementing **Dynamic Latent Recursion**. It dynamically unrolls sequences of 
variable lengths extracted from `geothoughts_arbitrary_cot.jsonl` and forcibly 
appends a final HALT state for verifiable dynamic-length supervision.
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
import csv
import json


import argparse
import os
import yaml


def upload_targets_to_hf(local_dir: str, repo_id: str, path_in_repo: str) -> None:
    """Upload generated target tensors to a Hugging Face model repository."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"[auto] upload target tensors from {os.path.basename(local_dir)}",
        )
        print(f"Uploaded targets to HF: {repo_id}/{path_in_repo}")
    except Exception as e:
        print(f"Warning: failed to upload targets to HF ({repo_id}/{path_in_repo}): {e}")

def load_qwen_target_model(model_id: str, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Loading {model_id} for Target Manifold extraction on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    is_vlm = "VL" in model_id
    ModelClass = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
    
    if device == "cpu":
        model = ModelClass.from_pretrained(model_id, torch_dtype=torch.float32)
    else:
        model = ModelClass.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if device == "cpu":
        model = model.to(device)
        
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Warning: Could not load AutoProcessor for {model_id}. Text fallback only.")
        processor = None
        
    return tokenizer, processor, model

def embed_steps_batch(texts: list[str], bases: list[str], tokenizer, model, device="cuda", images=None, processor=None):
    """Passes a batch of step texts natively through Qwen3-0.6B and extracts the final hidden states."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if images is not None and processor is not None and any(images):
        messages_full = []
        messages_base = []
        for txt, base_txt, img in zip(texts, bases, images):
            messages_full.append([{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}])
            messages_base.append([{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": base_txt}]}])
            
        prompts_full = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_full]
        prompts_base = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_base]
        
        inputs = processor(text=prompts_full, images=images, return_tensors="pt", padding=True).to(device)
        inputs_base = processor(text=prompts_base, images=images, return_tensors="pt", padding=True).to(device)
        
        base_lengths = inputs_base.attention_mask.sum(dim=1)
        full_lengths = inputs.attention_mask.sum(dim=1)
    else:
        inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        inputs_base = tokenizer(bases, padding=True, return_tensors="pt").to(device)
        base_lengths = inputs_base.attention_mask.sum(dim=1)
        full_lengths = inputs.attention_mask.sum(dim=1)
    
    target_mask = torch.zeros_like(inputs.attention_mask)
    for i in range(len(texts)):
        b_len = int(base_lengths[i].item())
        f_len = int(full_lengths[i].item())
        
        if tokenizer.padding_side == 'right':
            target_mask[i, b_len:f_len] = 1
        else:
            seq_len = target_mask.shape[1]
            target_mask[i, seq_len - f_len + b_len : seq_len] = 1
            
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states[-1]
        
        target_mask = target_mask.unsqueeze(-1).to(hidden_states.dtype)
        sum_embeddings = torch.sum(hidden_states * target_mask, dim=1)
        sum_mask = torch.clamp(target_mask.sum(dim=1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask
    
    return mean_pooled_embeddings.cpu()

def _pack_and_upload(output_dir: str, split_keys_path: str, hf_repo: str | None, hf_targets_prefix: str | None) -> None:
    """Pack individual problem_*_targets.pt files into one .pt dict per split and upload."""
    with open(split_keys_path) as f:
        split_keys = json.load(f)
    idx_to_split: dict[int, str] = {}
    for split in ("train", "val", "test"):
        for idx in split_keys.get(f"{split}_indices", []):
            idx_to_split[idx] = split

    packed: dict[str, dict] = {"train": {}, "val": {}, "test": {}}
    for fname in os.listdir(output_dir):
        if not (fname.startswith("problem_") and fname.endswith("_targets.pt")):
            continue
        try:
            idx = int(fname.split("_")[1])
        except (IndexError, ValueError):
            continue
        split = idx_to_split.get(idx)
        if split is None:
            continue
        packed[split][idx] = torch.load(
            os.path.join(output_dir, fname), map_location="cpu", weights_only=True
        )

    if hf_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
        except ImportError:
            api = None
    else:
        api = None

    for split, d in packed.items():
        if not d:
            continue
        out_path = os.path.join(output_dir, f"{split}_targets.pt")
        torch.save(d, out_path)
        print(f"Packed {len(d)} tensors -> {out_path}")
        if api and hf_targets_prefix:
            repo_path = f"{hf_targets_prefix}/{split}_targets.pt"
            api.upload_file(
                path_or_fileobj=out_path,
                path_in_repo=repo_path,
                repo_id=hf_repo,
                repo_type="model",
                commit_message=f"[auto] packed {split} target tensors",
            )
            print(f"Uploaded -> {hf_repo}/{repo_path}")


def build_manifold(
    model_id: str,
    input_jsonl: str,
    output_dir: str,
    filter_csv: str = None,
    hf_repo: str = None,
    hf_targets_prefix: str = None,
    split_keys_path: str = None,
):
    """Processes dynamic text and saves continuous target tensors."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, processor, model = load_qwen_target_model(model_id, device)
    
    batch_size = config.get("build_manifold", {}).get("batch_size", 4)
    print(f"Processing with chunked batch_size: {batch_size}")
    
    with open(input_jsonl, 'r') as f:
        lines = f.readlines()

    # Load CSV filter: only process rows where keep=True
    keep_map = None
    if filter_csv and os.path.exists(filter_csv):
        keep_map = {}
        with open(filter_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['keep'].strip() in ('True', '1', 'true'):
                    keep_map[int(row['idx'])] = row['image_path']
        print(f"CSV filter loaded: {len(keep_map)} samples with keep=True from {filter_csv}")
        
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        batch_data = [json.loads(line) for line in batch_lines]
        
        # Determine which items in this batch to process (filtered + not already saved)
        indices_to_process = []
        for j in range(len(batch_data)):
            global_idx = i + j
            if keep_map is not None and global_idx not in keep_map:
                continue
            if not os.path.exists(os.path.join(output_dir, f"problem_{global_idx}_targets.pt")):
                indices_to_process.append(j)

        if not indices_to_process:
            continue
            
        flat_steps = []
        flat_bases = []
        flat_images = []
        lengths = []
        kept_global_indices = []
        
        for j in indices_to_process:
            global_idx = i + j
            data = batch_data[j]

            q_text = data.get("question", data.get("text", "")).replace("<image>", "").strip()
            for k in range(1, 20):
                q_text = q_text.replace(f"<thought_{k}>", "")
                
            prefix = f"{q_text}\nAnswer: "
            # Use image path from CSV when available, fall back to JSONL field
            img_path = keep_map[global_idx] if keep_map is not None else data.get("image_path", data.get("image", ""))
            
            cod_array = data.get("CoD_steps", [])

            cumulative_text = ""
            for step_text in cod_array:
                flat_bases.append(f"{prefix}{cumulative_text}")
                cumulative_text = f"{cumulative_text}\n{step_text}" if cumulative_text else step_text
                flat_steps.append(f"{prefix}{cumulative_text}")
                flat_images.append(img_path)

            # Final HALT state
            flat_bases.append(f"{prefix}{cumulative_text}")
            flat_steps.append(f"{prefix}{cumulative_text}\n<HALT>")
            flat_images.append(img_path)

            lengths.append(len(cod_array) + 1)
            kept_global_indices.append(global_idx)
            
        target_tensors_flat = embed_steps_batch(flat_steps, flat_bases, tokenizer, model, device=device, images=flat_images, processor=processor)
        target_tensors = torch.split(target_tensors_flat, lengths)
        
        import io
        for tensor, global_idx in zip(target_tensors, kept_global_indices):
            target_path = os.path.join(output_dir, f"problem_{global_idx}_targets.pt")
            
            buf = io.BytesIO()
            torch.save(tensor.clone(), buf)
            
            with open(target_path, 'wb') as f:
                f.write(buf.getvalue())
                f.flush()
                os.fsync(f.fileno())
            
        if (i + len(batch_data)) % 25 < batch_size:
            print(f"Generated manifolds for {i + len(batch_data)} problems ({len(kept_global_indices)} kept in last batch)...")

    if split_keys_path:
        _pack_and_upload(output_dir, split_keys_path, hf_repo, hf_targets_prefix)
    elif hf_repo:
        if not hf_targets_prefix:
            hf_targets_prefix = f"target_tensors/{os.path.basename(output_dir)}"
        upload_targets_to_hf(output_dir, hf_repo, hf_targets_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dynamic continuous manifold targets.")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--input_jsonl", type=str, default=None)
    parser.add_argument("--filter_csv", type=str, default=None,
                        help="Path to CSV with 'idx', 'image_path', and 'keep' columns (e.g. geothought_cod_full_report.csv).")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="Optional Hugging Face model repo id for uploading generated target tensors.")
    parser.add_argument("--hf_targets_prefix", type=str, default=None,
                        help="Optional path prefix inside HF repo for targets (default: target_tensors/<output_dir_basename>).")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split_keys", type=str, default=None,
                        help="Path to v13_split_keys.json; if set, packs output into per-split .pt files.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    experiment_name = args.experiment_name or config.get("experiment", {}).get("name", "v12_dynamic_sft")
    model_id = args.model_id or config["model"]["target_model_id"]
    input_jsonl = args.input_jsonl or config.get("data", {}).get("jsonl_path", "data/geothoughts_arbitrary_cot.jsonl")
    
    output_dir = args.output_dir
    if output_dir is None:
        base_dir = config.get("data", {}).get("targets_dir", "./target_tensors")
        output_dir = os.path.join(base_dir, f"target_tensors_{experiment_name}")
        
    if "data" not in config: config["data"] = {}
    config["data"]["targets_dir"] = output_dir

    print("\n" + "="*50)
    print("LatentEuclid Phase 15 (Dynamic Continuous Manifold Target Generation)")
    print(f"Dynamically generating explicitly robust sequence lengths mapped identically with HALT blocks...")
    print("="*50 + "\n")
        
    hf_repo = args.hf_repo or config.get("build_manifold", {}).get("hf_repo")
    hf_targets_prefix = args.hf_targets_prefix or config.get("build_manifold", {}).get("hf_targets_prefix")
    split_keys_path = args.split_keys or config.get("build_manifold", {}).get("split_keys_path")

    build_manifold(
        model_id,
        input_jsonl,
        output_dir,
        filter_csv=args.filter_csv,
        hf_repo=hf_repo,
        hf_targets_prefix=hf_targets_prefix,
        split_keys_path=split_keys_path,
    )
