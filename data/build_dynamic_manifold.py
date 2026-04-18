'''
Data Engineering: Dynamic Expert Target Manifold (Y-Encoder)

This distinct script functionally preserves backwards compatibility with `build_manifold.py`
while implementing **Dynamic Latent Recursion**. It dynamically unrolls sequences of 
variable lengths extracted from `geothoughts_arbitrary_cot.jsonl` and forcibly 
prepends the structural Step 0 mathematical map globally.
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
import json
import re


import argparse
import os
import yaml

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

def parse_dynamic_steps(reasoning_text: str, step0_text: str):
    """Extracts arbitrary dynamic steps cumulatively from the LLM output."""
    steps = [step0_text]
    parts = re.split(r'Step \d+.*?\]?:', reasoning_text)
    
    cumulative_text = step0_text
    for p in parts[1:]:
        clean_p = p.strip()
        # Avoid empty splits
        if clean_p:
            cumulative_text += "\n" + clean_p
            steps.append(cumulative_text)
        
    # Append the explicit terminal HALT state condition to the final layer
    steps.append(cumulative_text + "\n<HALT>")
    return steps

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

def build_manifold(model_id: str, input_jsonl: str, output_dir: str):
    """Processes dynamic text and saves continuous target tensors."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, processor, model = load_qwen_target_model(model_id, device)
    
    batch_size = config.get("build_manifold", {}).get("batch_size", 4)
    print(f"Processing with chunked batch_size: {batch_size}")
    
    print("Caching Step 0 foundational mathematical parsed roots from strictly verified K=4 logic graph...")
    step0_map = {}
    if os.path.exists('data/geothoughts_verified.jsonl'):
        with open('data/geothoughts_verified.jsonl', 'r') as f:
            for line in f:
                d = json.loads(line)
                k4_ans = d.get('answer', d.get('conversations', [{}, {}])[1].get('content', ''))
                k4_parts = re.split(r'Step \d+.*?\]?:', k4_ans)
                base_img = d.get('image', '').split('/images/')[-1]
                if len(k4_parts) > 1:
                    step0_map[base_img] = k4_parts[1].strip()
                else:
                    step0_map[base_img] = k4_ans.strip()
    
    with open(input_jsonl, 'r') as f:
        lines = f.readlines()
        
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        batch_data = [json.loads(line) for line in batch_lines]
        
        flat_steps = []
        flat_bases = []
        flat_images = []
        lengths = []
        
        for data in batch_data:
            q_text = data.get("question", data.get("text", "")).replace("<image>", "").strip()
            for k in range(1, 20):
                q_text = q_text.replace(f"<thought_{k}>", "")
                
            prefix = f"{q_text}\nAnswer: "
            img_path = data.get("image_path", data.get("image", ""))
            
            base_img = img_path.split("/images/")[-1] if "/images/" in img_path else img_path
            step0_text = step0_map.get(base_img, "Analyze mathematical geometries dynamically explicitly extracted from raw source.")
            
            reasoning = data.get("reasoning", data.get('conversations', [{}, {}])[1].get('content', ''))
            cumulative_steps = parse_dynamic_steps(reasoning, step0_text)
            
            base = prefix
            for step_text in cumulative_steps:
                flat_bases.append(base)
                flat_steps.append(f"{prefix}{step_text}")
                flat_images.append(img_path)
                base = f"{prefix}{step_text}"
                
            lengths.append(len(cumulative_steps))
            
        target_tensors_flat = embed_steps_batch(flat_steps, flat_bases, tokenizer, model, device=device, images=flat_images, processor=processor)
        target_tensors = torch.split(target_tensors_flat, lengths)
        
        import io
        for j, tensor in enumerate(target_tensors):
            idx = i + j
            target_path = os.path.join(output_dir, f"problem_{idx}_targets.pt")
            
            # MooseFS Network-Safe Serialization
            # By dumping to a memory buffer globally prior to POSIX block write, we bypass C++ ZipStream IO timeouts
            buf = io.BytesIO()
            torch.save(tensor.clone(), buf)
            
            with open(target_path, 'wb') as f:
                f.write(buf.getvalue())
                f.flush()
                os.fsync(f.fileno())
            
        if (i + len(batch_data)) % 25 < batch_size:
            print(f"Generated manifolds for {i + len(batch_data)} problems...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dynamic continuous manifold targets.")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--input_jsonl", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    experiment_name = args.experiment_name or config.get("experiment", {}).get("name", "v12_dynamic_sft")
    model_id = args.model_id or config["model"]["target_model_id"]
    input_jsonl = args.input_jsonl or config.get("data", {}).get("jsonl_path", "data/geothoughts_arbitrary_cot.jsonl")
    
    output_dir = args.output_dir
    if output_dir is None:
        base_dir = config.get("data", {}).get("targets_dir", "/workspace/target_tensors")
        output_dir = os.path.join(base_dir, f"target_tensors_{experiment_name}")
        
    if "data" not in config: config["data"] = {}
    config["data"]["targets_dir"] = output_dir

    print("\n" + "="*50)
    print("LatentEuclid Phase 15 (Dynamic Continuous Manifold Target Generation)")
    print(f"Dynamically generating explicitly robust sequence lengths mapped identically with HALT blocks...")
    print("="*50 + "\n")
        
    build_manifold(model_id, input_jsonl, output_dir)
