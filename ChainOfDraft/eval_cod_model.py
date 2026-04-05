"""
Evaluate a HuggingFace VLM on the same validation split used by eval_e2e.py.

The validation set is the last 10% of data/geothoughts_verified.jsonl (shuffled
with seed=42), identical to the split used during LatentEuclid training.

Both modes evaluate only on questions NOT seen during CoD fine-tuning:
  --all-non-cod          : all non-CoD questions from the full geothoughts dataset
  --latent-euclid-non-cod: non-CoD questions within the LatentEuclid val split
                           (last 10%, seed=42 — same split as eval_e2e.py)

Usage:
    # Option 1: all non-CoD data
    python ChainOfDraft/eval_cod_model.py --all-non-cod

    # Option 2: only the LatentEuclid val subset of non-CoD data
    python ChainOfDraft/eval_cod_model.py --latent-euclid-non-cod
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# Reuse robust evaluation helpers from the LatentEuclid pipeline
sys.path.append("data")
from evaluate_generated import clean_gen_ans, normalize, safe_math_eval


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = (
    "You are a helpful AI assistant.\n"
    "When answering questions, you always consider the image diagram "
    "and must always show your step-by-step reasoning process first.\n"
    "Once you have reached a conclusion, you MUST answer in the following "
    "format: <think>[Chain of Thought]</think>[final answer]\n\n"
    "{question}"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a HuggingFace VLM on the GeoThought validation split.")
    p.add_argument("--model-path", type=str,
                   default="shilinm/cod_finetuned_qwen2.5",
                   help="HF model id or local path.")
    p.add_argument("--val-jsonl", type=str,
                   default="data/geothoughts_verified.jsonl",
                   help="Path to the full dataset JSONL (will be split).")
    p.add_argument("--ground-truths", type=str,
                   default="data/ground_truths.json",
                   help="Path to ground_truths.json.")
    p.add_argument("--cod-dataset", type=str,
                   default="ChainOfDraft/qwen3_vl_cod_dataset_filtered_sc.jsonl",
                   help="Path to CoD training JSONL (for overlap filtering).")
    p.add_argument("--split", type=float, default=0.9,
                   help="Train/val split ratio (default: 0.9 = last 10%% is val).")
    p.add_argument("--output-json", type=str,
                   default="ChainOfDraft/eval_cod_results.json")
    p.add_argument("--max-gen-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of samples to evaluate.")
    p.add_argument("--quantization", type=str,
                   choices=["4bit", "8bit", "fp16", "bf16"], default="bf16")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-non-cod", action="store_true",
                       help="Evaluate on ALL non-CoD questions from the full "
                            "geothoughts dataset (not just the val split).")
    group.add_argument("--latent-euclid-non-cod", action="store_true",
                       help="Evaluate only on non-CoD questions that fall in "
                            "the LatentEuclid validation split (last 10%%).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cod_questions(cod_path: str) -> set:
    """Return the set of question strings present in the CoD training JSONL."""
    questions = set()
    with open(cod_path, "r") as f:
        for line in f:
            d = json.loads(line)
            for msg in d["messages"]:
                if msg["role"] == "user":
                    for c in msg["content"]:
                        if c["type"] == "text":
                            questions.add(c["text"].strip())
    return questions


def load_full_dataset(jsonl_path: str):
    """Load the full dataset from the JSONL file."""
    with open(jsonl_path, "r") as f:
        return [json.loads(line) for line in f]


def load_val_split(jsonl_path: str, split: float):
    """Load and return the validation portion of the dataset (same split as
    eval_e2e.py and training)."""
    full_data = load_full_dataset(jsonl_path)
    random.seed(42)
    random.shuffle(full_data)
    split_idx = int(split * len(full_data))
    return full_data[split_idx:]


# ---------------------------------------------------------------------------
# Answer extraction from <think>...</think> output
# ---------------------------------------------------------------------------

def extract_answer_from_think(text: str) -> str:
    """Extract the final answer that appears after </think>, then clean it."""
    think_end = text.find("</think>")
    if think_end != -1:
        raw = text[think_end + len("</think>"):].strip()
    else:
        # Fallback: use last line
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        raw = lines[-1] if lines else text.strip()
    return clean_gen_ans(raw)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(model_path: str, quantization: str, device: str):
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"

    kwargs: Dict[str, Any] = {
        "device_map": "auto" if device == "cuda" else None,
    }

    if quantization == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif quantization == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif quantization in {"4bit", "8bit"} and device == "cuda":
        kwargs["torch_dtype"] = torch.float16
        from transformers import BitsAndBytesConfig
        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Load ground truths
    with open(args.ground_truths, "r") as f:
        ground_truths = json.load(f)

    # 2. Load data and filter to non-CoD questions
    cod_questions = load_cod_questions(args.cod_dataset)

    if args.all_non_cod:
        # Option 1: all non-CoD questions from the full dataset
        all_data = load_full_dataset(args.val_jsonl)
        val_data = [
            item for item in all_data
            if item["question"].replace("<image>", "").strip() not in cod_questions
        ]
        print(f"All non-CoD: {len(val_data)}/{len(all_data)} samples")
    else:
        # Option 2: non-CoD questions within the LatentEuclid val split
        le_val = load_val_split(args.val_jsonl, args.split)
        val_data = [
            item for item in le_val
            if item["question"].replace("<image>", "").strip() not in cod_questions
        ]
        print(f"LatentEuclid-val non-CoD: {len(val_data)}/{len(le_val)} samples")

    if args.limit:
        val_data = val_data[: args.limit]

    print(f"Evaluating {len(val_data)} samples with model {args.model_path}")

    # 4. Load model
    model, processor = load_model_and_processor(
        args.model_path, args.quantization, args.device
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 5. Evaluate in batches
    results: List[Dict[str, Any]] = []
    correct = 0
    total = 0

    batches = [
        val_data[i : i + args.batch_size]
        for i in range(0, len(val_data), args.batch_size)
    ]

    with torch.no_grad():
        pbar = tqdm(batches, desc="Evaluating")
        for batch in pbar:
            messages_batch = []
            images = []
            valid_items = []

            for item in batch:
                img_path = item["image_path"]
                true_answer = ground_truths.get(img_path)
                if true_answer is None:
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
                    continue

                question = item["question"].replace("<image>", "").strip()
                prompt_text = PROMPT.format(question=question)

                content: List[Dict[str, Any]] = [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ]
                messages_batch.append([{"role": "user", "content": content}])
                images.append(img)
                valid_items.append((item, true_answer))

            if not messages_batch:
                continue

            # Tokenise
            inputs = processor.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding=True,
                return_tensors="pt",
            )
            if args.device == "cuda":
                inputs = {
                    k: v.to("cuda") if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_gen_tokens,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[-1]
            generated_tokens = output_ids[:, input_len:]
            decoded = processor.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            for gen_text, (item, true_answer_raw) in zip(decoded, valid_items):
                gen_text = gen_text.strip()
                pred_raw = extract_answer_from_think(gen_text)
                gt_norm = normalize(true_answer_raw)
                pred_norm = normalize(pred_raw)

                is_correct = gt_norm == pred_norm

                if not is_correct:
                    gt_val = safe_math_eval(true_answer_raw)
                    pred_val = safe_math_eval(pred_raw)
                    if gt_val is not None and pred_val is not None:
                        is_correct = math.isclose(
                            gt_val, pred_val, rel_tol=1e-3, abs_tol=0.06
                        )

                total += 1
                if is_correct:
                    correct += 1

                results.append({
                    "image": item["image_path"],
                    "question": item["question"].replace("<image>", "").strip(),
                    "is_correct": is_correct,
                    "gt_raw": str(true_answer_raw),
                    "pred_raw": pred_raw,
                    "gt_norm": gt_norm,
                    "pred_norm": pred_norm,
                    "model_generation": gen_text,
                })

            if total > 0:
                pbar.set_postfix(
                    Correct=f"{correct}/{total}",
                    Accuracy=f"{correct / total * 100:.2f}%",
                )

    # 6. Report results
    acc = correct / total * 100 if total > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"Model:    {args.model_path}")
    print(f"Samples:  {total}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"{'='*50}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to {args.output_json}")


if __name__ == "__main__":
    main()
