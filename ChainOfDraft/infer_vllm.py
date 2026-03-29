"""
Inference for a fine-tuned Qwen2.5-VL-3B checkpoint using HuggingFace Transformers.

Evaluates on Geometry3K test split (MCQ: A/B/C/D).
Uses greedy decoding (temperature=0.0), max generation length 2048.
Output format matches molmo2/predictions_*.json.

Usage:
    # Direct (1-token answer):
    python ChainOfDraft/infer_vllm.py \
        --model-path checkpoints/qwen25vl3b-cod-sft/final \
        --test-dir test/ \
        --output-json ChainOfDraft/predictions_direct.json \
        --mode direct

    # Thinking (chain-of-thought then <answer>X</answer>):
    python ChainOfDraft/infer_vllm.py \
        --model-path checkpoints/qwen25vl3b-cod-sft/final \
        --test-dir test/ \
        --output-json ChainOfDraft/predictions_thinking.json \
        --mode thinking
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# ---------------------------------------------------------------------------
# Prompts (same as molmo2.ipynb)
# ---------------------------------------------------------------------------

PROMPT_DIRECT = (
    "The following problem refers to an image diagram. "
    "Please consider the diagram to answer the question. "
    "Your answer MUST be a single capital letter: A, B, C, or D.\n\n"
    "{problem_text}\n\n{choices_text}\n\n"
    "Answer:"
)

PROMPT_THINKING = (
    "You are a helpful AI assistant.\n"
    "When answering questions, you always consider the image diagram "
    "and must always show your step-by-step reasoning process first.\n"
    "Once you have reached a conclusion, you MUST answer in the following "
    "format: <answer>A, B, C, or D.</answer>\n\n"
    "{problem_text}\n\n{choices_text}\n\n"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference on Geometry3K for fine-tuned Qwen2.5-VL-3B.")
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to fine-tuned checkpoint or HF model id.")
    p.add_argument("--test-dir", type=str, default="test/",
                   help="Path to Geometry3K test/ folder.")
    p.add_argument("--output-json", type=str, default="predictions.json")
    p.add_argument("--mode", type=str, choices=["direct", "thinking"], default="direct",
                   help="Prompt mode: 'direct' (1-token) or 'thinking' (CoT).")
    p.add_argument("--max-gen-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for batched generation.")
    p.add_argument("--quantization", type=str, choices=["4bit", "8bit", "fp16", "bf16"],
                   default="bf16", help="Quantization mode.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Geometry3K loader (mirrors molmo2.ipynb)
# ---------------------------------------------------------------------------

def load_geometry3k_test(test_dir: str) -> List[Dict[str, Any]]:
    """Load Geometry3K test problems from the extracted zip folder."""
    choice_letters = {0: "A. ", 1: "B. ", 2: "C. ", 3: "D. ", 4: "E. ", 5: "F. "}
    problems = []

    problem_ids = sorted(
        [d for d in os.listdir(test_dir) if d.isdigit()], key=int
    )

    for pid in problem_ids:
        pdir = os.path.join(test_dir, pid)
        data_path = os.path.join(pdir, "data.json")
        img_path = os.path.join(pdir, "img_diagram.png")
        if not os.path.exists(data_path):
            continue

        with open(data_path, "r") as f:
            data = json.load(f)

        formatted_choices = [
            f"{choice_letters.get(i, str(i) + '.')}{c}"
            for i, c in enumerate(data.get("choices", []))
        ]

        problems.append({
            "problem_id": pid,
            "problem_text": data.get("problem_text", ""),
            "choices": formatted_choices,
            "answer": data.get("answer", ""),
            "image_path": img_path,
        })

    return problems


def extract_answer_tag(text: str) -> Optional[str]:
    """Extract letter from <answer>X</answer>."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(model_path: str, quantization: str, device: str):
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"

    kwargs: Dict[str, Any] = {
        "device_map": "auto" if device == "cuda" else None,
        "low_cpu_mem_usage": True,
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

    # Load Geometry3K test set.
    problems = load_geometry3k_test(args.test_dir)
    print(f"Loaded {len(problems)} Geometry3K test problems from {args.test_dir}")

    is_direct = args.mode == "direct"
    prompt_template = PROMPT_DIRECT if is_direct else PROMPT_THINKING
    max_new_tokens = 1 if is_direct else args.max_gen_tokens

    # Load model.
    model, processor = load_model_and_processor(args.model_path, args.quantization, args.device)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    results: List[Dict[str, Any]] = []

    # Process in batches.
    for batch_start in range(0, len(problems), args.batch_size):
        batch = problems[batch_start : batch_start + args.batch_size]

        messages_batch = []
        images_batch = []
        for prob in batch:
            choices_text = "\n".join(prob["choices"])
            prompt_text = prompt_template.format(
                problem_text=prob["problem_text"],
                choices_text=choices_text,
            )

            content: List[Dict[str, Any]] = []
            img = None
            if os.path.exists(prob["image_path"]):
                img = Image.open(prob["image_path"]).convert("RGB")
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt_text})

            messages_batch.append([{"role": "user", "content": content}])
            images_batch.append(img)

        # Tokenise the batch with the chat template.
        inputs = processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        if args.device == "cuda":
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Decode only the generated portion.
        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = output_ids[:, input_len:]
        decoded = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for prob, text in zip(batch, decoded):
            text = text.strip()
            if is_direct:
                results.append({
                    "problem_id": prob["problem_id"],
                    "prediction": text,
                })
            else:
                extracted = extract_answer_tag(text)
                results.append({
                    "problem_id": prob["problem_id"],
                    "prediction_full": text,
                    "prediction": extracted,
                })

        done = min(batch_start + args.batch_size, len(problems))
        print(f"Processed {done}/{len(problems)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(results)} predictions to {args.output_json}")


if __name__ == "__main__":
    main()
