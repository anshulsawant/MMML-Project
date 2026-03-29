import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


SYSTEM_PROMPT = """You are an expert mathematical validator. You will be provided with a geometry problem, its corresponding image, and a "Chain-of-Draft"—a highly compressed mathematical shorthand outlining the logical steps to solve it. 

The draft is incomplete and stops right before the final solution. Your task is to:
1. Examine the image to understand the spatial variables.
2. Follow the exact mathematical logic provided in the "Draft".
3. Calculate the final missing value.

Do not write a long explanation. Output only the final mathematical calculation, and conclude your response strictly with the format: `#### [Final Answer]`."""


@dataclass
class EvalRecord:
    idx: int
    image_path: str
    question: str
    gt_answer: str
    cod_raw: str
    cod_stripped: str
    pass_count: int
    runs: int
    keep: bool
    predictions: List[str]
    parsed_predictions: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-consistency filtering for CoD samples.")
    parser.add_argument("--cod-jsonl", type=str, default="qwen3_vl_cod_dataset_raw.jsonl")
    parser.add_argument("--output-filtered", type=str, default="qwen3_vl_cod_dataset_filtered_sc.jsonl")
    parser.add_argument("--output-report-json", type=str, default="cod_self_consistency_report.json")
    parser.add_argument("--output-report-csv", type=str, default="cod_self_consistency_report.csv")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--consensus-threshold", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit", "fp16"], default="8bit")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model_and_processor(model_id: str, quantization: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)

    kwargs: Dict[str, Any] = {
        "device_map": "auto" if device == "cuda" else None,
        "low_cpu_mem_usage": True,
    }

    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32

    if quantization in {"4bit", "8bit"} and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as exc:
            print(f"Warning: failed to enable {quantization} ({exc}). Falling back to fp16.")

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, processor


def extract_think_and_final(assistant_text: str) -> Tuple[str, str]:
    think_match = re.search(r"<think>(.*?)</think>", assistant_text, flags=re.DOTALL | re.IGNORECASE)
    think = think_match.group(1).strip() if think_match else ""
    final_part = re.sub(r"<think>.*?</think>", "", assistant_text, flags=re.DOTALL | re.IGNORECASE).strip()
    return think, final_part


def strip_cod_answer(cod_text: str) -> str:
    if not cod_text:
        return ""

    text = cod_text
    text = text.split("####")[0]

    refusal_markers = [
        "i can\'t provide",
        "i cannot provide",
        "chain-of-thought",
        "internal step-by-step",
        "internal reasoning",
        "can\'t share",
    ]
    lowered = text.lower()
    if any(marker in lowered for marker in refusal_markers):
        return ""

    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        ll = line.lower().strip()
        if ll.startswith("final answer") or ll.startswith("answer:"):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def extract_index_from_image_path(image_path: str) -> Optional[int]:
    match = re.search(r"geothought_(\d+)", image_path)
    if not match:
        return None
    return int(match.group(1))


def extract_question_from_messages(messages: List[Dict[str, Any]]) -> str:
    if not messages:
        return ""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        for item in msg.get("content", []):
            if item.get("type") == "text":
                return str(item.get("text", "")).replace("<image>", "").strip()
    return ""


def extract_image_path_from_messages(messages: List[Dict[str, Any]]) -> str:
    if not messages:
        return ""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        for item in msg.get("content", []):
            if item.get("type") == "image":
                return str(item.get("image", ""))
    return ""


def _parse_final_number(line: str) -> str:
    """Extract the final numeric answer from a single-line answer string.

    Handles patterns like:
      - 'Final Answer: 145'
      - 'Final answer: \\( 80 \\)'
      - 'Thus, ... is \\( \\boldsymbol{75} \\).'
      - '\\boldsymbol{360}'
    Returns the raw value string (still needs normalize_answer later).
    """
    # 1. \boldsymbol{X} — the highlighted final answer
    boldsymbol = re.findall(r"\\boldsymbol\{([^}]+)\}", line)
    if boldsymbol:
        return boldsymbol[-1].strip()

    # 2. Inline math \( X \) where X is a simple numeric expression
    inline = re.findall(r"\\\(\s*([^\\()]+?)\s*\\\)", line)
    if inline:
        # Pick the last match that looks like a number (digits, decimal, degree, pi, sqrt, /)
        number_like = [m.strip() for m in inline if re.match(r"^[\d.\-+/°piπ\s\\]+$", m.strip())]
        if number_like:
            return number_like[-1]
        # Fallback: last inline math content
        return inline[-1].strip()

    # 3. After "Final Answer:" / "Final answer:" — take the rest (bare number)
    fa_match = re.search(r"final\s+answer\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
    if fa_match:
        return fa_match.group(1).strip()

    # 4. Last decimal/integer in the line
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
    if numbers:
        return numbers[-1]

    return line.strip()


def extract_ground_truth(solution_text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", solution_text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        answer_block = match.group(1).strip()
        # The final answer is always on the last non-empty line of the block.
        lines = [ln.strip() for ln in answer_block.splitlines() if ln.strip()]
        last_line = lines[-1] if lines else ""
        return _parse_final_number(last_line)

    # Fallback: remove think block and take last non-empty line.
    tail = re.sub(r"<think>.*?</think>", "", solution_text, flags=re.DOTALL | re.IGNORECASE).strip()
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    return _parse_final_number(lines[-1]) if lines else ""


def extract_pred_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip()

    patterns = [
        r"final answer\s*[:=]\s*(.*)$",
        r"answer\s*[:=]\s*(.*)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def safe_math_eval(expr_str: str) -> Optional[float]:
    expr = expr_str
    expr = re.sub(r"(\d)(pi|sqrt)", r"\1*\2", expr)
    expr = re.sub(r"\)(pi|sqrt|\d)", r")*\1", expr)
    expr = re.sub(r"(pi)(\d)", r"\1*\2", expr)
    expr = expr.replace(")(", ")*(")

    allowed_names = {"pi": math.pi, "sqrt": math.sqrt, "__builtins__": {}}
    safe_chars = set("0123456789.+-*/()pisqrt ")
    if not expr or not all(ch in safe_chars for ch in expr):
        return None

    try:
        return float(eval(expr, allowed_names))
    except Exception:
        return None


def normalize_answer(ans: str) -> str:
    text = str(ans).strip()
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\\(?:boxed|boldsymbol|mathbf|mathrm|text|mathit|fbox)\{", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("$", "")
    text = text.replace("^\\circ", "").replace("\\circ", "").replace("°", "")
    text = re.sub(r"degrees?|degree", "", text)
    text = text.replace("\\pi", "pi").replace("π", "pi")
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\s*([^ ]+)", r"sqrt(\1)", text)
    text = re.sub(
        r"(?:cm|mm|km|m|meters|meter|inches|inch|feet|foot|ft|miles|mile|yards|yard|yd|units|unit)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("(", "").replace(")", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace(" ", "")
    text = text.strip(" .,:;!?")

    value = safe_math_eval(text)
    if value is not None:
        if value.is_integer():
            return str(int(value))
        return str(round(value, 6))

    return text


def is_correct(pred: str, gt: str) -> bool:
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    if pred_norm == gt_norm:
        return True

    pred_val = safe_math_eval(pred_norm)
    gt_val = safe_math_eval(gt_norm)
    if pred_val is not None and gt_val is not None:
        return math.isclose(pred_val, gt_val, rel_tol=1e-3, abs_tol=0.06)
    return False


def resolve_image_path(image_path: str, base_dir: str) -> str:
    if os.path.isabs(image_path):
        return image_path
    return os.path.normpath(os.path.join(base_dir, image_path))


def build_user_prompt(question: str, cod_without_answer: str) -> str:
    return (
        f"Problem:\n{question}\n\n"
        f"Draft Logic:\n{cod_without_answer}\n\n"
        "Task: Complete the calcuation and provide the final answer in the format `#### [Final Answer]`."
    )


def generate_once(
    model,
    processor,
    image: Image.Image,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: str,
) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if device == "cuda":
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    do_sample = temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        )

    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids[:, input_len:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return text.strip()


def load_cod_samples(cod_jsonl_path: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(cod_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def evaluate_dataset(args: argparse.Namespace) -> Tuple[List[EvalRecord], List[Dict[str, Any]]]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cod_base_dir = os.path.dirname(os.path.abspath(args.cod_jsonl)) or "."
    cod_samples = load_cod_samples(args.cod_jsonl)
    gt_dataset = load_dataset("xinlingdedeng/Geo-Thought", split="train")

    model, processor = load_model_and_processor(args.model_id, args.quantization, args.device)

    end_index = len(cod_samples) if args.limit is None else min(len(cod_samples), args.start_index + args.limit)
    selected = cod_samples[args.start_index:end_index]

    records: List[EvalRecord] = []
    filtered_samples: List[Dict[str, Any]] = []

    skip_counts: Dict[str, int] = {
        "no_image_path": 0,
        "bad_idx": 0,
        "empty_cod": 0,
        "no_gt": 0,
        "no_image_data": 0,
    }

    for local_i, sample in enumerate(selected):
        messages = sample.get("messages", [])
        question = extract_question_from_messages(messages)
        image_path = extract_image_path_from_messages(messages)
        if not image_path:
            skip_counts["no_image_path"] += 1
            continue

        idx = extract_index_from_image_path(image_path)
        if idx is None or idx >= len(gt_dataset):
            skip_counts["bad_idx"] += 1
            continue

        if not question:
            question = str(gt_dataset[idx].get("problem", "")).replace("<image>", "").strip()

        assistant_msg = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_msg = str(msg.get("content", ""))
                break

        cod_raw, _ = extract_think_and_final(assistant_msg)
        cod_stripped = strip_cod_answer(cod_raw)
        if not cod_stripped:
            skip_counts["empty_cod"] += 1
            continue

        gt_answer = extract_ground_truth(str(gt_dataset[idx].get("solution", "")))
        if not gt_answer:
            skip_counts["no_gt"] += 1
            continue

        # Prefer image from HF dataset (always available); fall back to local file.
        image: Optional[Image.Image] = None
        row = gt_dataset[idx]
        hf_img = row.get("image") if hasattr(row, "get") else row["image"] if "image" in row else None
        if hf_img is None:
            hf_img = row.get("images") if hasattr(row, "get") else None
        if hf_img is not None:
            if not isinstance(hf_img, Image.Image):
                hf_img = Image.fromarray(hf_img) if hasattr(hf_img, "__array__") else None
            if hf_img is not None:
                image = hf_img.convert("RGB")
        if image is None:
            resolved_image = resolve_image_path(image_path, cod_base_dir)
            if os.path.exists(resolved_image):
                image = Image.open(resolved_image).convert("RGB")
        if image is None:
            skip_counts["no_image_data"] += 1
            continue

        user_prompt = build_user_prompt(question, cod_stripped)
        predictions: List[str] = []
        parsed_predictions: List[str] = []
        pass_count = 0

        for _ in range(args.runs):
            out_text = generate_once(
                model=model,
                processor=processor,
                image=image,
                user_prompt=user_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
            pred = extract_pred_answer(out_text)
            predictions.append(out_text)
            parsed_predictions.append(pred)
            if is_correct(pred, gt_answer):
                pass_count += 1

        keep = pass_count >= args.consensus_threshold

        records.append(
            EvalRecord(
                idx=idx,
                image_path=image_path,
                question=question,
                gt_answer=gt_answer,
                cod_raw=cod_raw,
                cod_stripped=cod_stripped,
                pass_count=pass_count,
                runs=args.runs,
                keep=keep,
                predictions=predictions,
                parsed_predictions=parsed_predictions,
            )
        )

        if keep:
            filtered_samples.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": question},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": f"<think>\n{cod_stripped}\n</think>\n{gt_answer}",
                        },
                    ]
                }
            )

        # Lightweight progress logging for long runs.
        if (local_i + 1) % 10 == 0:
            kept_n = sum(1 for r in records if r.keep)
            print(f"Processed {local_i + 1}/{len(selected)} | kept={kept_n} | current_idx={idx}")

    print(
        f"Skip summary: no_image_path={skip_counts['no_image_path']} | "
        f"bad_idx={skip_counts['bad_idx']} | empty_cod={skip_counts['empty_cod']} | "
        f"no_gt={skip_counts['no_gt']} | no_image_data={skip_counts['no_image_data']}"
    )
    return records, filtered_samples


def save_outputs(
    records: List[EvalRecord],
    filtered_samples: List[Dict[str, Any]],
    output_filtered: str,
    output_report_json: str,
    output_report_csv: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_filtered)) or ".", exist_ok=True)

    with open(output_filtered, "w", encoding="utf-8") as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    report_payload: Dict[str, Any] = {
        "summary": {
            "evaluated": len(records),
            "kept": sum(1 for r in records if r.keep),
            "keep_rate": (sum(1 for r in records if r.keep) / len(records)) if records else 0.0,
            "mean_pass_rate": (
                sum(r.pass_count / max(r.runs, 1) for r in records) / len(records) if records else 0.0
            ),
        },
        "records": [
            {
                "idx": r.idx,
                "image_path": r.image_path,
                "question": r.question,
                "gt_answer": r.gt_answer,
                "cod_raw": r.cod_raw,
                "cod_stripped": r.cod_stripped,
                "pass_count": r.pass_count,
                "runs": r.runs,
                "pass_rate": r.pass_count / max(r.runs, 1),
                "keep": r.keep,
                "parsed_predictions": r.parsed_predictions,
                "predictions": r.predictions,
            }
            for r in records
        ],
    }

    with open(output_report_json, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    with open(output_report_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "image_path",
                "gt_answer",
                "pass_count",
                "runs",
                "pass_rate",
                "keep",
                "cod_preview",
                "predictions_preview",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "idx": r.idx,
                    "image_path": r.image_path,
                    "gt_answer": r.gt_answer,
                    "pass_count": r.pass_count,
                    "runs": r.runs,
                    "pass_rate": round(r.pass_count / max(r.runs, 1), 4),
                    "keep": r.keep,
                    "cod_preview": r.cod_stripped[:200],
                    "predictions_preview": " || ".join(r.parsed_predictions[:3]),
                }
            )


def main() -> None:
    args = parse_args()
    print("Starting CoD self-consistency filtering...")
    print(
        f"model={args.model_id}, quantization={args.quantization}, runs={args.runs}, "
        f"threshold={args.consensus_threshold}, temp={args.temperature}, top_p={args.top_p}"
    )
    records, filtered_samples = evaluate_dataset(args)
    save_outputs(
        records=records,
        filtered_samples=filtered_samples,
        output_filtered=args.output_filtered,
        output_report_json=args.output_report_json,
        output_report_csv=args.output_report_csv,
    )
    kept = sum(1 for r in records if r.keep)
    print(
        f"Done. evaluated={len(records)} kept={kept} "
        f"keep_rate={(kept / len(records) * 100 if records else 0):.2f}%"
    )


if __name__ == "__main__":
    main()
