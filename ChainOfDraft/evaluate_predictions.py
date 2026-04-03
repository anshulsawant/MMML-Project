"""
Evaluate predictions from the fine-tuned Qwen2.5-VL-3B model on Geometry3K.

Re-extracts the final answer from `prediction_full` (handles both
<answer>X</answer> tags and bare text after </think>), compares against
ground-truth labels, and reports reasoning-step length statistics.

Usage:
    python ChainOfDraft/evaluate_predictions.py \
        --predictions ChainOfDraft/predictions_qwen25_3b_finetuned.json \
        --test-dir test/
"""

import argparse
import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

LETTER_CHOICES = {"A", "B", "C", "D"}


def extract_answer(entry: Dict[str, Any], choices: List[str]) -> Tuple[Optional[str], str]:
    """Return (extracted_letter_or_value, extraction_method).

    Strategies (in order):
      1. <answer>X</answer> tag
      2. Text after </think> tag
      3. Last single capital letter A-D on its own line
      4. Numeric value matched against provided choices
    """
    full = (entry.get("prediction_full") or "").strip()
    if not full:
        return None, "empty"

    # 1. <answer>X</answer> tag
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", full, re.IGNORECASE | re.DOTALL)
    if m:
        ans = m.group(1).strip()
        letter = _normalise_to_letter(ans, choices)
        return letter, "answer_tag"

    # 2. Text after </think>
    m = re.search(r"</think>\s*(.*)", full, re.DOTALL)
    if m:
        ans = m.group(1).strip()
        letter = _normalise_to_letter(ans, choices)
        if letter:
            return letter, "post_think"

    # 3. Last standalone letter A-D
    letters = re.findall(r"(?:^|\n)\s*([A-D])\s*$", full, re.MULTILINE)
    if letters:
        return letters[-1], "last_letter"

    return None, "unextracted"


def _normalise_to_letter(raw: str, choices: List[str]) -> Optional[str]:
    """Try to map a raw answer string to a choice letter (A/B/C/D)."""
    raw = raw.strip()
    if not raw:
        return None

    # Direct letter
    if raw.upper() in LETTER_CHOICES:
        return raw.upper()

    # "A." or "B)" style
    m = re.match(r"^([A-D])[.)\s]", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Try matching the raw value against choice text (numeric or string)
    raw_clean = raw.rstrip(".").strip()
    idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    for i, choice in enumerate(choices):
        # Strip "A. " prefix from choice text
        choice_val = re.sub(r"^[A-D][.)]\s*", "", choice).strip()
        if raw_clean == choice_val:
            return idx_to_letter.get(i)
        # Loose numeric comparison
        try:
            if abs(float(raw_clean) - float(choice_val)) < 1e-6:
                return idx_to_letter.get(i)
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Reasoning length helpers
# ---------------------------------------------------------------------------

def get_think_block(entry: Dict[str, Any]) -> str:
    """Extract the text inside <think>...</think>."""
    full = entry.get("prediction_full") or ""
    m = re.search(r"<think>(.*?)</think>", full, re.DOTALL)
    return m.group(1).strip() if m else ""


def reasoning_stats(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {}
    return {
        "count": len(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": round(statistics.median(lengths), 1),
        "stdev": round(statistics.stdev(lengths), 1) if len(lengths) > 1 else 0.0,
        "min": min(lengths),
        "max": max(lengths),
        "p25": round(sorted(lengths)[len(lengths) // 4], 1),
        "p75": round(sorted(lengths)[3 * len(lengths) // 4], 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Geometry3K predictions.")
    parser.add_argument("--predictions", type=str,
                        default="ChainOfDraft/predictions_qwen25_3b_finetuned.json")
    parser.add_argument("--test-dir", type=str, default="test/")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-problem details for wrong / unextracted answers.")
    args = parser.parse_args()

    # Load predictions.
    with open(args.predictions, "r", encoding="utf-8") as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} predictions from {args.predictions}")

    # Load ground truths.
    gt: Dict[str, Dict[str, Any]] = {}
    choice_letters = {0: "A. ", 1: "B. ", 2: "C. ", 3: "D. "}
    for pid_dir in os.listdir(args.test_dir):
        data_path = os.path.join(args.test_dir, pid_dir, "data.json")
        if not os.path.exists(data_path):
            continue
        with open(data_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        formatted_choices = [
            f"{choice_letters.get(i, str(i) + '.')}{c}"
            for i, c in enumerate(d.get("choices", []))
        ]
        gt[pid_dir] = {
            "answer": d.get("answer", ""),
            "choices": formatted_choices,
            "raw_choices": d.get("choices", []),
        }
    print(f"Loaded {len(gt)} ground-truth entries from {args.test_dir}\n")

    # Evaluate.
    total = 0
    correct = 0
    wrong = 0
    unextracted = 0
    no_gt = 0
    extraction_methods: Dict[str, int] = {}

    char_lengths: List[int] = []          # reasoning block char count
    word_lengths: List[int] = []          # reasoning block word count
    line_lengths: List[int] = []          # reasoning block line count
    correct_char_lengths: List[int] = []
    wrong_char_lengths: List[int] = []

    wrong_details: List[Dict[str, str]] = []

    for entry in preds:
        pid = entry["problem_id"]
        if pid not in gt:
            no_gt += 1
            continue

        total += 1
        truth = gt[pid]["answer"].strip().upper()
        choices = gt[pid]["choices"]

        pred_letter, method = extract_answer(entry, choices)
        extraction_methods[method] = extraction_methods.get(method, 0) + 1

        # Reasoning length
        think_text = get_think_block(entry)
        if think_text:
            char_lengths.append(len(think_text))
            word_lengths.append(len(think_text.split()))
            line_lengths.append(len(think_text.strip().splitlines()))

        if pred_letter is None:
            unextracted += 1
            if args.verbose:
                raw_after = ""
                m = re.search(r"</think>\s*(.*)", entry.get("prediction_full", ""), re.DOTALL)
                if m:
                    raw_after = m.group(1).strip()[:80]
                wrong_details.append({
                    "pid": pid, "gt": truth, "pred": "(null)",
                    "raw_after_think": raw_after, "method": method,
                })
            continue

        # Compare
        if pred_letter == truth:
            correct += 1
            if think_text:
                correct_char_lengths.append(len(think_text))
        else:
            wrong += 1
            if think_text:
                wrong_char_lengths.append(len(think_text))
            if args.verbose:
                wrong_details.append({
                    "pid": pid, "gt": truth, "pred": pred_letter, "method": method,
                })

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    answered = correct + wrong
    print("=" * 60)
    print("ACCURACY REPORT")
    print("=" * 60)
    print(f"Total problems (with GT):  {total}")
    print(f"Answered (extracted):      {answered}  ({answered/total*100:.1f}%)")
    print(f"  Correct:                 {correct}")
    print(f"  Wrong:                   {wrong}")
    print(f"Unextracted (null pred):   {unextracted}  ({unextracted/total*100:.1f}%)")
    print(f"No ground truth:           {no_gt}")
    print()
    print(f"Accuracy (over all):       {correct/total*100:.2f}%  ({correct}/{total})")
    if answered:
        print(f"Accuracy (answered only):  {correct/answered*100:.2f}%  ({correct}/{answered})")
    print()

    print("Extraction methods:")
    for method, count in sorted(extraction_methods.items(), key=lambda x: -x[1]):
        print(f"  {method:20s}  {count:5d}  ({count/total*100:.1f}%)")

    print()
    print("=" * 60)
    print("REASONING LENGTH STATISTICS")
    print("=" * 60)

    if char_lengths:
        print(f"\n{'Metric':<25s} {'Chars':>10s} {'Words':>10s} {'Lines':>10s}")
        print("-" * 57)
        cs = reasoning_stats(char_lengths)
        ws = reasoning_stats(word_lengths)
        ls = reasoning_stats(line_lengths)
        for key in ["count", "mean", "median", "stdev", "min", "max", "p25", "p75"]:
            print(f"  {key:<23s} {cs.get(key, ''):>10} {ws.get(key, ''):>10} {ls.get(key, ''):>10}")

        if correct_char_lengths and wrong_char_lengths:
            print(f"\n  Mean chars (correct):   {statistics.mean(correct_char_lengths):.1f}")
            print(f"  Mean chars (wrong):     {statistics.mean(wrong_char_lengths):.1f}")
        elif correct_char_lengths:
            print(f"\n  Mean chars (correct):   {statistics.mean(correct_char_lengths):.1f}")
    else:
        print("  No <think> blocks found.")

    # Verbose wrong-answer dump
    if args.verbose and wrong_details:
        print()
        print("=" * 60)
        print(f"WRONG / UNEXTRACTED DETAILS  (first 30 of {len(wrong_details)})")
        print("=" * 60)
        for d in wrong_details[:30]:
            print(f"  PID={d['pid']}  GT={d['gt']}  Pred={d['pred']}  "
                  f"Method={d['method']}  {d.get('raw_after_think', '')}")

    print()


if __name__ == "__main__":
    main()
