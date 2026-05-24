#!/usr/bin/env python3
"""Convert verbose CoT explanations into symbolic CoD steps via OpenAI-compatible chat API.

Supports JSONL and CSV input. Appends a CoD_steps field/column to each record.
If step-tag parsing fails or API errors occur, writes detailed records (including raw
model responses) to a separate error log for manual review.
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tqdm import tqdm


MODEL_NAME = "gpt-5-nano"
TEMPERATURE = 0.0
STEP_PATTERN = re.compile(r"<step>(.*?)</step>", re.DOTALL | re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = r"""You are a strict neuro-symbolic mathematical translation engine.
Your task is to convert verbose geometric Chain-of-Thought (CoT) text into a dense, purely symbolic Chain-of-Draft (CoD).

RULES:
1. Strip away all conversational English (e.g., "To determine the measure of", "Using the formula").
2. Use strict mathematical notation and standard LaTeX operators (\implies, \land, \perp, \triangle, \cong).
3. Split the logic into distinct mathematical steps.
4. Wrap every extracted step in an XML <step> tag.
5. Output NOTHING else. No introductions, no explanations.

EXAMPLE 1
User Input:
<answer>To determine the measure of angle 2, we recognize that angle 1 and angle 2 form a linear pair (they lie on a straight line \( AB \)). A straight line forms a straight angle, which measures \( 180^\circ \). Therefore, the sum of angle 1 and angle 2 is \( 180^\circ \).
Given \( \angle 1 = 35.0^\circ \), we calculate \( \angle 2 \) as:
\( \angle 2 = 180^\circ - \angle 1 = 180^\circ - 35.0^\circ = 145^\circ \)
Final Answer: 145</answer>

Assistant Output:
<step>\angle 1 \text{ and } \angle 2 \text{ on line } AB \implies \angle 1 + \angle 2 = 180^\circ</step>
<step>\angle 1 = 35.0^\circ \implies \angle 2 = 180^\circ - 35.0^\circ = 145^\circ</step>

EXAMPLE 2
User Input:
<answer>To determine the measure of angle \( DCF \), we analyze the angles around point \( C \) using properties of squares and regular pentagons.
### Step 1: Angle in the square \( ACDE \)
In a square, all internal angles are \( 90^\circ \). Since \( ACDE \) is a square, \( \angle ACD = 90^\circ \).
### Step 2: Angle in the regular pentagon \( BCFGH \)
The formula for the internal angle of a regular \( n \)-sided polygon is \( \frac{(n-2) \cdot 180^\circ}{n} \). For a pentagon (\( n = 5 \)):
\[ \text{Internal angle} = \frac{(5-2) \cdot 180^\circ}{5} = \frac{3 \cdot 180^\circ}{5} = 108^\circ \]
Since \( BCFGH \) is a regular pentagon, \( \angle BCF = 108^\circ \).
### Step 3: Sum of angles around point \( C \)
The sum of angles around any point is \( 360^\circ \). Let \( \angle DCF = x \).
\[ \angle ACD + \angle DCF + \angle BCF + \angle ACB = 360^\circ \]
Substitute the known values (\angle ACB = 120^\circ):
\[ 90^\circ + x + 108^\circ + 120^\circ = 360^\circ \implies x = 42^\circ \]
Thus, the measure of angle \( DCF \) is \( \boldsymbol{42} \).</answer>

Assistant Output:
<step>ACDE \text{ square} \implies \angle ACD = 90^\circ</step>
<step>BCFGH \text{ regular pentagon} \implies \angle BCF = \frac{(5-2)180^\circ}{5} = 108^\circ</step>
<step>\angle ACB = 120^\circ</step>
<step>\angle ACD + \angle DCF + \angle BCF + \angle ACB = 360^\circ</step>
<step>90^\circ + \angle DCF + 108^\circ + 120^\circ = 360^\circ \implies \angle DCF = 42^\circ</step>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate CoT text to symbolic CoD steps using an OpenAI-compatible API."
    )
    parser.add_argument("--input", default=None, help="Path to input JSONL or CSV file.")
    parser.add_argument(
        "--source",
        choices=["file", "hf"],
        default="file",
        help="Input source: local file (jsonl/csv) or Hugging Face dataset.",
    )
    parser.add_argument(
        "--hf-dataset",
        default="xinlingdedeng/Geo-Thought",
        help="Hugging Face dataset name when --source hf is used.",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Hugging Face split name when --source hf is used.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional Hugging Face cache dir override.",
    )
    parser.add_argument(
        "--hf-local-only",
        action="store_true",
        help="Load HF dataset from local cache only (no network).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output file. If omitted, writes alongside input with a _cod suffix.",
    )
    parser.add_argument(
        "--cot-column",
        default="solution",
        help="Column/key name that contains the original CoT text. Default: solution",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="Custom OpenAI-compatible base URL (e.g., LiteLLM proxy).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for each API request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Number of retries for transient timeout/connection/rate-limit errors.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Delay between API calls to reduce rate-limit pressure.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows (use 50 for prompt validation).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this row index.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Persist progress every N processed rows.",
    )
    return parser.parse_args()


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Unsupported input format: {suffix}. Use .jsonl or .csv")


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_cod{input_path.suffix}")


def default_hf_output_path(dataset_name: str, split: str) -> Path:
    safe_name = dataset_name.replace("/", "__")
    return Path(f"{safe_name}_{split}_cod.jsonl")


def default_error_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_errors.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_num}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_steps(raw_text: str) -> list[str]:
    steps = [m.strip() for m in STEP_PATTERN.findall(raw_text) if m.strip()]
    return steps


def extract_answer_text(cot_text: str) -> str | None:
    match = ANSWER_PATTERN.search(cot_text)
    if not match:
        return None
    answer_text = match.group(1).strip()
    return answer_text if answer_text else None


def make_user_prompt(answer_text: str) -> str:
    return f"<answer>{answer_text}</answer>"


def call_model(
    client: OpenAI,
    answer_text: str,
    timeout_seconds: float,
    max_retries: int,
) -> str:
    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": make_user_prompt(answer_text)},
                ],
                timeout=timeout_seconds,
            )
            content = response.choices[0].message.content
            return content or ""
        except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"API call failed after retries: {exc}") from exc
            backoff = min(2.0 * attempt, 10.0)
            time.sleep(backoff)


def append_error(
    error_records: list[dict[str, Any]],
    row_idx: int,
    message: str,
    raw_response: str,
    row: dict[str, Any],
) -> None:
    error_records.append(
        {
            "row_index": row_idx,
            "error": message,
            "raw_response": raw_response,
            "input_row": row,
        }
    )


def load_hf_rows(
    dataset_name: str,
    split: str,
    cache_dir: str | None,
    local_only: bool,
) -> tuple[list[dict[str, Any]], int]:
    ds: Dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        download_mode="reuse_dataset_if_exists",
        streaming=False,
    )
    if local_only:
        # Retry with offline mode if requested.
        os.environ["HF_HUB_OFFLINE"] = "1"
    rows = ds.to_list()
    return rows, len(rows)


def read_existing_output_indices(output_path: Path) -> set[int]:
    done: set[int] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = obj.get("__source_index")
            if isinstance(idx, int):
                done.add(idx)
    return done


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    # Keep PIL/image-like objects compact in output files.
    cls_name = value.__class__.__name__
    mod_name = value.__class__.__module__
    return f"<{mod_name}.{cls_name}>"


def sanitize_row_for_output(row: dict[str, Any]) -> dict[str, Any]:
    return {k: to_json_safe(v) for k, v in row.items()}


def process_rows(
    rows: list[dict[str, Any]],
    cot_column: str,
    client: OpenAI,
    timeout_seconds: float,
    max_retries: int,
    sleep_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    error_records: list[dict[str, Any]] = []

    for idx, row in enumerate(tqdm(rows, desc="Converting CoT -> CoD", unit="row")):
        cot_text = row.get(cot_column)
        if cot_text is None or str(cot_text).strip() == "":
            row["CoD_steps"] = []
            append_error(
                error_records,
                idx,
                f"Missing or empty CoT in column/key '{cot_column}'",
                raw_response="",
                row=row,
            )
            continue

        answer_text = extract_answer_text(str(cot_text))
        if answer_text is None:
            row["CoD_steps"] = []
            append_error(
                error_records,
                idx,
                "Missing or empty <answer>...</answer> content in CoT field",
                raw_response="",
                row=row,
            )
            continue

        raw_response = ""
        try:
            raw_response = call_model(
                client=client,
                answer_text=answer_text,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            steps = extract_steps(raw_response)
            if not steps:
                row["CoD_steps"] = []
                append_error(
                    error_records,
                    idx,
                    "Failed to parse any <step>...</step> tags from model response",
                    raw_response=raw_response,
                    row=row,
                )
            else:
                row["CoD_steps"] = steps
        except Exception as exc:  # noqa: BLE001
            row["CoD_steps"] = []
            append_error(
                error_records,
                idx,
                f"API/parsing exception: {exc}",
                raw_response=raw_response,
                row=row,
            )

        time.sleep(sleep_seconds)

    return rows, error_records


def normalize_for_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        # CSV stores lists/dicts as JSON strings.
        if "CoD_steps" in row_copy and not isinstance(row_copy["CoD_steps"], str):
            row_copy["CoD_steps"] = json.dumps(row_copy["CoD_steps"], ensure_ascii=False)
        normalized.append(row_copy)
    return normalized


def main() -> None:
    args = parse_args()

    if args.source == "file":
        if not args.input:
            raise ValueError("--input is required when --source file is used.")
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        input_format = detect_format(input_path)
        output_path = Path(args.output) if args.output else default_output_path(input_path)
    else:
        input_path = None
        input_format = "jsonl"
        output_path = (
            Path(args.output)
            if args.output
            else default_hf_output_path(args.hf_dataset, args.hf_split)
        )

    error_log_path = default_error_path(output_path)

    if not args.api_key:
        raise ValueError("API key is required. Use --api-key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    if args.source == "file":
        if input_format == "jsonl":
            rows = read_jsonl(input_path)
        else:
            rows = read_csv(input_path)
        total_rows = len(rows)
    else:
        print(
            f"Loading HF dataset {args.hf_dataset} ({args.hf_split}) "
            f"local_only={args.hf_local_only}..."
        )
        rows, total_rows = load_hf_rows(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            cache_dir=args.hf_cache_dir,
            local_only=args.hf_local_only,
        )
        cache_dir = args.hf_cache_dir or os.path.expanduser("~/.cache/huggingface")
        cache_hint = Path(cache_dir)
        print(f"HF cache root: {cache_hint}")
        print(f"Total HF rows loaded: {total_rows}")

    if not rows:
        print("Input file has no rows. Writing empty output.")
        if input_format == "jsonl":
            write_jsonl(output_path, [])
        else:
            write_csv(output_path, [])
        return

    if args.cot_column not in rows[0]:
        raise KeyError(
            f"Column/key '{args.cot_column}' not found in input rows. "
            f"Available keys: {list(rows[0].keys())}"
        )

    done_indices = read_existing_output_indices(output_path)
    start_index = max(0, args.start_index)
    stop_index = total_rows if args.limit is None else min(total_rows, start_index + args.limit)
    candidate_indices = list(range(start_index, stop_index))
    pending_indices = [i for i in candidate_indices if i not in done_indices]

    if args.limit is not None:
        print(f"Limit mode enabled: processing up to {args.limit} rows.")
    print(f"Already completed rows in output: {len(done_indices)}")
    print(f"Pending rows in selected range: {len(pending_indices)}")

    if not pending_indices:
        print("No pending rows to process. Resume check complete.")
        print(f"Output file: {output_path}")
        print(f"Error log: {error_log_path}")
        return

    errors_batch: list[dict[str, Any]] = []
    success_batch: list[dict[str, Any]] = []
    processed_counter = 0

    for idx in tqdm(pending_indices, desc="Converting CoT -> CoD", unit="row"):
        row = sanitize_row_for_output(dict(rows[idx]))
        row["__source_index"] = idx

        cot_text = row.get(args.cot_column)
        if cot_text is None or str(cot_text).strip() == "":
            row["CoD_steps"] = []
            append_error(
                errors_batch,
                idx,
                f"Missing or empty CoT in column/key '{args.cot_column}'",
                raw_response="",
                row=row,
            )
        else:
            answer_text = extract_answer_text(str(cot_text))
            if answer_text is None:
                row["CoD_steps"] = []
                append_error(
                    errors_batch,
                    idx,
                    "Missing or empty <answer>...</answer> content in CoT field",
                    raw_response="",
                    row=row,
                )
            else:
                raw_response = ""
                try:
                    raw_response = call_model(
                        client=client,
                        answer_text=answer_text,
                        timeout_seconds=args.request_timeout,
                        max_retries=args.max_retries,
                    )
                    steps = extract_steps(raw_response)
                    if not steps:
                        row["CoD_steps"] = []
                        append_error(
                            errors_batch,
                            idx,
                            "Failed to parse any <step>...</step> tags from model response",
                            raw_response=raw_response,
                            row=row,
                        )
                    else:
                        row["CoD_steps"] = steps
                except Exception as exc:  # noqa: BLE001
                    row["CoD_steps"] = []
                    append_error(
                        errors_batch,
                        idx,
                        f"API/parsing exception: {exc}",
                        raw_response=raw_response,
                        row=row,
                    )

        success_batch.append(row)
        processed_counter += 1
        time.sleep(args.sleep_seconds)

        if processed_counter % max(1, args.flush_every) == 0:
            append_jsonl(output_path, success_batch)
            append_jsonl(error_log_path, errors_batch)
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"Checkpoint flushed at {processed_counter} processed rows."
            )
            success_batch = []
            errors_batch = []

    if success_batch:
        append_jsonl(output_path, success_batch)
    if errors_batch:
        append_jsonl(error_log_path, errors_batch)

    print(f"Done. Wrote/updated output: {output_path}")
    print(f"Error log: {error_log_path}")
    print(f"Rows processed this run: {processed_counter}")


if __name__ == "__main__":
    main()
