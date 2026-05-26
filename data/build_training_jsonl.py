"""
Build the training JSONL required by GeoThoughtsDataset.

Input files
-----------
geothought_cod_full.jsonl   – raw records with embedded image bytes,
                               CoD_steps (list[str]), and solution
                               (which contains the full <think>…</think> CoT).
                               Uses __source_index as the primary key.

geothought_cod_full_filtered.jsonl – self-consistency-filtered subset;
                               each record has only a `messages` field whose
                               image path encodes the source index, e.g.
                               ./geothought_images/geothought_00001.jpg → idx 1

Output fields (per record)
--------------------------
image_path  : str  – relative path to the saved .jpg, e.g.
                     geothought_images/geothought_00001.jpg
CoD_steps   : list[str] – chain-of-draft steps
CoT_text    : str  – full chain-of-thought reasoning (text inside <think>…</think>)

Usage
-----
    python -m data.build_training_jsonl                 # uses defaults
    python -m data.build_training_jsonl \\
        --raw      ChainOfDraft/geothought_cod_full.jsonl \\
        --filtered ChainOfDraft/geothought_cod_full_filtered.jsonl \\
        --output   ChainOfDraft/training_dataset.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_IDX_RE   = re.compile(r"geothought_(\d+)")


def extract_cot(solution: str) -> str:
    """Return the text inside <think>…</think>; fall back to full solution."""
    m = _THINK_RE.search(solution)
    return m.group(1).strip() if m else solution.strip()


def index_from_image_path(image_path: str) -> int | None:
    """Parse the 5-digit index from a path like ./geothought_images/geothought_00001.jpg."""
    m = _IDX_RE.search(image_path)
    return int(m.group(1)) if m else None


def load_raw_lookup(raw_path: Path) -> dict[int, dict]:
    """
    Returns {source_index: {"CoD_steps": [...], "CoT_text": "..."}}
    for every record in the raw JSONL.
    """
    lookup: dict[int, dict] = {}
    with raw_path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skipping malformed line {lineno} in raw file: {exc}", file=sys.stderr)
                continue

            idx = rec.get("__source_index")
            if idx is None:
                print(f"[WARN] line {lineno} has no __source_index, skipping", file=sys.stderr)
                continue

            cod_steps = rec.get("CoD_steps")
            if not isinstance(cod_steps, list) or len(cod_steps) == 0:
                # skip records with no CoD steps — they can't be trained on
                continue

            solution = rec.get("solution", "")
            cot_text = extract_cot(solution)

            lookup[int(idx)] = {
                "CoD_steps": cod_steps,
                "CoT_text":  cot_text,
            }

    print(f"[INFO] loaded {len(lookup):,} usable records from raw file", file=sys.stderr)
    return lookup


def build(raw_path: Path, filtered_path: Path, output_path: Path) -> None:
    lookup = load_raw_lookup(raw_path)

    written = 0
    skipped_no_idx = 0
    skipped_no_match = 0
    skipped_no_image = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with filtered_path.open() as fin, output_path.open("w") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skipping malformed line {lineno} in filtered file: {exc}", file=sys.stderr)
                continue

            # Extract image path from the messages structure
            messages = rec.get("messages", [])
            image_path: str | None = None
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                for part in msg.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "image":
                        image_path = part.get("image")
                        break
                if image_path:
                    break

            if not image_path:
                skipped_no_image += 1
                continue

            # Strip leading "./" so paths are workspace-relative
            image_path = image_path.lstrip("./")

            idx = index_from_image_path(image_path)
            if idx is None:
                skipped_no_idx += 1
                print(f"[WARN] could not parse index from image path: {image_path!r}", file=sys.stderr)
                continue

            payload = lookup.get(idx)
            if payload is None:
                skipped_no_match += 1
                continue

            out = {
                "image_path": image_path,
                "CoD_steps":  payload["CoD_steps"],
                "CoT_text":   payload["CoT_text"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"[INFO] wrote {written:,} records → {output_path}\n"
        f"       skipped: no image={skipped_no_image}, "
        f"no index={skipped_no_idx}, no raw match={skipped_no_match}",
        file=sys.stderr,
    )


def main() -> None:
    here = Path(__file__).resolve().parent.parent   # project root

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw",
        default=str(here / "ChainOfDraft/geothought_cod_full.jsonl"),
        help="Path to geothought_cod_full.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--filtered",
        default=str(here / "ChainOfDraft/geothought_cod_full_filtered.jsonl"),
        help="Path to the SC-filtered JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(here / "ChainOfDraft/training_dataset.jsonl"),
        help="Output path (default: %(default)s)",
    )
    args = parser.parse_args()

    build(Path(args.raw), Path(args.filtered), Path(args.output))


if __name__ == "__main__":
    main()
