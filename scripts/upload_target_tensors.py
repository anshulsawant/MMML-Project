"""
Upload target tensors to HuggingFace one by one, partitioned into train/val
folders using data/v4_split_keys.json.

Usage (run from repo root, tensors in one flat folder):
    python scripts/upload_target_tensors.py \
        --local_dir /path/to/target_tensors/test \
        --hf_repo shilinm/latent_euclid \
        --hf_prefix target_tensors_v13_contrast \
        --jsonl ChainOfDraft/training_dataset.jsonl \
        --split_keys data/v4_split_keys.json

Files are uploaded to:
    {hf_prefix}/train/problem_{i}_targets.pt
    {hf_prefix}/val/problem_{i}_targets.pt
    {hf_prefix}/unmatched/problem_{i}_targets.pt  (if not found in split keys)
"""

import argparse
import json
import os
import sys

from huggingface_hub import HfApi


def build_index_to_split(jsonl_path: str, split_keys_path: str) -> dict[int, str]:
    """Return {problem_idx: 'train'|'val'|'unmatched'}."""
    with open(split_keys_path) as f:
        splits = json.load(f)
    val_keys = set(splits["val_keys"])
    train_keys = set(splits["train_keys"])

    idx_to_split: dict[int, str] = {}
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            raw_path = item.get("image_path", "")
            norm_path = raw_path.lstrip("./")
            base = os.path.basename(norm_path)
            candidates = {raw_path, norm_path, base, f"./{norm_path}"}

            if any(k in val_keys for k in candidates):
                idx_to_split[idx] = "val"
            elif any(k in train_keys for k in candidates):
                idx_to_split[idx] = "train"
            else:
                idx_to_split[idx] = "unmatched"

    return idx_to_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True,
                        help="Local folder containing problem_*_targets.pt files")
    parser.add_argument("--hf_repo", default="shilinm/latent_euclid")
    parser.add_argument("--hf_prefix", default="target_tensors_v13_contrast")
    parser.add_argument("--jsonl", default="ChainOfDraft/training_dataset.jsonl")
    parser.add_argument("--split_keys", default="data/v4_split_keys.json")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be uploaded without actually uploading")
    args = parser.parse_args()

    print("Building index → split mapping...")
    idx_to_split = build_index_to_split(args.jsonl, args.split_keys)

    train_count = sum(1 for s in idx_to_split.values() if s == "train")
    val_count = sum(1 for s in idx_to_split.values() if s == "val")
    unmatched_count = sum(1 for s in idx_to_split.values() if s == "unmatched")
    print(f"  train={train_count}, val={val_count}, unmatched={unmatched_count}")

    if unmatched_count == train_count + val_count:
        print(
            "WARNING: All indices are unmatched — split keys may not correspond to "
            "this JSONL's image paths. All files will go to 'unmatched/' prefix.",
            file=sys.stderr,
        )

    api = HfApi()

    files = sorted(
        f for f in os.listdir(args.local_dir)
        if f.startswith("problem_") and f.endswith("_targets.pt")
    )
    print(f"Found {len(files)} tensor files in {args.local_dir}")

    uploaded = skipped = errors = 0
    for fname in files:
        # Extract index from filename: problem_{idx}_targets.pt
        try:
            idx = int(fname.split("_")[1])
        except (IndexError, ValueError):
            print(f"  SKIP (can't parse idx): {fname}")
            skipped += 1
            continue

        split = idx_to_split.get(idx, "unmatched")
        repo_path = f"{args.hf_prefix}/{split}/{fname}"
        local_path = os.path.join(args.local_dir, fname)

        if args.dry_run:
            print(f"  DRY RUN: {local_path} → {repo_path}")
            uploaded += 1
            continue

        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=args.hf_repo,
                repo_type="model",
            )
            uploaded += 1
            if uploaded % 100 == 0:
                print(f"  Uploaded {uploaded}/{len(files)}...")
        except Exception as e:
            print(f"  ERROR uploading {fname}: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone. uploaded={uploaded}, skipped={skipped}, errors={errors}")
    print(f"\nAdd to configs/v13_contrast.yaml under data.targets_hf_prefixes:")
    print(f"  - \"{args.hf_prefix}/train\"")
    print(f"  - \"{args.hf_prefix}/val\"")
    if unmatched_count > 0:
        print(f"  - \"{args.hf_prefix}/unmatched\"")


if __name__ == "__main__":
    main()
