"""
Pack existing per-file target tensors into split-level .pt dicts and upload to HF.

Run from repo root on the machine that has the individual .pt files:

    python scripts/pack_and_upload_targets.py \
        --local_dir /path/to/target_tensors/test \
        --hf_repo shilinm/latent_euclid \
        --hf_prefix target_tensors_v13_contrast \
        --split_keys data/v13_split_keys.json

Output files (saved locally then uploaded):
    {local_dir}/train_targets.pt  -> {hf_prefix}/train_targets.pt
    {local_dir}/val_targets.pt    -> {hf_prefix}/val_targets.pt
    {local_dir}/test_targets.pt   -> {hf_prefix}/test_targets.pt
"""

import argparse
import json
import os

import torch
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True,
                        help="Folder containing problem_*_targets.pt files")
    parser.add_argument("--hf_repo", default="shilinm/latent_euclid")
    parser.add_argument("--hf_prefix", default="target_tensors_v13_contrast")
    parser.add_argument("--split_keys", default="data/v13_split_keys.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    with open(args.split_keys) as f:
        split_keys = json.load(f)

    idx_to_split: dict[int, str] = {}
    for split in ("train", "val", "test"):
        key = f"{split}_indices"
        if key in split_keys:
            for idx in split_keys[key]:
                idx_to_split[idx] = split

    print(f"Split key counts: " + ", ".join(
        f"{s}={sum(1 for v in idx_to_split.values() if v == s)}"
        for s in ("train", "val", "test")
    ))

    packed: dict[str, dict[int, torch.Tensor]] = {"train": {}, "val": {}, "test": {}}
    missing = 0

    files = sorted(
        f for f in os.listdir(args.local_dir)
        if f.startswith("problem_") and f.endswith("_targets.pt")
    )
    print(f"Found {len(files)} tensor files")

    for fname in files:
        try:
            idx = int(fname.split("_")[1])
        except (IndexError, ValueError):
            continue
        split = idx_to_split.get(idx)
        if split is None:
            missing += 1
            continue
        t = torch.load(os.path.join(args.local_dir, fname),
                       map_location="cpu", weights_only=True)
        packed[split][idx] = t

    for split, d in packed.items():
        print(f"  {split}: {len(d)} tensors")

    if missing:
        print(f"  {missing} files had no split assignment (dropped)")

    api = HfApi()
    for split, d in packed.items():
        if not d:
            print(f"Skipping empty {split} pack")
            continue
        out_path = os.path.join(args.local_dir, f"{split}_targets.pt")
        print(f"Saving {out_path} ...")
        if not args.dry_run:
            torch.save(d, out_path)
        repo_path = f"{args.hf_prefix}/{split}_targets.pt"
        print(f"Uploading -> {args.hf_repo}/{repo_path}")
        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=out_path,
                path_in_repo=repo_path,
                repo_id=args.hf_repo,
                repo_type="model",
                commit_message=f"Add {split} packed target tensors",
            )
            print(f"  Uploaded {split}_targets.pt ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
