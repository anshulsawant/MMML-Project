import argparse
import json
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a few target tensors from HF and verify they are not all zero."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v13_contrast.yaml",
        help="Config path used to resolve HF repo and prefixes.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HF model repo id. Defaults to data.targets_hf_repo from config.",
    )
    parser.add_argument(
        "--prefixes",
        nargs="*",
        default=None,
        help="Repo prefixes to search for problem_<idx>_targets.pt files.",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[10000, 10001],
        help="Problem indices to download and inspect.",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=".hf_target_probe",
        help="Local directory for downloaded probe tensors.",
    )
    return parser.parse_args()


def load_defaults(config_path: str):
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)
    data_cfg = config.get("data", {})
    return data_cfg.get("targets_hf_repo"), data_cfg.get("targets_hf_prefixes", [])


def resolve_download(repo_id: str, prefixes: list[str], idx: int, local_dir: str) -> tuple[str, str]:
    filename = f"problem_{idx}_targets.pt"
    last_error = None

    for prefix in prefixes or [""]:
        repo_filename = f"{prefix}/{filename}" if prefix else filename
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=repo_filename,
                repo_type="model",
                local_dir=local_dir,
            )
            return repo_filename, local_path
        except Exception as exc:
            last_error = exc

    raise FileNotFoundError(
        f"Could not resolve {filename} in repo {repo_id} across prefixes {prefixes}: {last_error}"
    )


def tensor_stats(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().cpu()
    nnz = int(torch.count_nonzero(tensor).item())
    abs_sum = float(tensor.abs().sum().item())
    max_abs = float(tensor.abs().max().item()) if tensor.numel() > 0 else 0.0
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "nnz": nnz,
        "all_zero": nnz == 0,
        "abs_sum": abs_sum,
        "max_abs": max_abs,
        "mean": float(tensor.float().mean().item()) if tensor.numel() > 0 else 0.0,
        "min": float(tensor.min().item()) if tensor.numel() > 0 else 0.0,
        "max": float(tensor.max().item()) if tensor.numel() > 0 else 0.0,
    }


def main():
    args = parse_args()
    config_repo_id, config_prefixes = load_defaults(args.config)
    repo_id = args.repo_id or config_repo_id
    prefixes = args.prefixes if args.prefixes is not None else config_prefixes

    if not repo_id:
        raise ValueError("No HF repo id provided and none found in config.")

    Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    print(f"Repo: {repo_id}")
    print(f"Prefixes: {prefixes}")
    print(f"Indices: {args.indices}")

    results = []
    found_all_zero = False

    for idx in args.indices:
        repo_filename, local_path = resolve_download(repo_id, prefixes, idx, args.local_dir)
        tensor = torch.load(local_path, map_location="cpu", weights_only=True)
        stats = tensor_stats(tensor)
        stats["index"] = idx
        stats["repo_filename"] = repo_filename
        stats["local_path"] = local_path
        results.append(stats)
        found_all_zero = found_all_zero or stats["all_zero"]
        print(json.dumps(stats, indent=2))

    if found_all_zero:
        raise SystemExit("One or more downloaded tensors are entirely zero.")

    print(f"Verified {len(results)} tensors; none are all-zero.")


if __name__ == "__main__":
    main()