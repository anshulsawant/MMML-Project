"""
Manifold Anchoring via Knowledge Distillation

Trains a small MLP projector to align Chain-of-Draft (CoD) embeddings with
Chain-of-Thought (CoT) anchor embeddings in the frozen Y-encoder's latent space.

The CoT anchor is the mean of all K=4 cumulative step embeddings (same embedding
method as build_manifold.py). The CoD embedding is a single mean-pooled vector
over the full compressed reasoning block.

After training, build_manifold.py can apply this projector to produce
"anchored" target tensors for samples with CoD, while unpaired samples
retain their original CoT embeddings (identity fallback).

Features:
    - RunPod S3 sync: download/upload checkpoints from RunPod object storage
    - HuggingFace Hub auto-upload: push projector weights after training

Usage:
    python training/train_manifold_anchor.py --config training/config.yaml

    # Quick debug run
    python training/train_manifold_anchor.py --config training/config.yaml --limit 100 --epochs 3

    # With S3 sync and HF upload
    python training/train_manifold_anchor.py --config training/config.yaml \
        --s3-bucket 7ih6gcggwr --hf-repo anshulsawant/latent-euclid-manifold-anchor
"""

import argparse
import json
import os
import re
import subprocess

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Reuse the embedding infrastructure from build_manifold
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.build_manifold import load_qwen_target_model, embed_steps_batch, parse_k4_steps


# ---------------------------------------------------------------------------
# Projector architecture
# ---------------------------------------------------------------------------

class ManifoldProjector(nn.Module):
    """Small MLP that maps a CoD embedding onto the CoT anchor manifold."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# Data pairing
# ---------------------------------------------------------------------------

def load_cod_lookup(cod_jsonl_path: str) -> dict:
    """Build a dict mapping question text -> CoD assistant content."""
    lookup = {}
    with open(cod_jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            question = None
            cod_text = None
            for msg in entry["messages"]:
                if msg["role"] == "user":
                    for c in msg["content"]:
                        if c["type"] == "text":
                            question = c["text"].strip()
                elif msg["role"] == "assistant":
                    cod_text = msg["content"]
            if question and cod_text:
                lookup[question] = cod_text
    return lookup


def extract_cod_reasoning(cod_text: str) -> str:
    """Extract the reasoning block from <think>...</think> tags."""
    m = re.search(r"<think>(.*?)</think>", cod_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: everything before the final answer line
    return cod_text.strip()


def build_paired_dataset(geothoughts_jsonl: str, cod_jsonl: str):
    """Return list of dicts with 'question', 'cot_steps' (list[str]), 'cod_reasoning' (str), 'image_path'."""
    cod_lookup = load_cod_lookup(cod_jsonl)

    pairs = []
    with open(geothoughts_jsonl, "r") as f:
        for line in f:
            item = json.loads(line)
            question = item["question"].replace("<image>", "").strip()
            if question not in cod_lookup:
                continue

            cot_steps = parse_k4_steps(item["reasoning"])
            cod_reasoning = extract_cod_reasoning(cod_lookup[question])
            if not cod_reasoning:
                continue

            prefix = f"{question}\nAnswer: "
            pairs.append({
                "question": question,
                "prefix": prefix,
                "cot_steps": cot_steps,          # list of 4 cumulative step strings
                "cod_reasoning": cod_reasoning,   # single compressed reasoning string
                "image_path": item.get("image_path"),
            })

    return pairs


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_cot_anchor(pairs_batch, tokenizer, model, device, processor=None):
    """Compute the CoT anchor: mean of K=4 cumulative step embeddings.
    
    This replicates build_manifold.py's logic exactly so the anchor space
    is identical to the existing target manifold.
    Returns: [batch, hidden_dim]
    """
    flat_steps = []
    flat_bases = []
    flat_images = []
    for pair in pairs_batch:
        prefix = pair["prefix"]
        img_path = pair["image_path"]
        base = prefix
        for step_text in pair["cot_steps"]:
            flat_bases.append(base)
            flat_steps.append(f"{prefix}{step_text}")
            flat_images.append(img_path)
            base = f"{prefix}{step_text}"

    # [batch*4, hidden_dim]
    all_embeddings = embed_steps_batch(
        flat_steps, flat_bases, tokenizer, model, device=device,
        images=flat_images, processor=processor,
    )

    # Reshape to [batch, 4, hidden_dim] and mean-pool across K steps
    batch_size = len(pairs_batch)
    all_embeddings = all_embeddings.view(batch_size, 4, -1)
    return all_embeddings.mean(dim=1)  # [batch, hidden_dim]


def embed_cod_single(pairs_batch, tokenizer, model, device, processor=None):
    """Compute the CoD embedding: single mean-pooled vector over the full reasoning block.
    
    Returns: [batch, hidden_dim]
    """
    texts = []
    bases = []
    images = []
    for pair in pairs_batch:
        prefix = pair["prefix"]
        texts.append(f"{prefix}{pair['cod_reasoning']}")
        bases.append(prefix)
        images.append(pair["image_path"])

    return embed_steps_batch(
        texts, bases, tokenizer, model, device=device,
        images=images, processor=processor,
    )


# ---------------------------------------------------------------------------
# RunPod S3 helpers
# ---------------------------------------------------------------------------

RUNPOD_S3_REGION = "us-md-1"
RUNPOD_S3_ENDPOINT = "https://s3api-us-md-1.runpod.io"


def _s3_cmd(args: list[str], bucket: str) -> list[str]:
    """Build an aws s3 command list with RunPod endpoint configuration."""
    return [
        "aws", "s3",
        *args,
        "--region", RUNPOD_S3_REGION,
        "--endpoint-url", RUNPOD_S3_ENDPOINT,
    ]


def s3_download(bucket: str, s3_prefix: str, local_dir: str) -> None:
    """Download files from RunPod S3 to local_dir."""
    s3_uri = f"s3://{bucket}/{s3_prefix}"
    os.makedirs(local_dir, exist_ok=True)
    cmd = _s3_cmd(["sync", s3_uri, local_dir], bucket)
    print(f"  S3 download: {s3_uri} -> {local_dir}")
    subprocess.run(cmd, check=True)


def s3_upload(bucket: str, local_dir: str, s3_prefix: str) -> None:
    """Upload local_dir to RunPod S3."""
    s3_uri = f"s3://{bucket}/{s3_prefix}"
    cmd = _s3_cmd(["sync", local_dir, s3_uri], bucket)
    print(f"  S3 upload: {local_dir} -> {s3_uri}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# HuggingFace Hub upload
# ---------------------------------------------------------------------------

def hf_upload(output_dir: str, repo_id: str, commit_message: str = "Update manifold anchor projector") -> None:
    """Upload projector checkpoint directory to HuggingFace Hub."""
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"  HF upload: {output_dir} -> {repo_id}")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"  Uploaded to https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Manifold Anchoring Projector")
    p.add_argument("--config", type=str, default="training/config.yaml")
    p.add_argument("--cod-dataset", type=str, default=None,
                   help="Path to CoD JSONL (default: from config or ChainOfDraft/qwen3_vl_cod_dataset_filtered_sc.jsonl)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--mse-weight", type=float, default=None,
                   help="Weight for MSE component of anchoring loss (default: 0.1)")
    p.add_argument("--limit", type=int, default=None, help="Limit paired samples for debugging")
    p.add_argument("--output-dir", type=str, default=None)
    # S3 sync
    p.add_argument("--s3-bucket", type=str, default=None,
                   help="RunPod S3 bucket name for checkpoint sync (e.g. 7ih6gcggwr)")
    p.add_argument("--s3-prefix", type=str, default=None,
                   help="S3 key prefix for checkpoints (default: manifold_anchor/<experiment_name>)")
    # HuggingFace Hub
    p.add_argument("--hf-repo", type=str, default=None,
                   help="HuggingFace Hub repo ID to auto-upload projector (e.g. anshulsawant/latent-euclid-manifold-anchor)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resolve parameters from config with CLI overrides
    anchor_cfg = config.get("train_manifold_anchor", {})
    model_id = config["model"]["target_model_id"]
    geothoughts_jsonl = config["data"]["jsonl_path"]
    cod_jsonl = args.cod_dataset or anchor_cfg.get("cod_dataset", "ChainOfDraft/qwen3_vl_cod_dataset_filtered_sc.jsonl")
    epochs = args.epochs or anchor_cfg.get("epochs", 10)
    batch_size = args.batch_size or anchor_cfg.get("batch_size", 4)
    lr = args.lr or anchor_cfg.get("learning_rate", 1e-4)
    mse_weight = args.mse_weight if args.mse_weight is not None else anchor_cfg.get("mse_weight", 0.1)

    experiment_name = anchor_cfg.get("experiment_name") or config.get("experiment", {}).get("name", "default")
    output_dir = args.output_dir or anchor_cfg.get("output_dir") or f"/workspace/checkpoints/{experiment_name}/manifold_anchor"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve S3 / HF config (CLI overrides config)
    s3_bucket = args.s3_bucket or anchor_cfg.get("s3_bucket")
    s3_prefix = args.s3_prefix or anchor_cfg.get("s3_prefix") or f"manifold_anchor/{experiment_name}"
    hf_repo = args.hf_repo or anchor_cfg.get("hf_repo")

    # Download existing checkpoint from S3 if available
    if s3_bucket:
        try:
            s3_download(s3_bucket, s3_prefix, output_dir)
            print("  Downloaded existing checkpoints from S3")
        except subprocess.CalledProcessError:
            print("  No existing S3 checkpoints found (starting fresh)")

    # 1. Build paired dataset
    print("Building CoT-CoD paired dataset...")
    pairs = build_paired_dataset(geothoughts_jsonl, cod_jsonl)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"  Paired samples: {len(pairs)}")

    if len(pairs) == 0:
        print("ERROR: No paired CoT-CoD samples found. Check dataset paths.")
        return

    # 2. Load frozen Y-encoder
    print(f"Loading frozen Y-encoder ({model_id})...")
    tokenizer, processor, model = load_qwen_target_model(model_id, device)
    hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 2560

    # 3. Create projector
    projector = ManifoldProjector(hidden_size).to(device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(output_dir, exist_ok=True)

    # 4. Training loop
    print(f"\n{'='*60}")
    print("Manifold Anchoring Training")
    print(f"  Model:      {model_id}")
    print(f"  Pairs:      {len(pairs)}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print(f"  MSE weight: {mse_weight}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    best_loss = float("inf")

    for epoch in range(epochs):
        # Shuffle pairs each epoch
        import random
        random.shuffle(pairs)

        epoch_loss = 0.0
        epoch_cos_loss = 0.0
        epoch_mse_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            range(0, len(pairs), batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for batch_start in pbar:
            batch = pairs[batch_start: batch_start + batch_size]

            # A. Compute CoT anchor embeddings (frozen, no grad)
            with torch.no_grad():
                cot_anchors = embed_cot_anchor(
                    batch, tokenizer, model, device, processor
                ).to(device, dtype=torch.bfloat16)

            # B. Compute CoD embeddings (frozen, no grad)
            with torch.no_grad():
                cod_embeddings = embed_cod_single(
                    batch, tokenizer, model, device, processor
                ).to(device, dtype=torch.bfloat16)

            # C. Project CoD → CoT anchor space
            projected_cod = projector(cod_embeddings)

            # D. Anchoring loss: cosine + weighted MSE
            cosine_loss = (
                1.0
                - nn.functional.cosine_similarity(projected_cod, cot_anchors, dim=-1).mean()
            )
            mse_loss = nn.functional.mse_loss(projected_cod, cot_anchors)
            total_loss = cosine_loss + mse_weight * mse_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_cos_loss += cosine_loss.item()
            epoch_mse_loss += mse_loss.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "cos": f"{cosine_loss.item():.4f}",
                "mse": f"{mse_loss.item():.4f}",
            })

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_cos = epoch_cos_loss / max(n_batches, 1)
        avg_mse = epoch_mse_loss / max(n_batches, 1)
        print(
            f"  Epoch {epoch + 1}: loss={avg_loss:.4f}  "
            f"cos={avg_cos:.4f}  mse={avg_mse:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(output_dir, "projector_best.pt")
            torch.save({
                "model_state_dict": projector.state_dict(),
                "hidden_size": hidden_size,
                "epoch": epoch + 1,
                "loss": avg_loss,
            }, save_path)
            print(f"  -> Saved best projector (loss={avg_loss:.4f}) to {save_path}")

    # Save final checkpoint
    final_path = os.path.join(output_dir, f"projector_epoch_{epochs}.pt")
    torch.save({
        "model_state_dict": projector.state_dict(),
        "hidden_size": hidden_size,
        "epoch": epochs,
        "loss": avg_loss,
    }, final_path)
    print(f"\nSaved final projector to {final_path}")
    print(f"Best loss: {best_loss:.4f}")

    # Upload to RunPod S3
    if s3_bucket:
        print("\nUploading checkpoints to RunPod S3...")
        try:
            s3_upload(s3_bucket, output_dir, s3_prefix)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: S3 upload failed: {e}")

    # Upload to HuggingFace Hub
    if hf_repo:
        print(f"\nUploading to HuggingFace Hub ({hf_repo})...")
        try:
            hf_upload(output_dir, hf_repo,
                      commit_message=f"Manifold anchor projector (epoch {epochs}, loss {best_loss:.4f})")
        except Exception as e:
            print(f"  WARNING: HF upload failed: {e}")


if __name__ == "__main__":
    main()
