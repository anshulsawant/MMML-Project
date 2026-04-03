"""
SFT fine-tuning of Qwen2.5-VL-3B on filtered Chain-of-Draft data.

Objective : minimise negative log-likelihood on assistant tokens.
Optimizer : AdamW  (β1=0.9, β2=0.95, ε=1e-8, weight_decay=0.1)
Schedule  : 3 epochs, linear warmup 5 %, cosine decay
Precision : bfloat16 mixed precision + activation checkpointing
Clipping  : gradient norm ≤ 1.0
Tokens    : input ≤ 4096, target ≤ 1024
Batch     : effective global batch = 256  (micro_bs × grad_accum × n_gpu)

Usage (single GPU, micro_bs=1, grad_accum=256):
    python ChainOfDraft/train_sft.py \
        --data-jsonl ChainOfDraft/qwen3_vl_cod_dataset_filtered_sc.jsonl \
        --output-dir checkpoints/qwen25vl3b-cod-sft

Multi-GPU with torchrun (e.g. 4 GPUs, grad_accum auto-adjusted to 64):
    torchrun --nproc_per_node 4 ChainOfDraft/train_sft.py \
        --data-jsonl ChainOfDraft/qwen3_vl_cod_dataset_filtered_sc.jsonl \
        --output-dir checkpoints/qwen25vl3b-cod-sft
"""

import argparse
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT fine-tune Qwen2.5-VL-3B on CoD data.")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--data-jsonl", type=str, required=True,
                   help="Path to the filtered CoD JSONL (output of filter_cod_self_consistency.py).")
    p.add_argument("--output-dir", type=str, default="checkpoints/qwen25vl3b-cod-sft")
    # Hyperparams from the spec
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=1,
                   help="Per-device train batch size. grad_accum = global_bs / (micro_bs * n_gpu).")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-target-tokens", type=int, default=1024)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def extract_index_from_image_path(image_path: str) -> Optional[int]:
    match = re.search(r"geothought_(\d+)", image_path)
    return int(match.group(1)) if match else None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


class CoDDataset(torch.utils.data.Dataset):
    """Lazily tokenises each sample at __getitem__ time so the Trainer
    DataLoader can collate variable-length sequences."""

    IGNORE_INDEX = -100  # CrossEntropyLoss default ignore

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        processor: AutoProcessor,
        hf_dataset,
        max_input_tokens: int = 4096,
        max_target_tokens: int = 1024,
    ):
        self.samples = samples
        self.processor = processor
        self.hf_dataset = hf_dataset
        self.max_input_tokens = max_input_tokens
        self.max_target_tokens = max_target_tokens
        self.max_seq_len = max_input_tokens + max_target_tokens

    # ---- helpers ----------------------------------------------------------
    def _get_image(self, sample: dict) -> Optional[Image.Image]:
        """Resolve image: prefer HF dataset row, fall back to local file."""
        messages = sample.get("messages", [])
        image_path = ""
        for msg in messages:
            if msg.get("role") != "user":
                continue
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    image_path = str(item.get("image", ""))
                    break
        idx = extract_index_from_image_path(image_path)
        if idx is not None and idx < len(self.hf_dataset):
            hf_img = self.hf_dataset[idx].get("image")
            if isinstance(hf_img, Image.Image):
                return hf_img.convert("RGB")
        if image_path and os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
        return None

    def _get_question(self, sample: dict) -> str:
        for msg in sample.get("messages", []):
            if msg.get("role") != "user":
                continue
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    return str(item.get("text", "")).replace("<image>", "").strip()
        return ""

    def _get_assistant(self, sample: dict) -> str:
        for msg in sample.get("messages", []):
            if msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        return ""

    # ---- main tokenisation ------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = self._get_image(sample)
        question = self._get_question(sample)
        assistant = self._get_assistant(sample)

        # Build the chat messages that the processor expects.
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": image}, {"type": "text", "text": question}]
                    if image is not None
                    else [{"type": "text", "text": question}]
                ),
            },
            {"role": "assistant", "content": [{"type": "text", "text": assistant}]},
        ]

        # apply_chat_template gives us the full prompt with special tokens.
        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenise the full conversation (with image if present).
        if image is not None:
            inputs = self.processor(
                text=[full_text],
                images=[image],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_seq_len,
            )
        else:
            inputs = self.processor(
                text=[full_text],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_seq_len,
            )

        input_ids = inputs["input_ids"].squeeze(0)  # (seq_len,)

        # Build the prompt-only version (up to the assistant turn) so we know
        # which tokens are "input" vs "target" for the loss mask.
        prompt_messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": image}, {"type": "text", "text": question}]
                    if image is not None
                    else [{"type": "text", "text": question}]
                ),
            },
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if image is not None:
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_input_tokens,
            )
        else:
            prompt_inputs = self.processor(
                text=[prompt_text],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_input_tokens,
            )
        prompt_len = prompt_inputs["input_ids"].shape[-1]

        # Labels: mask out the prompt portion with IGNORE_INDEX.
        labels = input_ids.clone()
        labels[:prompt_len] = self.IGNORE_INDEX

        # Truncate target portion to max_target_tokens.
        target_len = len(labels) - prompt_len
        if target_len > self.max_target_tokens:
            total = prompt_len + self.max_target_tokens
            input_ids = input_ids[:total]
            labels = labels[:total]
            # Also truncate any pixel/image tensors if present
            for k in list(inputs.keys()):
                if k == "input_ids":
                    continue
                v = inputs[k]
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[-1] == inputs["input_ids"].shape[-1]:
                    inputs[k] = v[..., :total]

        result = {"input_ids": input_ids, "labels": labels}

        # Forward pixel values / image grid info if present.
        for key in ("pixel_values", "image_grid_thw"):
            if key in inputs:
                val = inputs[key]
                result[key] = val.squeeze(0) if isinstance(val, torch.Tensor) and val.dim() > 1 else val

        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad to longest in batch; set padding positions to IGNORE_INDEX in labels."""
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_list, labels_list, attention_mask_list = [], [], []
    pixel_values_list, image_grid_thw_list = [], []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([b["labels"], torch.full((pad_len,), CoDDataset.IGNORE_INDEX, dtype=torch.long)])
        )
        attention_mask_list.append(
            torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        )

        if "pixel_values" in b:
            pixel_values_list.append(b["pixel_values"])
        if "image_grid_thw" in b:
            image_grid_thw_list.append(b["image_grid_thw"])

    collated: Dict[str, Any] = {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
    }

    if pixel_values_list:
        collated["pixel_values"] = torch.cat(pixel_values_list, dim=0)
    if image_grid_thw_list:
        collated["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

    return collated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Compute gradient accumulation from global batch size.
    n_gpu = int(os.environ.get("WORLD_SIZE", max(torch.cuda.device_count(), 1)))
    grad_accum = max(1, args.global_batch_size // (args.micro_batch_size * n_gpu))
    effective_bs = args.micro_batch_size * grad_accum * n_gpu
    print(f"GPUs={n_gpu}  micro_bs={args.micro_batch_size}  grad_accum={grad_accum}  effective_bs={effective_bs}")

    # Load processor & model.
    processor = AutoProcessor.from_pretrained(args.model_id)
    model_load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            attn_implementation="flash_attention_2",
            **model_load_kwargs,
        )
        print("Attention backend: flash_attention_2")
    except Exception as flash_err:
        print(
            "flash_attention_2 unavailable "
            f"({type(flash_err).__name__}: {flash_err}). Falling back to sdpa."
        )
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                attn_implementation="sdpa",
                **model_load_kwargs,
            )
            print("Attention backend: sdpa")
        except Exception as sdpa_err:
            print(
                "sdpa unavailable "
                f"({type(sdpa_err).__name__}: {sdpa_err}). Falling back to eager."
            )
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_id,
                attn_implementation="eager",
                **model_load_kwargs,
            )
            print("Attention backend: eager")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Load dataset.
    raw_samples = load_jsonl(args.data_jsonl)
    print(f"Loaded {len(raw_samples)} filtered CoD samples from {args.data_jsonl}")

    hf_dataset = load_dataset("xinlingdedeng/Geo-Thought", split="train")
    train_ds = CoDDataset(
        samples=raw_samples,
        processor=processor,
        hf_dataset=hf_dataset,
        max_input_tokens=args.max_input_tokens,
        max_target_tokens=args.max_target_tokens,
    )

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        fp16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=lambda batch: collate_fn(batch, pad_token_id=pad_token_id),
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    processor.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"Training complete. Model saved to {os.path.join(args.output_dir, 'final')}")


if __name__ == "__main__":
    main()
