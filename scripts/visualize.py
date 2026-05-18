import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate clean LatentEuclid plots and metric summaries.")
    parser.add_argument("--eval-json", type=str, default=None, help="Path to eval_*.json from eval/e2e.py")
    parser.add_argument("--dataset-jsonl", type=str, default="data/geothoughts_verified.jsonl", help="Dataset JSONL used to recover question text.")
    parser.add_argument("--latent-summary-json", type=str, default=None, help="Optional latent summary JSON from analyze_v12_latent_errors_colab.py")
    parser.add_argument("--metrics-jsonl", type=str, default=None, help="Optional training metrics JSONL for loss curves.")
    parser.add_argument("--output-dir", type=str, default="artifacts/figures", help="Directory to write figures and summaries.")
    parser.add_argument("--cgap-field", type=str, default="question_length_words", help="Complexity field to use for C-Gap if present.")
    parser.add_argument("--cgap-low-threshold", type=float, default=None, help="Inclusive low-complexity cutoff.")
    parser.add_argument("--cgap-high-threshold", type=float, default=None, help="Inclusive high-complexity cutoff.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def load_jsonl(path):
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_question_lookup(dataset_jsonl):
    lookup = {}
    for item in load_jsonl(dataset_jsonl):
        image_key = os.path.basename(item["image_path"])
        question = item.get("question", "")
        lookup[image_key] = {
            "question": question,
            "question_length_words": len(question.replace("<image>", " ").split()),
            "question_length_chars": len(question),
        }
    return lookup


def attach_question_metadata(eval_rows, question_lookup):
    enriched = []
    for row in eval_rows:
        image_key = os.path.basename(row.get("image", ""))
        merged = dict(row)
        merged.update(question_lookup.get(image_key, {}))
        enriched.append(merged)
    return enriched


def summarize_accuracy(eval_rows):
    total = len(eval_rows)
    correct = sum(1 for row in eval_rows if row.get("is_correct"))
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else float("nan"),
    }


def compute_cgap(eval_rows, field, low_threshold=None, high_threshold=None):
    values = [row.get(field) for row in eval_rows if isinstance(row.get(field), (int, float))]
    if not values:
        return None

    if low_threshold is None:
        low_threshold = min(values)
    if high_threshold is None:
        high_threshold = max(values)

    low_bucket = [row for row in eval_rows if isinstance(row.get(field), (int, float)) and row[field] <= low_threshold]
    high_bucket = [row for row in eval_rows if isinstance(row.get(field), (int, float)) and row[field] >= high_threshold]
    if not low_bucket or not high_bucket:
        return None

    low_acc = sum(1 for row in low_bucket if row.get("is_correct")) / len(low_bucket)
    high_acc = sum(1 for row in high_bucket if row.get("is_correct")) / len(high_bucket)
    return {
        "field": field,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "low_count": len(low_bucket),
        "high_count": len(high_bucket),
        "low_accuracy": low_acc,
        "high_accuracy": high_acc,
        "c_gap": low_acc - high_acc,
    }


def plot_accuracy_breakdown(summary, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["accuracy"], [summary["accuracy"]], color="#1f77b4")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("End-to-End Accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy.png"), dpi=150)
    plt.close(fig)


def plot_cgap(cgap, output_dir):
    if cgap is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["low complexity", "high complexity"], [cgap["low_accuracy"], cgap["high_accuracy"]], color=["#2ca02c", "#d62728"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"C-Gap on {cgap['field']}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "c_gap.png"), dpi=150)
    plt.close(fig)


def plot_latent_summary(latent_summary, output_dir):
    if latent_summary is None:
        return

    labels = ["correct", "failed"]
    mse_values = [latent_summary[label]["avg_mse"] for label in labels]
    cosine_values = [latent_summary[label]["avg_cosine"] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, mse_values, color=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("Latent MSE")
    axes[0].set_ylabel("MSE")

    axes[1].bar(labels, cosine_values, color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Latent Cosine Similarity")
    axes[1].set_ylabel("Cosine")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "latent_metrics.png"), dpi=150)
    plt.close(fig)


def plot_training_curves(metrics_rows, output_dir):
    if not metrics_rows:
        return

    steps = list(range(1, len(metrics_rows) + 1))
    train_loss = [row.get("train/total_loss") for row in metrics_rows]
    cosine = [row.get("loss/cosine_angular") for row in metrics_rows]
    huber = [row.get("loss/huber_magnitude") for row in metrics_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if any(value is not None for value in train_loss):
        axes[0].plot(steps, train_loss, color="#1f77b4")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Step")

    if any(value is not None for value in cosine) or any(value is not None for value in huber):
        if any(value is not None for value in cosine):
            axes[1].plot(steps, cosine, label="cosine", color="#2ca02c")
        if any(value is not None for value in huber):
            axes[1].plot(steps, huber, label="huber", color="#d62728")
        axes[1].legend()
        axes[1].set_title("Alignment Components")
        axes[1].set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def write_summary(output_dir, accuracy_summary, cgap_summary, latent_summary):
    summary_path = os.path.join(output_dir, "summary.json")
    payload = {
        "accuracy": accuracy_summary,
        "c_gap": cgap_summary,
        "latent": latent_summary,
    }
    with open(summary_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    accuracy_summary = None
    cgap_summary = None
    latent_summary = load_json(args.latent_summary_json) if args.latent_summary_json else None

    if args.eval_json:
        eval_rows = load_json(args.eval_json)
        question_lookup = build_question_lookup(args.dataset_jsonl)
        eval_rows = attach_question_metadata(eval_rows, question_lookup)
        accuracy_summary = summarize_accuracy(eval_rows)
        cgap_summary = compute_cgap(
            eval_rows,
            args.cgap_field,
            low_threshold=args.cgap_low_threshold,
            high_threshold=args.cgap_high_threshold,
        )
        plot_accuracy_breakdown(accuracy_summary, args.output_dir)
        plot_cgap(cgap_summary, args.output_dir)

    if latent_summary:
        plot_latent_summary(latent_summary, args.output_dir)

    if args.metrics_jsonl:
        plot_training_curves(load_jsonl(args.metrics_jsonl), args.output_dir)

    write_summary(args.output_dir, accuracy_summary, cgap_summary, latent_summary)
    print(f"Saved figures and summary to {args.output_dir}")


if __name__ == "__main__":
    main()