#!/usr/bin/env bash
# Training pipeline for experiment v13_contrast
# Usage:
#   bash scripts/train_v13_contrast.sh           # auto-detect GPUs
#   bash scripts/train_v13_contrast.sh 2         # explicit GPU count
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG="configs/v13_contrast.yaml"
EXPERIMENT="v13_contrast"
FILTER_CSV="ChainOfDraft/geothought_cod_full_report.csv"
# Optional: export TARGETS_HF_REPO="your-username/your-model-repo" before running.
TARGETS_HF_REPO="${TARGETS_HF_REPO:-}"
NUM_GPUS="${1:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

# ── Resolve project root ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# WandB defaults (branch-aware). Override these before running if needed.
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
export GIT_BRANCH
export WANDB_PROJECT="${WANDB_PROJECT:-LatentEuclid}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${EXPERIMENT}_${GIT_BRANCH}}"

echo "========================================"
echo " v13_contrast training pipeline"
echo " Config   : $CONFIG"
echo " Filter   : $FILTER_CSV"
echo " Targets HF repo: ${TARGETS_HF_REPO:-<none>}"
echo " Git branch: $GIT_BRANCH"
echo " WandB project/group: $WANDB_PROJECT / $WANDB_RUN_GROUP"
echo " GPUs     : $NUM_GPUS"
echo "========================================"

# ── Step 1: Pre-generate manifold target tensors ─────────────────────────────
echo ""
echo "[Step 1/2] Generating manifold target tensors..."
if [ -n "$TARGETS_HF_REPO" ]; then
    python -m data.build_dynamic_manifold \
        --config "$CONFIG" \
        --experiment_name "$EXPERIMENT" \
        --filter_csv "$FILTER_CSV" \
        --hf_repo "$TARGETS_HF_REPO"
else
    python -m data.build_dynamic_manifold \
        --config "$CONFIG" \
        --experiment_name "$EXPERIMENT" \
        --filter_csv "$FILTER_CSV"
fi

echo "[Step 1/2] Done."

# ── Step 2: Train X-Encoder ───────────────────────────────────────────────────
echo ""
echo "[Step 2/2] Launching X-Encoder training on $NUM_GPUS GPU(s)..."

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        -m training.train_x_encoder \
        --config "$CONFIG" \
        --experiment_name "$EXPERIMENT"
else
    python -m training.train_x_encoder \
        --config "$CONFIG" \
        --experiment_name "$EXPERIMENT"
fi

echo ""
echo "[Step 2/2] Done."
echo "========================================"
echo " Training complete. Checkpoints saved to:"
echo "   checkpoints/$EXPERIMENT/"
echo "========================================"
