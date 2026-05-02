#!/bin/bash
#
# Submit a compressor-only training job (Stage 1 of the NPE pipeline).
# Shorter and lighter than the full pipeline — designed for better
# backfill scheduling when the queue is congested.
#
# Usage:
#   sbatch scripts/submit_compressor_only.sh \
#     <compressor_config.yaml> <run_name> [budget] [seed]

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --tasks=1
#SBATCH --time=00-01:30
#SBATCH --account=rrg-lplevass
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=hos_compressor
#SBATCH --output=jobout/%x_%j.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

COMPRESSOR_CONFIG="${1:?Missing compressor config path argument}"
RUN_NAME="${2:-hos_compressor_run}"
BUDGET="${3:-20200}"
SEED="${4:-42}"

FLAT_DATA_ROOT="${FLAT_DATA_ROOT:-/project/rrg-lplevass/shared/wl_chall_data}"

echo "======================================================================"
echo "CosmOrford – compressor-only training (Stage 1)"
echo "======================================================================"
echo "Compressor config : $COMPRESSOR_CONFIG"
echo "Run name          : $RUN_NAME"
echo "Budget            : $BUDGET"
echo "Seed              : $SEED"
echo "Flat data root    : $FLAT_DATA_ROOT"
echo "Node              : $SLURMD_NODENAME"
echo "Start time        : $(date)"
echo "======================================================================"

module load python/3.11.5
module load gcc arrow/23.0.1
module load cuda/12.6

source "$HOME/wl-challenge-env/bin/activate"
cd "$HOME/software/CosmOrford"
mkdir -p jobout

export WANDB_MODE=offline
export COSMOFORD_SEED="$SEED"

CHECKPOINT_DIR="${CHECKPOINTS_PATH:-$HOME/experiments/checkpoints/$RUN_NAME}/budget-$BUDGET"
TRAIN_LOG="$CHECKPOINT_DIR/compressor_training.log"

if [ "${CLEAN_PREVIOUS_RUN:-1}" = "1" ]; then
  rm -rf "$CHECKPOINT_DIR"
fi
mkdir -p "$CHECKPOINT_DIR"
STAGE1_MARKER="$CHECKPOINT_DIR/.stage1_start_marker"
rm -f "$STAGE1_MARKER"
touch "$STAGE1_MARKER"

echo ""
echo ">>> Compressor training (30 epochs)"
python -m cosmoford.trainer fit \
  --config "$COMPRESSOR_CONFIG" \
  --seed_everything="$SEED" \
  --data.data_dir="$FLAT_DATA_ROOT" \
  --data.dataset_mode=train \
  --trainer.devices=1 \
  --trainer.default_root_dir="$CHECKPOINT_DIR" \
  --trainer.logger.init_args.save_dir="$CHECKPOINT_DIR" \
  --trainer.enable_progress_bar=false \
  "--trainer.callbacks+={class_path: cosmoford.trainer.EpochProgressPrinter}" \
  2>&1 | tee "$TRAIN_LOG"

echo ""
echo ">>> Saving compressor learning curves"
if ! python - "$TRAIN_LOG" "$CHECKPOINT_DIR/compressor_learning_curves.png" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
if not log_path.exists():
    raise FileNotFoundError(f"Missing training log: {log_path}")

pattern = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|.*?train_loss:\s*([0-9eE+.\-]+)\s+\|.*?val_score:\s*([0-9eE+.\-]+)"
)
epochs, train_loss, val_score = [], [], []
for line in log_path.read_text().splitlines():
    m = pattern.search(line)
    if not m:
        continue
    epochs.append(int(m.group(1)))
    train_loss.append(float(m.group(3)))
    val_score.append(float(m.group(4)))

if not epochs:
    raise RuntimeError(f"No epoch metrics found in {log_path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(8.0, 4.6))
ax1.plot(epochs, train_loss, color="#1f77b4", linewidth=1.8, label="train_loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train loss", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(epochs, val_score, color="#ff7f0e", linewidth=1.8, label="val_score")
ax2.set_ylabel("Validation score", color="#ff7f0e")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")

lines = ax1.get_lines() + ax2.get_lines()
labels = [ln.get_label() for ln in lines]
ax1.legend(lines, labels, loc="best", fontsize=9)
fig.suptitle("Compressor learning curves")
fig.tight_layout()
fig.savefig(out_path, dpi=130)
plt.close(fig)
print(f"Saved compressor learning curves plot: {out_path}")
PY
then
  echo "WARNING: Failed to generate compressor learning curves plot from $TRAIN_LOG" >&2
fi

echo ""
echo ">>> Verifying checkpoints"
mapfile -t CKPTS < <(find "$CHECKPOINT_DIR" -type f -name '*.ckpt' | sort)
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "ERROR: No checkpoint files found under $CHECKPOINT_DIR" >&2
  exit 2
fi
echo "Found ${#CKPTS[@]} checkpoint file(s):"
printf '  %s\n' "${CKPTS[@]:0:6}"

echo ""
echo "======================================================================"
echo "Compressor training complete: $(date)"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Submit NPE stage with:"
echo "  sbatch scripts/submit_npe_light.sh <npe_config.yaml> $RUN_NAME"
echo "======================================================================"
