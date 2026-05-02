#!/bin/bash
#
# Submit a lightweight NPE/FoM job using an already-trained compressor.
# Intended for quick re-runs where Stage 1 (compressor) is already done.
#
# Usage:
#   sbatch scripts/submit_npe_light.sh <npe_config.yaml> <run_name> [checkpoints_path]
#
# The third argument overrides the compressor checkpoint directory so you can
# point at a pre-existing run without renaming things.

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --tasks=1
#SBATCH --time=00-01:00
#SBATCH --account=rrg-lplevass
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=hos_npe_light
#SBATCH --output=jobout/%x_%j.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

NPE_CONFIG="${1:?Missing NPE config path argument}"
RUN_NAME="${2:-run}"
CHECKPOINTS_PATH="${3:-${CHECKPOINTS_PATH:-$HOME/experiments/checkpoints/$RUN_NAME}}"

echo "======================================================================"
echo "CosmOrford – HOS NPE light job (NPE-only)"
echo "======================================================================"
echo "NPE Config        : $NPE_CONFIG"
echo "Run name          : $RUN_NAME"
echo "Checkpoints path  : $CHECKPOINTS_PATH"
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
export WANDB_RUN_GROUP="hos_npe_representative"

NPE_RESULTS_PATH="${NPE_RESULTS_PATH:-$HOME/experiments/npe_results/$RUN_NAME}"
SUMMARIES_CACHE_PATH="${SUMMARIES_CACHE_PATH:-$HOME/experiments/summaries_cache/$RUN_NAME}"
HOLDOUT_PATH="${HOLDOUT_PATH:-/project/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout}"

python scripts/run_npe_budget_scan.py \
  --checkpoints_path "$CHECKPOINTS_PATH" \
  --npe_results_path "$NPE_RESULTS_PATH" \
  --summaries_cache_path "$SUMMARIES_CACHE_PATH" \
  --holdout_path "$HOLDOUT_PATH" \
  --config "$NPE_CONFIG" \
  --offline

echo ""
echo "======================================================================"
echo "Job finished: $(date)"
echo "======================================================================"
