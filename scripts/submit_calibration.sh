#!/bin/bash
#SBATCH --tasks=1
#SBATCH --time=00-02:59
#SBATCH --account=rrg-lplevass
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --job-name=calib
#SBATCH --output=jobout/%x_%j.out
set -euo pipefail
LABEL="${1:?label}"; CKPT="${2:?checkpoints root}"; CONTEXT="${3:?raw|compressed}"
shift 3 2>/dev/null || shift $#
module load python/3.11.5; module load gcc arrow/23.0.1; module load cuda/12.6
source "$HOME/wl-challenge-env/bin/activate"; cd "$HOME/software/CosmOrford"; mkdir -p jobout
export WANDB_MODE=offline
HOLDOUT_PATH="${HOLDOUT_PATH:-/project/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout}"
OUT="$HOME/experiments/hos_cond_sweep/calibration/$LABEL"
python -u scripts/run_calibration.py --checkpoints_path "$CKPT" --context "$CONTEXT" \
  --holdout_path "$HOLDOUT_PATH" --out "$OUT" --label "$LABEL" "$@"
