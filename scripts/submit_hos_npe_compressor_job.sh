#!/bin/bash
#
# Submit a stage-1 analytical compressor job with budget-style checkpoint layout.
#
# Usage:
#   sbatch scripts/submit_hos_npe_compressor_job.sh configs/experiments/hos_npe_compressor_ps.yaml ps 20200

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --tasks=1
#SBATCH --time=00-06:00
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --job-name=hos_cmp
#SBATCH --output=jobout/%x_%j.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CONFIG="${1:?Missing compressor config path argument}"
RUN_NAME="${2:-run}"
BUDGET="${3:-20200}"
SEED="${4:-42}"

echo "======================================================================"
echo "CosmOrford – HOS compressor job"
echo "======================================================================"
echo "Config     : $CONFIG"
echo "Run name   : $RUN_NAME"
echo "Budget tag : $BUDGET"
echo "Seed       : $SEED"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "======================================================================"

module load python/3.11.5
module load gcc arrow/23.0.1
module load cuda/12.6

source "$HOME/wl-challenge-env/bin/activate"
cd "$HOME/software/CosmOrford"
mkdir -p jobout

export WANDB_MODE=offline
export COSMOFORD_SEED="$SEED"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/experiments/checkpoints/$RUN_NAME/budget-$BUDGET}"
mkdir -p "$CHECKPOINT_DIR"

python -m cosmoford.trainer fit \
  --config "$CONFIG" \
  --seed_everything="$SEED" \
  --trainer.devices=1 \
  --trainer.default_root_dir="$CHECKPOINT_DIR" \
  --trainer.enable_progress_bar=false \
  "--trainer.callbacks+={class_path: cosmoford.trainer.EpochProgressPrinter}"

echo ""
echo "======================================================================"
echo "Job finished: $(date)"
echo "======================================================================"
