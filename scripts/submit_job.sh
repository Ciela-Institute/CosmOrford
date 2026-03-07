#!/bin/bash
#
# Submit a training job to the Rorqual (Compute Canada) cluster.
#
# Usage:
#   sbatch scripts/submit_job.sh configs/experiments/ps_hos_only.yaml
#   sbatch --export=ALL,CONFIG=configs/experiments/ps_hos_only.yaml scripts/submit_job.sh

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --tasks=1
#SBATCH --time=00-12:00          # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12       # matches num_workers in data config
#SBATCH --gpus-per-node=1
#SBATCH --job-name=cosmoford
#SBATCH --output=jobout/%x_%j.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config file ───────────────────────────────────────────────────────────────
CONFIG="${1:-${CONFIG:-configs/experiments/ps_hos_only.yaml}}"

echo "======================================================================"
echo "CosmOrford – Training job"
echo "======================================================================"
echo "Config     : $CONFIG"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "======================================================================"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11.5
module load gcc arrow/23.0.1
module load cuda/12.6

source "$HOME/wl-challenge-env/bin/activate"

# ── Working directory ─────────────────────────────────────────────────────────
cd "$HOME/software/CosmOrford"

mkdir -p jobout

# Compute nodes have no internet – save WandB run locally and sync afterwards.
export WANDB_MODE=offline

# ── Run training ──────────────────────────────────────────────────────────────
# Disable tqdm progress bar (uses \r overwrites that corrupt log files) and
# inject EpochProgressPrinter which emits one clean line per epoch instead.
trainer fit \
    --config "$CONFIG" \
    --trainer.devices=1 \
    --trainer.enable_progress_bar=false \
    "--trainer.callbacks+={class_path: cosmoford.trainer.EpochProgressPrinter}"

echo ""
echo "======================================================================"
echo "Job finished: $(date)"
echo "======================================================================"
