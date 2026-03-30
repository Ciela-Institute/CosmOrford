#!/bin/bash
#
# Submit an NPE/FoM job for HOS representative configs.
#
# Usage:
#   sbatch scripts/submit_hos_npe_job.sh configs/experiments/hos_npe_budget_ps.yaml ps

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --tasks=1
#SBATCH --time=00-08:00
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --job-name=hos_npe
#SBATCH --output=jobout/%x_%j.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

NPE_CONFIG="${1:?Missing NPE config path argument}"
RUN_NAME="${2:-run}"

echo "======================================================================"
echo "CosmOrford – HOS NPE job"
echo "======================================================================"
echo "NPE Config : $NPE_CONFIG"
echo "Run name   : $RUN_NAME"
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
export WANDB_RUN_GROUP="hos_npe_representative"

# Update these paths for your cluster layout as needed.
CHECKPOINTS_PATH="${CHECKPOINTS_PATH:-$HOME/experiments/checkpoints/$RUN_NAME}"
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
