#!/bin/bash
#
# Submit a standalone RAW-analytical-summary NPE job (new-NDE baseline).
# Runs scripts/run_raw_stats_npe.py (raw _compute_stats context -> MAF, best mode).
# Leaves Justine's run_npe_budget_scan.py untouched.
#
# Usage:
#   sbatch scripts/submit_raw_npe.sh <run_name> <checkpoints_root> [budget] [extra args...]
#   e.g. sbatch scripts/submit_raw_npe.sh raw_l1_best_ctx0 \
#          $HOME/experiments/budget_scan/checkpoints/raw_l1_p5 20200
#   extra args are forwarded to run_raw_stats_npe.py (e.g. --npe_seeds 2 --npe_epochs 20)

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --tasks=1
#SBATCH --time=00-02:59
#SBATCH --account=rrg-lplevass
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --job-name=raw_npe
#SBATCH --output=jobout/%x_%j.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RUN_NAME="${1:?Missing run_name}"
CKPT_ROOT="${2:?Missing checkpoints root (contains budget-N/)}"
BUDGET="${3:-20200}"
shift 3 2>/dev/null || shift $#   # remaining args forwarded to the python script

echo "======================================================================"
echo "CosmOrford – RAW-stats NPE (new-NDE baseline)"
echo "  run_name    : $RUN_NAME"
echo "  checkpoints : $CKPT_ROOT"
echo "  budget      : $BUDGET"
echo "  extra args  : $*"
echo "  node/start  : ${SLURMD_NODENAME:-local} / $(date)"
echo "======================================================================"

module load python/3.11.5
module load gcc arrow/23.0.1
module load cuda/12.6
source "$HOME/wl-challenge-env/bin/activate"
cd "$HOME/software/CosmOrford"
mkdir -p jobout

export WANDB_MODE=offline
HOLDOUT_PATH="${HOLDOUT_PATH:-/project/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout}"
OUT="$HOME/experiments/hos_cond_sweep/results/$RUN_NAME/budget-$BUDGET"

python -u scripts/run_raw_stats_npe.py \
  --checkpoints_path "$CKPT_ROOT" \
  --budget "$BUDGET" \
  --holdout_path "$HOLDOUT_PATH" \
  --out "$OUT" \
  "$@"

echo ""
echo "RAW-stats NPE '$RUN_NAME' finished: $(date)"
echo "Results: $OUT"
