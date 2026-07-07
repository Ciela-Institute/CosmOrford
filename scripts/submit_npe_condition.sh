#!/bin/bash
#
# Run ONE stage-2 NPE condition on an already-trained (frozen) compressor.
# Used by scripts/submit_hos_conditioning_sweep.sh for the conditioning sweep
# (plan: pure-hugging-marble.md). Does NOT train a compressor and does NOT edit
# the analysis code — it only re-runs run_npe_budget_scan.py with swept knobs.
#
# Usage:
#   sbatch scripts/submit_npe_condition.sh \
#     <npe_config.yaml> <run_name> <checkpoints_path> <inference_mode> <ctxnorm:0|1> [budget]
#
#   inference_mode : best | ensemble        (-> --inference_mode)
#   ctxnorm        : 1 = context norm ON (-> --context_normalization 1); 0 = OFF (flag omitted)
#
# Results/cache are derived from <run_name> so every condition is isolated.

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --tasks=1
#SBATCH --time=00-02:59
#SBATCH --account=rrg-lplevass
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --job-name=hos_cond
#SBATCH --output=jobout/%x_%j.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

NPE_CONFIG="${1:?Missing NPE config path}"
RUN_NAME="${2:?Missing run_name}"
CHECKPOINTS_PATH="${3:?Missing checkpoints_path (compressor root containing budget-N/)}"
MODE="${4:?Missing inference_mode (best|ensemble)}"
CTXNORM="${5:?Missing ctxnorm (0|1)}"
BUDGET="${6:-20200}"

echo "======================================================================"
echo "CosmOrford – HOS NPE conditioning-sweep condition"
echo "  config          : $NPE_CONFIG"
echo "  run_name        : $RUN_NAME"
echo "  checkpoints     : $CHECKPOINTS_PATH"
echo "  inference_mode  : $MODE"
echo "  context_norm    : $CTXNORM"
echo "  budget          : $BUDGET"
echo "  node/start      : ${SLURMD_NODENAME:-local} / $(date)"
echo "======================================================================"

module load python/3.11.5
module load gcc arrow/23.0.1
module load cuda/12.6

source "$HOME/wl-challenge-env/bin/activate"
cd "$HOME/software/CosmOrford"
mkdir -p jobout

export WANDB_MODE=offline

BASE="$HOME/experiments/hos_cond_sweep"
NPE_RESULTS_PATH="$BASE/results/$RUN_NAME"
SUMMARIES_CACHE_PATH="$BASE/cache/$RUN_NAME"
HOLDOUT_PATH="${HOLDOUT_PATH:-/project/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout}"
FLAT_DATASET_PATH="${FLAT_DATASET_PATH:-/project/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-flat}"
mkdir -p "$NPE_RESULTS_PATH" "$SUMMARIES_CACHE_PATH"

# Context normalization: type=bool argparse => pass "1" to enable, OMIT to disable
# (passing "False" would evaluate truthy). Same reasoning for --offline.
CTX_ARG=()
[ "$CTXNORM" = "1" ] && CTX_ARG=(--context_normalization 1)

# Calibration (TARP+MIRA) runs only when --flat_dataset_path is passed and present.
CALIB_ARG=()
if [ -d "$FLAT_DATASET_PATH" ]; then
  CALIB_ARG=(--flat_dataset_path "$FLAT_DATASET_PATH")
else
  echo "WARNING: flat dataset not found at $FLAT_DATASET_PATH -> skipping TARP/MIRA calibration"
fi

python scripts/run_npe_budget_scan.py \
  --checkpoints_path "$CHECKPOINTS_PATH" \
  --npe_results_path "$NPE_RESULTS_PATH" \
  --summaries_cache_path "$SUMMARIES_CACHE_PATH" \
  --holdout_path "$HOLDOUT_PATH" \
  --config "$NPE_CONFIG" \
  --budgets "$BUDGET" \
  --inference_mode "$MODE" \
  "${CTX_ARG[@]}" "${CALIB_ARG[@]}" \
  --offline 1

echo ""
echo "Condition '$RUN_NAME' finished: $(date)"
echo "Results: $NPE_RESULTS_PATH/budget-$BUDGET"
