#!/bin/bash
#
# Submit end-to-end PS test pipeline:
#   1) train PS compressor on flat train/validation
#   2) run NPE on holdout train and evaluate FoM on holdout fiducial (10 realizations)
#
# Usage:
#   sbatch scripts/submit_hos_npe_ps_test_pipeline.sh [run_name] [budget] [seed]
#

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --tasks=1
#SBATCH --time=00-18:00
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ps_npe_test
#SBATCH --output=jobout/%x_%j.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RUN_NAME="${1:-hos_npe_ps_test10}"
BUDGET="${2:-20200}"
SEED="${3:-42}"

COMPRESSOR_CONFIG="configs/experiments/hos_npe_compressor_ps.yaml"
NPE_CONFIG="configs/experiments/hos_npe_budget_ps_test10.yaml"

FLAT_DATA_ROOT="/project/rrg-lplevass/shared/wl_chall_data"
HOLDOUT_TRAIN_PATH="/project/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout/train"
HOLDOUT_FIDUCIAL_PATH="/project/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout/fiducial"

echo "======================================================================"
echo "CosmOrford – PS compressor + NPE/FoM test pipeline"
echo "======================================================================"
echo "Run name              : $RUN_NAME"
echo "Budget                : $BUDGET"
echo "Seed                  : $SEED"
echo "Compressor config     : $COMPRESSOR_CONFIG"
echo "NPE config            : $NPE_CONFIG"
echo "Flat data root        : $FLAT_DATA_ROOT"
echo "Holdout train         : $HOLDOUT_TRAIN_PATH"
echo "Holdout fiducial      : $HOLDOUT_FIDUCIAL_PATH"
echo "Node                  : $SLURMD_NODENAME"
echo "Start time            : $(date)"
echo "======================================================================"

module load python/3.11.5
module load gcc arrow/23.0.1
module load cuda/12.6

source "$HOME/wl-challenge-env/bin/activate"

python - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("nflows") is None:
    sys.stderr.write(
        "ERROR: 'nflows' is not installed in $HOME/wl-challenge-env. "
        "Install it once with: source $HOME/wl-challenge-env/bin/activate && pip install nflows\n"
    )
    raise SystemExit(3)
PY

cd "$HOME/software/CosmOrford"
mkdir -p jobout

export WANDB_MODE=offline
export COSMOFORD_SEED="$SEED"

CHECKPOINTS_BASE="${CHECKPOINTS_PATH:-$HOME/experiments/checkpoints/$RUN_NAME}"
CHECKPOINT_DIR="$CHECKPOINTS_BASE/budget-$BUDGET"
NPE_RESULTS_PATH="${NPE_RESULTS_PATH:-$HOME/experiments/npe_results/$RUN_NAME}"
SUMMARIES_CACHE_PATH="${SUMMARIES_CACHE_PATH:-$HOME/experiments/summaries_cache/$RUN_NAME}"

if [ "${CLEAN_PREVIOUS_RUN:-1}" = "1" ]; then
  rm -rf "$CHECKPOINT_DIR"
  rm -rf "$NPE_RESULTS_PATH/budget-$BUDGET"
  rm -rf "$SUMMARIES_CACHE_PATH"
fi
mkdir -p "$CHECKPOINT_DIR" "$NPE_RESULTS_PATH" "$SUMMARIES_CACHE_PATH"
STAGE1_MARKER="$CHECKPOINT_DIR/.stage1_start_marker"
rm -f "$STAGE1_MARKER"
touch "$STAGE1_MARKER"

echo ""
echo ">>> Stage 1/2: training PS compressor"
python -m cosmoford.trainer fit \
  --config "$COMPRESSOR_CONFIG" \
  --seed_everything="$SEED" \
  --data.data_dir="$FLAT_DATA_ROOT" \
  --data.dataset_mode=train \
  --trainer.devices=1 \
  --trainer.default_root_dir="$CHECKPOINT_DIR" \
  --trainer.logger.init_args.save_dir="$CHECKPOINT_DIR" \
  --trainer.enable_progress_bar=false \
  "--trainer.callbacks+={class_path: cosmoford.trainer.EpochProgressPrinter}"

echo ""
echo ">>> Collecting stage-1 checkpoints"
mapfile -t CKPTS < <(find "$CHECKPOINT_DIR" -type f -name '*.ckpt' | sort)
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "No checkpoints found under $CHECKPOINT_DIR; searching fallback logger dirs..."
  mapfile -t FALLBACK_CKPTS < <(
    {
      find "$HOME/software/CosmOrford/neurips-wl-challenge" -type f -name '*.ckpt' -newer "$STAGE1_MARKER" 2>/dev/null || true
      find "$HOME/software/CosmOrford/lightning_logs" -type f -name '*.ckpt' -newer "$STAGE1_MARKER" 2>/dev/null || true
    } | sort -u
  )
  if [ "${#FALLBACK_CKPTS[@]}" -gt 0 ]; then
    mkdir -p "$CHECKPOINT_DIR/stage1_ckpts"
    for src in "${FALLBACK_CKPTS[@]}"; do
      cp -f "$src" "$CHECKPOINT_DIR/stage1_ckpts/"
    done
    mapfile -t CKPTS < <(find "$CHECKPOINT_DIR" -type f -name '*.ckpt' | sort)
  fi
fi
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "ERROR: Stage-1 completed but no checkpoint files were found for run $RUN_NAME" >&2
  exit 2
fi
echo "Found ${#CKPTS[@]} checkpoint file(s)."
printf '  %s\n' "${CKPTS[@]:0:6}"

echo ""
echo ">>> Stage 2/2: NPE + FoM + posterior samples/plots"
python scripts/run_npe_budget_scan.py \
  --checkpoints_path "$CHECKPOINTS_BASE" \
  --npe_results_path "$NPE_RESULTS_PATH" \
  --summaries_cache_path "$SUMMARIES_CACHE_PATH" \
  --holdout_train_path "$HOLDOUT_TRAIN_PATH" \
  --holdout_fiducial_path "$HOLDOUT_FIDUCIAL_PATH" \
  --config "$NPE_CONFIG" \
  --offline

RESULTS_DIR="$NPE_RESULTS_PATH/budget-$BUDGET"
echo ""
echo "Pipeline complete."
echo "Results dir: $RESULTS_DIR"
echo "  - results.json"
echo "  - npe_flow.pt"
echo "  - posterior_samples_norm.npy"
echo "  - posterior_samples_phys.npy"
echo "  - posterior_fiducial_obsXX.png"
echo "End time: $(date)"
