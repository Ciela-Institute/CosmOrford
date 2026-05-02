#!/bin/bash
set -euo pipefail

EXPLORE_BASE="$HOME/experiments/compressor_explore"

# ── baseline_repro ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_baseline_repro" sbatch --parsable --time=00-01:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_baseline_repro_compressor.yaml l1_baseline_repro)
echo "baseline_repro: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_baseline_repro" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_baseline_repro" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_baseline_repro_npe.yaml l1_baseline_repro $HOME/experiments/compressor_explore/checkpoints/l1_baseline_repro

# ── flow_s ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_flow_s" sbatch --parsable --time=00-01:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_flow_s_compressor.yaml l1_flow_s)
echo "flow_s: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_flow_s" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_flow_s" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_flow_s_npe.yaml l1_flow_s $HOME/experiments/compressor_explore/checkpoints/l1_flow_s

# ── flow_m ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_flow_m" sbatch --parsable --time=00-01:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_flow_m_compressor.yaml l1_flow_m)
echo "flow_m: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_flow_m" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_flow_m" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_flow_m_npe.yaml l1_flow_m $HOME/experiments/compressor_explore/checkpoints/l1_flow_m

# ── flow_l ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_flow_l" sbatch --parsable --time=00-01:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_flow_l_compressor.yaml l1_flow_l)
echo "flow_l: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_flow_l" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_flow_l" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_flow_l_npe.yaml l1_flow_l $HOME/experiments/compressor_explore/checkpoints/l1_flow_l

# ── flow_s_long ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_flow_s_long" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_flow_s_long_compressor.yaml l1_flow_s_long)
echo "flow_s_long: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_flow_s_long" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_flow_s_long" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_flow_s_long_npe.yaml l1_flow_s_long $HOME/experiments/compressor_explore/checkpoints/l1_flow_s_long

# ── deep_flow_s ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_deep_flow_s" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_deep_flow_s_compressor.yaml l1_deep_flow_s)
echo "deep_flow_s: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_deep_flow_s" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_deep_flow_s" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_deep_flow_s_npe.yaml l1_deep_flow_s $HOME/experiments/compressor_explore/checkpoints/l1_deep_flow_s

# ── deep_flow_m ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_deep_flow_m" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_deep_flow_m_compressor.yaml l1_deep_flow_m)
echo "deep_flow_m: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_deep_flow_m" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_deep_flow_m" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_deep_flow_m_npe.yaml l1_deep_flow_m $HOME/experiments/compressor_explore/checkpoints/l1_deep_flow_m

# ── deep_gauss ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_deep_gauss" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_deep_gauss_compressor.yaml l1_deep_gauss)
echo "deep_gauss: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_deep_gauss" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_deep_gauss" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_deep_gauss_npe.yaml l1_deep_gauss $HOME/experiments/compressor_explore/checkpoints/l1_deep_gauss

# ── nodrop_flow ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_nodrop_flow" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_nodrop_flow_compressor.yaml l1_nodrop_flow)
echo "nodrop_flow: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_nodrop_flow" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_nodrop_flow" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_nodrop_flow_npe.yaml l1_nodrop_flow $HOME/experiments/compressor_explore/checkpoints/l1_nodrop_flow

# ── score_loss ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_score_loss" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_score_loss_compressor.yaml l1_score_loss)
echo "score_loss: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_score_loss" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_score_loss" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_score_loss_npe.yaml l1_score_loss $HOME/experiments/compressor_explore/checkpoints/l1_score_loss

# ── dim16_flow ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_dim16_flow" sbatch --parsable --time=00-01:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_dim16_flow_compressor.yaml l1_dim16_flow)
echo "dim16_flow: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_dim16_flow" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_dim16_flow" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_dim16_flow_npe.yaml l1_dim16_flow $HOME/experiments/compressor_explore/checkpoints/l1_dim16_flow

# ── dim32_flow ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/compressor_explore/checkpoints/l1_dim32_flow" sbatch --parsable --time=00-01:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/compressor_explore/l1_dim32_flow_compressor.yaml l1_dim32_flow)
echo "dim32_flow: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/compressor_explore/npe_results/l1_dim32_flow" SUMMARIES_CACHE_PATH="$HOME/experiments/compressor_explore/summaries_cache/l1_dim32_flow" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/compressor_explore/l1_dim32_flow_npe.yaml l1_dim32_flow $HOME/experiments/compressor_explore/checkpoints/l1_dim32_flow

