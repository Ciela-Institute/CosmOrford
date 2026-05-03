#!/bin/bash
set -euo pipefail

VMIM_BASE="$HOME/experiments/vmim_explore"

# ── vmim_s ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/vmim_explore/checkpoints/l1_vmim_s" sbatch --parsable --time=00-01:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/vmim_explore/l1_vmim_s_compressor.yaml l1_vmim_s)
echo "vmim_s: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/vmim_explore/npe_results/l1_vmim_s" SUMMARIES_CACHE_PATH="$HOME/experiments/vmim_explore/summaries_cache/l1_vmim_s" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/vmim_explore/l1_vmim_s_npe.yaml l1_vmim_s $HOME/experiments/vmim_explore/checkpoints/l1_vmim_s

# ── vmim_m ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/vmim_explore/checkpoints/l1_vmim_m" sbatch --parsable --time=00-01:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/vmim_explore/l1_vmim_m_compressor.yaml l1_vmim_m)
echo "vmim_m: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/vmim_explore/npe_results/l1_vmim_m" SUMMARIES_CACHE_PATH="$HOME/experiments/vmim_explore/summaries_cache/l1_vmim_m" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/vmim_explore/l1_vmim_m_npe.yaml l1_vmim_m $HOME/experiments/vmim_explore/checkpoints/l1_vmim_m

# ── vmim_s_long ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/vmim_explore/checkpoints/l1_vmim_s_long" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/vmim_explore/l1_vmim_s_long_compressor.yaml l1_vmim_s_long)
echo "vmim_s_long: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/vmim_explore/npe_results/l1_vmim_s_long" SUMMARIES_CACHE_PATH="$HOME/experiments/vmim_explore/summaries_cache/l1_vmim_s_long" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/vmim_explore/l1_vmim_s_long_npe.yaml l1_vmim_s_long $HOME/experiments/vmim_explore/checkpoints/l1_vmim_s_long

# ── vmim_m_long ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/vmim_explore/checkpoints/l1_vmim_m_long" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/vmim_explore/l1_vmim_m_long_compressor.yaml l1_vmim_m_long)
echo "vmim_m_long: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/vmim_explore/npe_results/l1_vmim_m_long" SUMMARIES_CACHE_PATH="$HOME/experiments/vmim_explore/summaries_cache/l1_vmim_m_long" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/vmim_explore/l1_vmim_m_long_npe.yaml l1_vmim_m_long $HOME/experiments/vmim_explore/checkpoints/l1_vmim_m_long

# ── vmim_l_long ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/vmim_explore/checkpoints/l1_vmim_l_long" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/vmim_explore/l1_vmim_l_long_compressor.yaml l1_vmim_l_long)
echo "vmim_l_long: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/vmim_explore/npe_results/l1_vmim_l_long" SUMMARIES_CACHE_PATH="$HOME/experiments/vmim_explore/summaries_cache/l1_vmim_l_long" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/vmim_explore/l1_vmim_l_long_npe.yaml l1_vmim_l_long $HOME/experiments/vmim_explore/checkpoints/l1_vmim_l_long

# ── vmim_deep_m ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/vmim_explore/checkpoints/l1_vmim_deep_m" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/vmim_explore/l1_vmim_deep_m_compressor.yaml l1_vmim_deep_m)
echo "vmim_deep_m: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/vmim_explore/npe_results/l1_vmim_deep_m" SUMMARIES_CACHE_PATH="$HOME/experiments/vmim_explore/summaries_cache/l1_vmim_deep_m" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/vmim_explore/l1_vmim_deep_m_npe.yaml l1_vmim_deep_m $HOME/experiments/vmim_explore/checkpoints/l1_vmim_deep_m

# ── vmim_cosine ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/vmim_explore/checkpoints/l1_vmim_cosine" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/vmim_explore/l1_vmim_cosine_compressor.yaml l1_vmim_cosine)
echo "vmim_cosine: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/vmim_explore/npe_results/l1_vmim_cosine" SUMMARIES_CACHE_PATH="$HOME/experiments/vmim_explore/summaries_cache/l1_vmim_cosine" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/vmim_explore/l1_vmim_cosine_npe.yaml l1_vmim_cosine $HOME/experiments/vmim_explore/checkpoints/l1_vmim_cosine

# ── vmim_lr3 ──
COMP_JOB=$(CHECKPOINTS_PATH="$HOME/experiments/vmim_explore/checkpoints/l1_vmim_lr3" sbatch --parsable --time=00-02:30 --gpus-per-node=h100:1 scripts/submit_compressor_only.sh configs/experiments/vmim_explore/l1_vmim_lr3_compressor.yaml l1_vmim_lr3)
echo "vmim_lr3: compressor job $COMP_JOB"
NPE_RESULTS_PATH="$HOME/experiments/vmim_explore/npe_results/l1_vmim_lr3" SUMMARIES_CACHE_PATH="$HOME/experiments/vmim_explore/summaries_cache/l1_vmim_lr3" sbatch --parsable --dependency=afterok:$COMP_JOB --gpus-per-node=h100:1 scripts/submit_npe_light.sh configs/experiments/vmim_explore/l1_vmim_lr3_npe.yaml l1_vmim_lr3 $HOME/experiments/vmim_explore/checkpoints/l1_vmim_lr3

