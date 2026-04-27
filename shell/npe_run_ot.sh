#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-08:00 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100_20gb
#SBATCH --job-name=run_npe_ot_lognormal
#SBATCH --output=jobout/%x_%A_%a.out

# Running the python script
source ../.venv/bin/activate 
wandb offline

# Getting user-level config from global_config.yaml
WDIR=$(yq -r '.wdir' ../global_config.yaml)
SAVE_DIR=$(yq -r '.save_dir' ../global_config.yaml)

# Changes Hugging face cache directory  
export HF_HOME="~/links/scratch/cache"

# Going to the repository directory
cd $WDIR

DSET_NAME=ot_lognormal # dataset used for pretraining
EXP_DIR=/home/noedia/links/scratch/wl_chall/
OT_BUDGET=5000

python scripts/run_npe_budget_scan.py \
    --checkpoints_path="$EXP_DIR/finetune_$DSET_NAME/ot_budget_$OT_BUDGET/"\
    --npe_results_path="$EXP_DIR/finetune_$DSET_NAME/ot_budget_$OT_BUDGET/npe_results"\
    --summaries_cache_path="$EXP_DIR/finetune_$DSET_NAME/ot_budget_$OT_BUDGET/" \
    --holdout_path="/home/noedia/links/projects/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout"\
    --config=configs/experiments/npe_budget_scan.yaml\
    --offline=true