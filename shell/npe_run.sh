#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-08:00 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=pretrain_gowerstreet
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

uv run scripts/run_npe_budget_scan.py \
    --checkpoints_path=/home/noedia/links/scratch/wl_chall/budget_scan_nbody_final\
    --npe_results_path=/home/noedia/links/scratch/wl_chall/npe\
    --summaries_cache_path= \
    --holdout_path=/home/noedia/links/projects/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout\
    --config=configs/experiments/npe_budget_scan.yaml\
    --budgets=100,500,1000,5000,10000,20200\
    --offline=true