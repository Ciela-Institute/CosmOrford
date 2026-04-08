#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-03:00 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=run_npe_nbody
#SBATCH --output=jobout/%x_%A_%a.out
#SBATCH --array=0-5

# Running the python script
module load python gcc arrow
source .venv/bin/activate 
wandb offline

# Getting user-level config from global_config.yaml
WDIR=$(yq -r '.wdir' global_config.yaml)
SAVE_DIR=$(yq -r '.save_dir' global_config.yaml)

# Changes Hugging face cache directory  
export HF_HOME="~/links/scratch/cache"

# Going to the repository directory
cd $WDIR

# Define your budget samples
BUDGETS=(100 500 1000 5000 10000 20200)

CURRENT_BUDGET=${BUDGETS[$SLURM_ARRAY_TASK_ID]}

python scripts/run_npe_budget_scan.py \
    --checkpoints_path=/home/jlinhart/links/scratch/wl_chall/budget_scan_nbody_final\
    --npe_results_path=/home/jlinhart/links/scratch/wl_chall/npe_results\
    --summaries_cache_path=/home/jlinhart/links/scratch/wl_chall/summaries_cache\
    --holdout_path=/home/jlinhart/links/projects/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout\
    --config=configs/experiments/npe_budget_scan.yaml\
    --budgets=$CURRENT_BUDGET\
    --offline=true