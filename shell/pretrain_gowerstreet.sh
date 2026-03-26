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

# Running experiment.
uv run trainer fit \
    -c configs/experiments/pretrain_gowerstreet_nopatch_logp.yaml\
    --trainer.logger.init_args.name="effnet_v2_s_gowerstreet_budget_final_$CURRENT_BUDGET" \
    --trainer.logger.init_args.save_dir="$SAVE_DIR/gowerstreet/pretrain/budget-$CURRENT_BUDGET"

