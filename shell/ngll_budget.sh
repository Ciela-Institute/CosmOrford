#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-08:00 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ngll
#SBATCH --array=0-7
#SBATCH --output=jobout/%x_%A_%a.out


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
BUDGETS=(100 200 500 1000 2000 5000 10000 20200)

CURRENT_BUDGET=${BUDGETS[$SLURM_ARRAY_TASK_ID]}

echo "Running job for budget: $CURRENT_BUDGET"

# We use dot notation to reach deep into the YAML structure
trainer fit \
    -c configs/finetune_from_pretrain_nopatch_logp.yaml \
    --data.init_args.max_train_samples=$CURRENT_BUDGET \
    --trainer.logger.init_args.name="effnet_v2_s_nbody_budget_final_$CURRENT_BUDGET" \
    --trainer.logger.init_args.save_dir="$SAVE_DIR/budget_scan_nbody_final/budget-$CURRENT_BUDGET"


    