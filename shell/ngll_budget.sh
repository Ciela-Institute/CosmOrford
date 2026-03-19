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
cd /home/noedia/links/projects/rrg-lplevass/noedia/wl_neurips/ciela_branch/CosmOrford
source .venv/bin/activate 
wandb offline
# Define your budget samples
BUDGETS=(100 200 500 1000 2000 5000 10000 20200)

CURRENT_BUDGET=${BUDGETS[$SLURM_ARRAY_TASK_ID]}

echo "Running job for budget: $CURRENT_BUDGET"

# We use dot notation to reach deep into the YAML structure
uv run trainer fit \
    -c configs/experiments/efficientnet_v2_s_logp.yaml \
    --data.init_args.max_train_samples=$CURRENT_BUDGET \
    --trainer.logger.init_args.name="effnet_v2_s_nbody_budget_final_$CURRENT_BUDGET" \
    --trainer.logger.init_args.save_dir="/home/noedia/links/scratch/wl_chall/models/budget_scan_nbody_final/budget-$CURRENT_BUDGET"


