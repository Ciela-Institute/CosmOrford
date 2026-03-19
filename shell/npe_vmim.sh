#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-03:00 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ngll
#SBATCH --output=jobout/%x_%A_%a.out


# Running the python script
cd /home/noedia/links/projects/rrg-lplevass/noedia/wl_neurips/CosmOrford
source .venv/bin/activate 
wandb offline
# Define your budget samples
BUDGETS=(100 200 500 1000 2000 5000 10000 20200)

for N in "${BUDGETS[@]}"
do
    echo "------------------------------------------"
    echo "Starting run with max_train_samples = $N"
    echo "------------------------------------------"
    
    # We use dot notation to reach deep into the YAML structure
    uv run trainer fit \
        -c configs/experiments/efficientnet_v2_s_vmim.yaml \
        --data.init_args.max_train_samples=$N \
        --trainer.logger.init_args.name="effnet_v2_s_vmim_budget_$N"
done
