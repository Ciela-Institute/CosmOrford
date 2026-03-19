#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-03:00 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=pretrain_gowerstreet
#SBATCH --output=jobout/%x_%A_%a.out


# Running the python script
cd /home/noedia/links/projects/rrg-lplevass/noedia/wl_neurips/ciela_branch/CosmOrford
source .venv/bin/activate 
wandb offline


export HF_HOME="~/links/scratch/cache"
# We use dot notation to reach deep into the YAML structure
uv run trainer fit \
    -c configs/experiments/pretrain_gowerstreet_nopatch_logp.yaml
