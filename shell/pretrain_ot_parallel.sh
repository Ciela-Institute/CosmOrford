#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-08:00 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100_40gb
#SBATCH --job-name=pretrain_otlognormal
#SBATCH --array=0-4
#SBATCH --output=jobout/%x_%A_%a.out


# Running the python script
module load python gcc arrow
source ../.venv/bin/activate 
wandb offline

# Getting user-level config from global_config.yaml
WDIR=$(yq -r '.wdir' ../global_config.yaml)
SAVE_DIR=$(yq -r '.save_dir' ../global_config.yaml)

# Changes Hugging face cache directory  
export HF_HOME="~/links/scratch/cache"

# Going to the repository directory
cd $WDIR
DSET=ot_lognormal
DSET_MODE=ot-lognormal

# Define your budget samples
BUDGETS=(100 1000 5000 10000 20200) # CHANGE THIS ! 
BUDGET=${BUDGETS[$SLURM_ARRAY_TASK_ID]}


# We use dot notation to reach deep into the YAML structure
uv run trainer fit \
    -c configs/experiments/pretrain_${DSET}_nopatch_logp.yaml \
    --data.init_args.dataset_mode=$DSET_MODE \
    --trainer.logger.init_args.name="effnet_v2_s_pretrain_${DSET}_${BUDGET}" \
    --trainer.logger.init_args.save_dir="$SAVE_DIR/pretrain_$DSET/budget-$BUDGET" \
    --trainer.callbacks='[
    {"class_path": "LearningRateMonitor", "init_args": {"logging_interval": "step"}},
    {"class_path": "EMAWeightAveraging"},
    {"class_path": "ModelCheckpoint",
     "init_args": {
       "dirpath": "'"$SAVE_DIR"'/'"pretrain_$DSET"'/checkpoints",
       "monitor": "val_log_prob",
       "mode": "min",
       "save_top_k": 3,
       "save_last": true,
       "filename": "{step}-{val_log_prob:.4f}"
     }
    }
  ]'


    