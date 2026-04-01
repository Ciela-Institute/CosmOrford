#!/bin/bash
# SLURM parameters for every job submitted
#SBATCH --tasks=1
#SBATCH --time=00-08:00 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=finetune_gowerstreet
#SBATCH --array=0-5
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

# Define your budget samples
BUDGETS=(100 500 1000 5000 10000 20200)

CURRENT_BUDGET=${BUDGETS[$SLURM_ARRAY_TASK_ID]}

echo "Running job for budget: $CURRENT_BUDGET"

# Running experiment.
uv run trainer fit \
    -c configs/experiments/finetune_from_pretrain_gowerstreet_nopatch_logp.yaml\
    --data.init_args.max_train_samples=$CURRENT_BUDGET \
    --model.init_args.pretrained_checkpoint_path=/home/noedia/links/scratch/wl_chall/pretrain_gowerstreet/checkpoints/last.ckpt \
    --trainer.logger.init_args.name="effnet_v2_s_gowerstreet_$CURRENT_BUDGET" \
    --trainer.logger.init_args.save_dir="$SAVE_DIR/budget_scan_gowerstreet/budget-$CURRENT_BUDGET" \
    --trainer.callbacks='[
    {"class_path": "LearningRateMonitor", "init_args": {"logging_interval": "step"}},
    {"class_path": "EMAWeightAveraging"},
    {"class_path": "ModelCheckpoint",
     "init_args": {
       "dirpath": "'"$SAVE_DIR"'/budget_scan_gowerstreet/budget-'"$CURRENT_BUDGET"'/checkpoints",
       "monitor": "val_log_prob",
       "mode": "min",
       "save_top_k": 3,
       "save_last": true,
       "filename": "{step}-{val_log_prob:.4f}"
     }
    }
  ]'

