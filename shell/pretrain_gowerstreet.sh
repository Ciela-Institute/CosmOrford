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

# We use dot notation to reach deep into the YAML structure
uv run trainer fit \
    -c configs/experiments/pretrain_gowerstreet_nopatch_logp.yaml \
    --data.init_args.dataset_mode=gowerstreet \
    --trainer.logger.init_args.name="effnet_v2_s_pretrain_gowerstreet_$CURRENT_BUDGET" \
    --trainer.logger.init_args.save_dir="$SAVE_DIR/pretrain_gowerstreet" \
    --trainer.callbacks='[
    {"class_path": "LearningRateMonitor", "init_args": {"logging_interval": "step"}},
    {"class_path": "EMAWeightAveraging"},
    {"class_path": "ModelCheckpoint",
     "init_args": {
       "dirpath": "'"$SAVE_DIR"'/pretrain_gowerstreet/checkpoints",
       "monitor": "val_log_prob",
       "mode": "min",
       "save_top_k": 3,
       "save_last": true,
       "filename": "{step}-{val_log_prob:.4f}"
     }
    }
  ]'


    