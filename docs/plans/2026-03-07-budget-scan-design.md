# Simulation Budget Scan Design

## Goal

Measure how the constraining power of learned summary statistics scales with the number of training simulations. Produce a plot of best `val_score` vs `n_train` (log-scale x-axis) for the ResNet-18 compressor.

## Budgets

Log-spaced: 100, 200, 500, 1000, 2000, 5000, 10000, 20200 (full dataset).

Single run per budget for now. Validation set (5656 sims) stays fixed across all runs.

## Changes

### 1. `cosmoford/dataset.py`

Add `max_train_samples: int = 0` to `ChallengeDataModule.__init__`. In `setup()`, after building `self.train_dataset`, if `max_train_samples > 0`, truncate with `self.train_dataset = self.train_dataset.select(range(max_train_samples))`. The HF dataset order is deterministic, giving reproducible subsets.

### 2. `scripts/run_budget_scan.py`

Launch script that spawns 8 Modal runs in parallel. For each budget N:
- Uses `resnet18.yaml` as base config
- Overrides `data.init_args.max_train_samples=N`
- Sets wandb run name to `budget-N` and adds tag `budget-scan`
- Logs `n_train=N` to wandb config

### 3. `scripts/plot_budget_scan.py`

Pulls results from wandb:
- Queries runs with tag `budget-scan` from the configured project
- Extracts best `val_score` and `n_train` from each run
- Plots val_score vs n_train with log x-axis
- Saves plot to file
