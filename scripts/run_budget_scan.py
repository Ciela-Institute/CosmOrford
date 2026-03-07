"""Launch simulation budget scan on Modal.

Spawns parallel training runs with different training set sizes
to measure how constraining power scales with simulation budget.

Usage:
    .venv/bin/modal run scripts/run_budget_scan.py
"""
from train_modal import app, train

BUDGETS = [100, 200, 500, 1000, 2000, 5000, 10000, 20200]
BASE_CONFIG = "configs/experiments/resnet18.yaml"


@app.local_entrypoint()
def main():
    handles = []
    for n in BUDGETS:
        name = f"budget-{n}"
        overrides = [
            f"--data.init_args.max_train_samples={n}",
            f"--trainer.logger.init_args.name={name}",
            "--trainer.logger.init_args.tags=[budget-scan]",
        ]
        print(f"Spawning {name} (max_train_samples={n})")
        handles.append(train.spawn(BASE_CONFIG, name, overrides))

    print(f"Waiting for {len(handles)} runs to complete...")
    for h in handles:
        h.get()
    print("All budget scan runs completed.")
