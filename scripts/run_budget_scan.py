"""Launch simulation budget scan.

Spawns training runs with different training set sizes to measure how
constraining power scales with simulation budget.

Usage (Modal):
    .venv/bin/modal run scripts/run_budget_scan.py

Usage (local):
    python scripts/run_budget_scan.py \\
        --checkpoints_path /path/to/checkpoints \\
        --base_config configs/experiments/resnet18.yaml \\
        [--budgets 100,500,20200]
"""
from pathlib import Path
from typing import List


BUDGETS = [100, 200, 500, 1000, 2000, 5000, 10000, 20200]
BASE_CONFIG = "configs/experiments/resnet18.yaml"


def _run_training(base_config: str, checkpoint_dir: Path, cli_overrides: List[str], cwd: str = None):
    """Invoke `trainer fit` for a single budget. Shared by Modal and local modes."""
    import subprocess
    import shutil
    import yaml

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overlay = {
        "trainer": {
            "default_root_dir": str(checkpoint_dir),
            "callbacks": [
                {
                    "class_path": "LearningRateMonitor",
                    "init_args": {"logging_interval": "step"},
                },
                {
                    "class_path": "ModelCheckpoint",
                    "init_args": {
                        "dirpath": str(checkpoint_dir),
                        "monitor": "val_mse",
                        "mode": "min",
                        "save_top_k": 3,
                        "save_last": True,
                    },
                },
            ],
        }
    }
    overlay_path = Path("/tmp/runtime_config.yaml")
    overlay_path.write_text(yaml.dump(overlay))

    config_arg = base_config if cwd is None else f"{cwd}/{base_config}"
    cmd = [
        "trainer", "fit",
        f"--config={config_arg}",
        f"--config={overlay_path}",
    ]
    cmd.extend(cli_overrides)

    # Always start fresh for budget scan (clear old checkpoints)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print("Starting training from scratch")

    subprocess.run(cmd, check=True, cwd=cwd)


# ── Modal entry point (only loaded when invoked via `modal run`) ──────────────
if __name__ != "__main__":
    import modal

    volume = modal.Volume.from_name("cosmoford-training", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "torch>=2.4",
            "torchvision>=0.19",
            "lightning>=2.4",
            "datasets",
            "numpy",
            "wandb",
            "omegaconf",
            "pyyaml",
            "jsonargparse[signatures,omegaconf]>=4.27.7",
            "peft",
            "nflows",
            "matplotlib",
            "scikit-learn",
        )
        .add_local_dir("cosmoford", "/root/cosmoford", copy=True)
        .add_local_dir("configs", "/root/configs", copy=True)
        .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
        .run_commands("cd /root && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -e . --no-deps")
    )

    app = modal.App("cosmoford-budget-scan", image=image)

    VOLUME_PATH = Path("/experiments")
    CHECKPOINTS_PATH = VOLUME_PATH / "checkpoints"

    @app.function(
        volumes={VOLUME_PATH: volume},
        gpu="a10g",
        timeout=86400,
        retries=modal.Retries(initial_delay=0.0, max_retries=0),
        single_use_containers=True,
        secrets=[modal.Secret.from_name("wandb-secret")],
    )
    def train_budget(config_path: str, experiment_name: str, cli_overrides: list):
        checkpoint_dir = CHECKPOINTS_PATH / experiment_name
        _run_training(config_path, checkpoint_dir, cli_overrides, cwd="/root")
        volume.commit()

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
            handles.append(train_budget.spawn(BASE_CONFIG, name, overrides))

        print(f"Waiting for {len(handles)} runs to complete...")
        for h in handles:
            h.get()
        print("All budget scan runs completed.")


# ── Local entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulation budget scan — local mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoints_path", required=True,
                        help="Root directory where checkpoints will be written (budget-N/ subdirs created inside)")
    parser.add_argument("--base_config", default=BASE_CONFIG,
                        help="Path to base Lightning config YAML")
    parser.add_argument("--budgets",
                        help="Comma-separated list of budgets to run, e.g. 100,500,20200 "
                             "(overrides the default list)")
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else BUDGETS
    checkpoints_root = Path(args.checkpoints_path)

    for n in budgets:
        name = f"budget-{n}"
        checkpoint_dir = checkpoints_root / name
        overrides = [
            f"--data.init_args.max_train_samples={n}",
            f"--trainer.logger.init_args.name={name}",
            "--trainer.logger.init_args.tags=[budget-scan]",
            "--model.init_args.pretrained=false",
            "--trainer.check_val_every_n_epoch=null",
        ]
        print(f"\n=== Running {name} (max_train_samples={n}) ===")
        _run_training(args.base_config, checkpoint_dir, overrides)

    print("All budget scan runs completed.")
