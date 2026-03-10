# train_modal.py
"""Modal entrypoint for remote GPU training."""
from pathlib import Path
from typing import Optional

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

app = modal.App("cosmoford", image=image)

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
def train(config_path: str, experiment_name: Optional[str] = None, cli_overrides: Optional[list] = None):
    _train_impl(config_path, experiment_name, cli_overrides)


@app.function(
    volumes={VOLUME_PATH: volume},
    gpu="a100",
    timeout=86400,
    retries=modal.Retries(initial_delay=0.0, max_retries=0),
    single_use_containers=True,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_a100(config_path: str, experiment_name: Optional[str] = None, cli_overrides: Optional[list] = None):
    _train_impl(config_path, experiment_name, cli_overrides)


def _train_impl(config_path: str, experiment_name: Optional[str] = None, cli_overrides: Optional[list] = None):
    import subprocess
    import yaml

    checkpoint_dir = CHECKPOINTS_PATH / (experiment_name or "default")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = checkpoint_dir / "last.ckpt"

    # Write a runtime overlay config that sets checkpoint dir on the volume
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

    cmd = [
        "trainer", "fit",
        f"--config=/root/{config_path}",
        f"--config={overlay_path}",
    ]

    if cli_overrides:
        cmd.extend(cli_overrides)

    if last_ckpt.exists():
        print(f"Resuming from checkpoint: {last_ckpt}")
        cmd.append(f"--ckpt_path={last_ckpt}")
    else:
        print("Starting training from scratch")

    subprocess.run(cmd, check=True, cwd="/root")
    volume.commit()


@app.local_entrypoint()
def main(
    config: str = "configs/default.yaml",
    name: Optional[str] = None,
    gpu: str = "a10g",
):
    print(f"Starting training with config: {config} on GPU: {gpu}")
    if gpu == "a100":
        train_a100.spawn(config, name).get()
    else:
        train.spawn(config, name).get()
