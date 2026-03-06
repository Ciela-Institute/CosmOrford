# train_modal.py
"""Modal entrypoint for remote GPU training."""
from pathlib import Path
from typing import Optional

import modal

volume = modal.Volume.from_name("cosmorford-training", create_if_missing=True)

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
    )
    .add_local_dir("cosmorford", "/root/cosmorford", copy=True)
    .add_local_dir("configs", "/root/configs", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .run_commands("cd /root && pip install -e .")
)

app = modal.App("cosmorford", image=image)

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
def train(config_path: str, experiment_name: Optional[str] = None):
    import subprocess

    checkpoint_dir = CHECKPOINTS_PATH / (experiment_name or "default")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = checkpoint_dir / "last.ckpt"

    cmd = [
        "trainer", "fit",
        f"--config=/root/{config_path}",
        f"--trainer.default_root_dir={checkpoint_dir}",
    ]

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
):
    print(f"Starting training with config: {config}")
    train.spawn(config, name).get()
