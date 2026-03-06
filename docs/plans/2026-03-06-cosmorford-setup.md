# CosmOrford Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up a minimal repo for training neural network compressors (8-dim bottleneck) on weak lensing mass maps, with configurable backbones and Modal-based training.

**Architecture:** A LightningModule with a configurable vision backbone (EfficientNet, ConvNeXt, ResNet), an 8-dim bottleneck layer, and a small prediction head outputting Gaussian parameters for (Omega_m, S8). Trained via NLL. Data from HuggingFace `CosmoStat/neurips-wl-challenge-flat`. Training runs on Modal with GPU.

**Tech Stack:** PyTorch, PyTorch Lightning, HuggingFace datasets, Modal, W&B, torchvision

---

### Task 1: Project scaffolding and pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `cosmorford/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cosmorford"
version = "0.1.0"
description = "Neural network compression of weak lensing mass maps"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Francois Lanusse", email = "francois.lanusse@cnrs.fr"}
]

dependencies = [
    "torch>=2.4",
    "torchvision>=0.19",
    "lightning>=2.4",
    "datasets",
    "numpy",
    "wandb",
    "omegaconf",
    "pyyaml",
]

[project.optional-dependencies]
modal = ["modal"]
dev = ["pytest", "ipython", "ipykernel"]

[project.scripts]
trainer = "cosmorford.trainer:trainer_cli"

[tool.setuptools]
packages = ["cosmorford"]
```

**Step 2: Create `cosmorford/__init__.py`**

```python
"""CosmOrford - Neural network compression of weak lensing mass maps."""
import numpy as np

# Normalization factors for cosmological parameters (computed on training set)
# 5 parameters: Omega_m, S8, ...  (we only use first 2)
THETA_MEAN = np.array([2.9022e-01, 8.1345e-01, 7.8500e+00, 1.3262e-02, 9.2743e-04])
THETA_STD = np.array([0.1055, 0.0660, 0.3764, 0.0076, 0.0219])
NOISE_STD = 0.4 / (2 * 30 * 2.0**2) ** 0.5
```

**Step 3: Commit**

```bash
git add pyproject.toml cosmorford/__init__.py
git commit -m "feat: project scaffolding with pyproject.toml and constants"
```

---

### Task 2: Dataset module

**Files:**
- Create: `cosmorford/dataset.py`
- Create: `tests/test_dataset.py`

**Step 1: Write the failing test**

```python
# tests/test_dataset.py
import torch
import pytest


def test_reshape_field_output_shape():
    from cosmorford.dataset import reshape_field
    kappa = torch.randn(2, 1424, 176)
    result = reshape_field(kappa)
    assert result.shape == (2, 1834, 88)


def test_reshape_inverse_roundtrip():
    from cosmorford.dataset import reshape_field, inverse_reshape_field
    kappa = torch.randn(2, 1424, 176)
    result = inverse_reshape_field(reshape_field(kappa))
    # Check the regions that survive the roundtrip
    assert torch.allclose(result[:, :, :88], kappa[:, :, :88])
    assert torch.allclose(result[:, 620:1030, 88:], kappa[:, 620:1030, 88:])


def test_data_module_init():
    from cosmorford.dataset import WLDataModule
    dm = WLDataModule(batch_size=32, num_workers=0)
    assert dm.batch_size == 32
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_dataset.py -v`
Expected: FAIL (module not found)

**Step 3: Write implementation**

```python
# cosmorford/dataset.py
import torch
import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from cosmorford import THETA_MEAN, THETA_STD


def reshape_field(kappa):
    """Reshape [B, 1424, 176] -> [B, 1834, 88] to remove masked empty space."""
    return torch.concat([kappa[:, :, :88], kappa[:, 620:1030, 88:]], dim=1)


def inverse_reshape_field(kappa_reduced, fill_value=0.0):
    """Inverse of reshape_field: [B, 1834, 88] -> [B, 1424, 176]."""
    B = kappa_reduced.shape[0]
    part1 = kappa_reduced[:, :1424, :]
    part2 = kappa_reduced[:, 1424:, :]
    kappa_full = torch.full((B, 1424, 176), fill_value, dtype=kappa_reduced.dtype, device=kappa_reduced.device)
    kappa_full[:, :, :88] = part1
    kappa_full[:, 620:1030, 88:] = part2
    return kappa_full


class WLDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _collate_fn(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        kappa = reshape_field(batch["kappa"]).float()
        device = batch["theta"].device
        theta = (batch["theta"][:, :2] - torch.tensor(THETA_MEAN[:2], device=device)) / torch.tensor(THETA_STD[:2], device=device)
        theta = theta.float()
        return kappa, theta

    def setup(self, stage=None):
        dset = load_dataset("CosmoStat/neurips-wl-challenge-flat")
        dset = dset.with_format("torch")
        self.train_dataset = dset["train"]
        self.val_dataset = dset["validation"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_dataset.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add cosmorford/dataset.py tests/test_dataset.py
git commit -m "feat: dataset module with WLDataModule and field reshaping"
```

---

### Task 3: Backbone registry

**Files:**
- Create: `cosmorford/backbones.py`
- Create: `tests/test_backbones.py`

**Step 1: Write the failing test**

```python
# tests/test_backbones.py
import torch
import pytest


def test_get_backbone_efficientnet_b0():
    from cosmorford.backbones import get_backbone
    backbone, out_dim = get_backbone("efficientnet_b0")
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape[0] == 2
    assert out_dim == 1280


def test_get_backbone_convnext_tiny():
    from cosmorford.backbones import get_backbone
    backbone, out_dim = get_backbone("convnext_tiny")
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape[0] == 2
    assert out_dim == 768


def test_get_backbone_resnet18():
    from cosmorford.backbones import get_backbone
    backbone, out_dim = get_backbone("resnet18")
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape[0] == 2
    assert out_dim == 512


def test_get_backbone_unknown_raises():
    from cosmorford.backbones import get_backbone
    with pytest.raises(ValueError, match="Unknown backbone"):
        get_backbone("vgg99")
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_backbones.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# cosmorford/backbones.py
"""Backbone registry for vision feature extractors."""
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_v2_s
from torchvision.models.convnext import convnext_tiny, convnext_small
from torchvision.models.resnet import resnet18, resnet34


BACKBONES = {
    "efficientnet_b0": (efficientnet_b0, 1280),
    "efficientnet_v2_s": (efficientnet_v2_s, 1280),
    "convnext_tiny": (convnext_tiny, 768),
    "convnext_small": (convnext_small, 768),
    "resnet18": (resnet18, 512),
    "resnet34": (resnet34, 512),
}


def get_backbone(name: str):
    """Return (feature_extractor, output_dim) for a given backbone name.

    The feature extractor is the backbone without the classification head,
    outputting a 4D tensor [B, C, H, W].
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONES.keys())}")

    factory, out_dim = BACKBONES[name]
    model = factory()

    # Extract just the feature layers (no classifier head)
    if name.startswith("efficientnet"):
        features = model.features
    elif name.startswith("convnext"):
        features = model.features
    elif name.startswith("resnet"):
        import torch.nn as nn
        # ResNet: everything except avgpool and fc
        features = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )

    return features, out_dim
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_backbones.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add cosmorford/backbones.py tests/test_backbones.py
git commit -m "feat: backbone registry with EfficientNet, ConvNeXt, ResNet"
```

---

### Task 4: Compressor model (LightningModule)

**Files:**
- Create: `cosmorford/compressor.py`
- Create: `tests/test_compressor.py`

**Step 1: Write the failing test**

```python
# tests/test_compressor.py
import torch
import pytest


def test_compressor_forward_shapes():
    from cosmorford.compressor import CompressorModel
    model = CompressorModel(backbone="resnet18")
    x = torch.randn(4, 1834, 88)
    mean, scale = model(x)
    assert mean.shape == (4, 2)
    assert scale.shape == (4, 2)
    assert (scale > 0).all()


def test_compressor_bottleneck_output():
    from cosmorford.compressor import CompressorModel
    model = CompressorModel(backbone="resnet18")
    x = torch.randn(4, 1834, 88)
    z = model.compress(x)
    assert z.shape == (4, 8)


def test_compressor_training_step():
    from cosmorford.compressor import CompressorModel
    model = CompressorModel(backbone="resnet18")
    x = torch.randn(4, 1834, 88)
    y = torch.randn(4, 2)
    loss = model.training_step((x, y), 0)
    assert loss.shape == ()
    assert loss.requires_grad
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_compressor.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# cosmorford/compressor.py
"""Compressor model: vision backbone -> 8-dim bottleneck -> Gaussian prediction head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from cosmorford import THETA_MEAN, THETA_STD, NOISE_STD
from cosmorford.backbones import get_backbone


class CompressorModel(L.LightningModule):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        bottleneck_dim: int = 8,
        warmup_steps: int = 500,
        max_lr: float = 0.008,
        decay_rate: float = 0.85,
        decay_every_epochs: int = 1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        features, feat_dim = get_backbone(backbone)
        self.backbone = features

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
        )

        self.bottleneck = nn.Linear(feat_dim, bottleneck_dim)

        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2 * 2),  # mean(2) + log_scale(2)
        )

    def _features(self, x):
        """Run backbone on input map, handling channel expansion."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.pool(self.backbone(x.float()))

    def compress(self, x):
        """Return the 8-dim bottleneck representation."""
        return self.bottleneck(self._features(x))

    def forward(self, x):
        z = self.compress(x)
        out = self.head(z)
        mean = out[..., :2]
        scale = F.softplus(out[..., 2:]) + 0.001
        return mean, scale

    def _augment(self, x):
        """Apply augmentations: noise, random flips, cyclic shifts."""
        noise = torch.randn_like(x) * NOISE_STD
        x = x + noise

        batch_size = x.size(0)
        # Random flips
        flip_lr = torch.rand(batch_size, device=x.device) < 0.5
        x[flip_lr] = torch.flip(x[flip_lr], dims=[1])
        flip_ud = torch.rand(batch_size, device=x.device) < 0.5
        x[flip_ud] = torch.flip(x[flip_ud], dims=[2])

        # Random cyclic shifts
        shift_x = torch.randint(0, x.size(1), (batch_size,), device=x.device)
        x = torch.stack([torch.roll(x[i], shifts=(shift_x[i].item(),), dims=(0,)) for i in range(batch_size)])
        shift_y = torch.randint(0, x.size(2), (batch_size,), device=x.device)
        x = torch.stack([torch.roll(x[i], shifts=(shift_y[i].item(),), dims=(1,)) for i in range(batch_size)])
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self._augment(x)
        mean, std = self(x)
        loss = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        noise = torch.randn_like(x) * NOISE_STD
        x = x + noise

        mean, std = self(x)

        # NLL loss
        loss = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()

        # Also log MSE in original parameter space for interpretability
        mean_orig = mean * torch.tensor(THETA_STD[:2], device=mean.device) + torch.tensor(THETA_MEAN[:2], device=mean.device)
        y_orig = y * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)
        mse = F.mse_loss(mean_orig, y_orig)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", mse, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.max_lr, weight_decay=1e-5)
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = self.hparams.warmup_steps
        steps_per_epoch = total_steps // self.trainer.max_epochs
        step_size = self.hparams.decay_every_epochs * steps_per_epoch

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps)
        decay = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=self.hparams.decay_rate)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/test_compressor.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add cosmorford/compressor.py tests/test_compressor.py
git commit -m "feat: compressor model with bottleneck and Gaussian NLL training"
```

---

### Task 5: Trainer and LightningCLI setup

**Files:**
- Create: `cosmorford/trainer.py`

**Step 1: Write implementation**

```python
# cosmorford/trainer.py
import torch
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")


class CustomSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        for logger in trainer.loggers:
            if issubclass(type(logger), WandbLogger):
                logger.experiment.config.update(self.config.as_dict())
        return super().save_config(trainer, pl_module, stage)


def trainer_cli(args: ArgsType = None, run: bool = True):
    return LightningCLI(
        args=args,
        run=run,
        save_config_kwargs={"overwrite": True},
        save_config_callback=CustomSaveConfigCallback,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    trainer_cli(run=True)
```

**Step 2: Commit**

```bash
git add cosmorford/trainer.py
git commit -m "feat: LightningCLI trainer with W&B config callback"
```

---

### Task 6: Experiment configs

**Files:**
- Create: `configs/default.yaml`
- Create: `configs/experiments/efficientnet_b0.yaml`
- Create: `configs/experiments/convnext_tiny.yaml`
- Create: `configs/experiments/resnet18.yaml`

**Step 1: Create default config**

```yaml
# configs/default.yaml
model:
  class_path: cosmorford.compressor.CompressorModel
  init_args:
    backbone: "efficientnet_b0"
    bottleneck_dim: 8
    warmup_steps: 500
    max_lr: 0.008
    decay_rate: 0.85
    decay_every_epochs: 1
    dropout_rate: 0.2

data:
  class_path: cosmorford.dataset.WLDataModule
  init_args:
    batch_size: 128
    num_workers: 8

trainer:
  max_epochs: 30
  accelerator: gpu
  precision: "16-mixed"
  log_every_n_steps: 1
  check_val_every_n_epoch: 1
  callbacks:
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: "step"
    - class_path: ModelCheckpoint
      init_args:
        monitor: "val_loss"
        mode: "min"
        save_top_k: 3
        save_last: true
  logger:
    class_path: WandbLogger
    init_args:
      project: "cosmorford"
      log_model: true
```

**Step 2: Create experiment configs**

```yaml
# configs/experiments/efficientnet_b0.yaml
model:
  class_path: cosmorford.compressor.CompressorModel
  init_args:
    backbone: "efficientnet_b0"
    bottleneck_dim: 8
    warmup_steps: 500
    max_lr: 0.008
    decay_rate: 0.85
    dropout_rate: 0.2

data:
  class_path: cosmorford.dataset.WLDataModule
  init_args:
    batch_size: 128
    num_workers: 8

trainer:
  max_epochs: 30
  accelerator: gpu
  precision: "16-mixed"
  log_every_n_steps: 1
  check_val_every_n_epoch: 1
  callbacks:
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: "step"
    - class_path: ModelCheckpoint
      init_args:
        monitor: "val_loss"
        mode: "min"
        save_top_k: 3
        save_last: true
  logger:
    class_path: WandbLogger
    init_args:
      name: "efficientnet_b0"
      project: "cosmorford"
      log_model: true
```

```yaml
# configs/experiments/convnext_tiny.yaml
model:
  class_path: cosmorford.compressor.CompressorModel
  init_args:
    backbone: "convnext_tiny"
    bottleneck_dim: 8
    warmup_steps: 500
    max_lr: 0.004
    decay_rate: 0.85
    dropout_rate: 0.2

data:
  class_path: cosmorford.dataset.WLDataModule
  init_args:
    batch_size: 128
    num_workers: 8

trainer:
  max_epochs: 30
  accelerator: gpu
  precision: "16-mixed"
  log_every_n_steps: 1
  check_val_every_n_epoch: 1
  callbacks:
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: "step"
    - class_path: ModelCheckpoint
      init_args:
        monitor: "val_loss"
        mode: "min"
        save_top_k: 3
        save_last: true
  logger:
    class_path: WandbLogger
    init_args:
      name: "convnext_tiny"
      project: "cosmorford"
      log_model: true
```

```yaml
# configs/experiments/resnet18.yaml
model:
  class_path: cosmorford.compressor.CompressorModel
  init_args:
    backbone: "resnet18"
    bottleneck_dim: 8
    warmup_steps: 500
    max_lr: 0.01
    decay_rate: 0.85
    dropout_rate: 0.2

data:
  class_path: cosmorford.dataset.WLDataModule
  init_args:
    batch_size: 128
    num_workers: 8

trainer:
  max_epochs: 30
  accelerator: gpu
  precision: "16-mixed"
  log_every_n_steps: 1
  check_val_every_n_epoch: 1
  callbacks:
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: "step"
    - class_path: ModelCheckpoint
      init_args:
        monitor: "val_loss"
        mode: "min"
        save_top_k: 3
        save_last: true
  logger:
    class_path: WandbLogger
    init_args:
      name: "resnet18"
      project: "cosmorford"
      log_model: true
```

**Step 3: Commit**

```bash
git add configs/
git commit -m "feat: YAML configs for default and 3 experiment architectures"
```

---

### Task 7: Modal training entrypoint

**Files:**
- Create: `train_modal.py`

**Step 1: Write implementation**

```python
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
    )
    .copy_local_dir("cosmorford", "/root/cosmorford")
    .copy_local_dir("configs", "/root/configs")
    .copy_local_file("pyproject.toml", "/root/pyproject.toml")
    .run_commands("cd /root && pip install -e .")
)

app = modal.App("cosmorford", image=image)

VOLUME_PATH = Path("/experiments")
CHECKPOINTS_PATH = VOLUME_PATH / "checkpoints"


@app.function(
    volumes={VOLUME_PATH: volume},
    gpu="a10g",
    timeout=86400,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
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
        f"--trainer.callbacks.1.init_args.dirpath={checkpoint_dir}",
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
```

**Step 2: Commit**

```bash
git add train_modal.py
git commit -m "feat: Modal training entrypoint with checkpointing and W&B"
```

---

### Task 8: Local training entrypoint

**Files:**
- Create: `train.py`

**Step 1: Write implementation**

```python
# train.py
"""Local training entrypoint."""
from cosmorford.trainer import trainer_cli

if __name__ == "__main__":
    trainer_cli()
```

**Step 2: Commit**

```bash
git add train.py
git commit -m "feat: local training entrypoint"
```

---

### Task 9: .gitignore

**Files:**
- Create: `.gitignore`

**Step 1: Write .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
*.egg
venv/
.venv/
wandb/
lightning_logs/
*.ckpt
.DS_Store
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

---

### Task 10: Run full test suite and verify

**Step 1: Install the package locally**

Run: `cd /home/francois/repo/CosmOrford && pip install -e ".[dev]"`

**Step 2: Run all tests**

Run: `cd /home/francois/repo/CosmOrford && python -m pytest tests/ -v`
Expected: All 7 tests PASS

**Step 3: Verify trainer CLI works (dry run)**

Run: `cd /home/francois/repo/CosmOrford && trainer --help`
Expected: Shows LightningCLI help output

**Step 4: Final commit if any fixes needed**
