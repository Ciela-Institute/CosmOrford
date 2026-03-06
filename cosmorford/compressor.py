"""Compressor model: vision backbone -> 8-dim bottleneck -> Gaussian prediction head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from cosmorford import THETA_MEAN, THETA_STD, NOISE_STD, SURVEY_MASK
from cosmorford.backbones import get_backbone, adapt_first_conv
from cosmorford.dataset import reshape_field


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
        lr_schedule: str = "cosine",
        use_sam: bool = False,
        sam_rho: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()
        if use_sam:
            self.automatic_optimization = False

        features, feat_dim = get_backbone(backbone, pretrained=True)
        self.backbone = adapt_first_conv(features, backbone)

        # Reshaped survey mask matching the field layout after reshape_field
        self.mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

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
        """Run backbone on input map."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
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
        """Apply augmentations: noise, survey mask, random flips, cyclic shifts."""
        noise = torch.randn_like(x) * NOISE_STD
        x = x + noise
        x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

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

        if self.hparams.use_sam:
            opt = self.optimizers()
            sch = self.lr_schedulers()

            # First forward-backward
            mean, std = self(x)
            loss = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
            self.manual_backward(loss)
            opt.first_step(zero_grad=True)

            # Second forward-backward
            mean, std = self(x)
            loss2 = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
            self.manual_backward(loss2)
            opt.second_step(zero_grad=True)
            sch.step()

            self.log("train_loss", loss)
            return loss
        else:
            mean, std = self(x)
            loss = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
            self.log("train_loss", loss)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        noise = torch.randn_like(x) * NOISE_STD
        x = x + noise
        x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

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
        if self.hparams.use_sam:
            from cosmorford.sam import SAM
            optimizer = SAM(
                self.parameters(), torch.optim.AdamW,
                lr=self.hparams.max_lr, weight_decay=1e-5, rho=self.hparams.sam_rho,
            )
            # Use base_optimizer for schedulers since SequentialLR needs a real optimizer
            sched_optimizer = optimizer.base_optimizer
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.max_lr, weight_decay=1e-5)
            sched_optimizer = optimizer

        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = self.hparams.warmup_steps

        warmup = torch.optim.lr_scheduler.LinearLR(
            sched_optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
        )

        if self.hparams.lr_schedule == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                sched_optimizer, T_max=total_steps - warmup_steps
            )
        else:  # "step"
            steps_per_epoch = total_steps // self.trainer.max_epochs
            step_size = self.hparams.decay_every_epochs * steps_per_epoch
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                sched_optimizer, step_size=step_size, gamma=self.hparams.decay_rate
            )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            sched_optimizer, schedulers=[warmup, main_scheduler], milestones=[warmup_steps]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
