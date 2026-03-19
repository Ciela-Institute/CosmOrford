"""Summary-statistics compressor and normalizing flow for cosmological posterior inference.

Two-stage training pipeline:

  Stage 1 - HOSCompressor:
    Computes HOS (peak counts + L1-norms) and wavelet scattering transform features
    from convergence maps using cosmoford.summaries, then compresses them through an
    MLP to an 8-dimensional bottleneck. Trained with Gaussian NLL. The compress()
    method returns the bottleneck vector used as context for the flow in stage 2.

  Stage 2 - HOSPosterior:
    Loads a trained HOSCompressor checkpoint and freezes all its parameters.
    Attaches a conditional Masked Autoregressive Flow (MAF) that models
    p(theta | z) where z is the 8-dim summary vector. After training, the
    sample_posterior() method draws samples from the full posterior
    p(Omega_m, sigma_8 | convergence map).

Usage::

    # Stage 1 - train compressor
    sbatch scripts/submit_job.sh configs/experiments/hos_compressor.yaml

    # Stage 2 - train flow (update compressor_checkpoint_path in the config first)
    sbatch scripts/submit_job.sh configs/experiments/hos_posterior.yaml
"""

import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    MaskedAffineAutoregressiveTransform,
    RandomPermutation,
)

from cosmoford import THETA_MEAN, THETA_STD, SURVEY_MASK, NOISE_STD
from cosmoford.summaries import (
    power_spectrum_batch,
    compute_higher_order_statistics_batch,
    compute_scattering_batch,
    scattering_n_coefficients,
)


def _inverse_reshape_field(kappa_reduced: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Reconstruct full (1424, 176) maps from reduced (1834, 88) representation."""
    bsz, _, _ = kappa_reduced.shape
    part1 = kappa_reduced[:, :1424, :]
    part2 = kappa_reduced[:, 1424:, :]
    kappa_full = torch.full(
        (bsz, 1424, 176),
        fill_value,
        dtype=kappa_reduced.dtype,
        device=kappa_reduced.device,
    )
    kappa_full[:, :, :88] = part1
    kappa_full[:, 620:1030, 88:] = part2
    return kappa_full


def build_flow(
    param_dim: int = 2,
    context_dim: int = 8,
    n_transforms: int = 8,
    hidden_dim: int = 128,
) -> Flow:
    """Build a conditional Masked Autoregressive Flow: p(params | summaries).

    Args:
        param_dim: Dimensionality of the parameter space (default 2: Omega_m, sigma_8).
        context_dim: Dimensionality of the summary/context vector (default 8).
        n_transforms: Number of MAF coupling transforms.
        hidden_dim: Hidden layer size in the MAF networks.

    Returns:
        A nflows.flows.Flow that can evaluate log_prob(params, context=summaries)
        and sample(n, context=summaries).
    """
    transforms = []
    for _ in range(n_transforms):
        transforms.append(RandomPermutation(features=param_dim))
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=param_dim,
                hidden_features=hidden_dim,
                context_features=context_dim,
            )
        )
    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal([param_dim]),
    )


def _compute_input_dim(
    use_hos: bool,
    hos_n_scales: int,
    hos_n_bins: int,
    hos_l1_nbins: int,
    use_scattering: bool,
    scattering_J: int,
    scattering_L: int,
    scattering_feature_pooling: str,
    use_ps: bool,
) -> int:
    """Compute the total raw feature dimensionality from summary settings."""
    dim = 0
    if use_hos:
        # Peak count histograms: n_scales * n_bins
        # L1-norm histograms:    n_scales * l1_nbins
        dim += hos_n_scales * (hos_n_bins + hos_l1_nbins)
    if use_scattering:
        dim += scattering_n_coefficients(
            scattering_J,
            scattering_L,
            feature_pooling=scattering_feature_pooling,
        )
    if use_ps:
        dim += 10
    return dim


class HOSCompressor(L.LightningModule):
    """Stage-1 summary compressor: summary statistics -> summary_dim bottleneck.

    The model computes configurable summary statistics from the input convergence
    map (HOS peak counts + L1-norm histograms, wavelet scattering transform
    coefficients, and optionally the angular power spectrum), concatenates them,
    and projects the result through a multi-layer perceptron to a low-dimensional
    bottleneck. A Gaussian prediction head is attached for training.

    After training, the compress() method returns the bottleneck representation,
    which serves as context for the normalizing flow in HOSPosterior (stage 2).
    """

    def __init__(
        self,
        # --- Summary statistics ---
        use_hos: bool = True,
        hos_n_scales: int = 4,
        hos_n_bins: int = 51,
        hos_l1_nbins: int = 80,
        hos_min_snr: float = -5.0,
        hos_max_snr: float = 10.0,
        hos_l1_min_snr: float = -10.0,
        hos_l1_max_snr: float = 10.0,
        use_scattering: bool = True,
        scattering_J: int = 4,
        scattering_L: int = 8,
        scattering_normalization: str = "log1p_zscore",
        scattering_feature_pooling: str = "mean",
        scattering_mask_pooling: str = "soft",
        scattering_geometry: str = "reduced",
        use_ps: bool = False,
        augment_flip: bool = True,
        augment_shift: bool = True,
        # --- Compressor MLP ---
        summary_dim: int = 8,
        hidden_dim: int = 512,
        n_hidden: int = 3,
        dropout_rate: float = 0.1,
        # --- Training ---
        warmup_steps: int = 500,
        max_lr: float = 1e-3,
        decay_rate: float = 0.85,
        decay_every_epochs: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        if scattering_normalization not in ["log1p_zscore", "zscore", "none"]:
            raise ValueError(
                "scattering_normalization must be one of ['log1p_zscore', 'zscore', 'none'], "
                f"got '{scattering_normalization}'"
            )
        if scattering_mask_pooling not in ["soft", "hard"]:
            raise ValueError(
                "scattering_mask_pooling must be one of ['soft', 'hard'], "
                f"got '{scattering_mask_pooling}'"
            )
        if scattering_feature_pooling not in ["mean", "mean_std"]:
            raise ValueError(
                "scattering_feature_pooling must be one of ['mean', 'mean_std'], "
                f"got '{scattering_feature_pooling}'"
            )
        if scattering_geometry not in ["reduced", "full"]:
            raise ValueError(
                "scattering_geometry must be one of ['reduced', 'full'], "
                f"got '{scattering_geometry}'"
            )

        self.mask_reduced = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])
        self.mask_full = SURVEY_MASK
        self.mask = self.mask_reduced

        input_dim = _compute_input_dim(
            use_hos, hos_n_scales, hos_n_bins, hos_l1_nbins,
            use_scattering, scattering_J, scattering_L, scattering_feature_pooling, use_ps,
        )
        if input_dim == 0:
            raise ValueError(
                "At least one of use_hos, use_scattering, use_ps must be True."
            )

        # MLP: input_dim -> hidden_dim x n_hidden -> summary_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_hidden - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ]
        layers.append(nn.Linear(hidden_dim, summary_dim))
        self.compressor = nn.Sequential(*layers)

        # Gaussian prediction head (for stage-1 training only)
        self.head = nn.Sequential(
            nn.Linear(summary_dim, summary_dim * 8),
            nn.GELU(),
            nn.Linear(summary_dim * 8, 2 * 2),  # mean + log-scale for (Omega_m, sigma_8)
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _compute_summaries(self, x: torch.Tensor) -> torch.Tensor:
        """Compute concatenated summary statistics (no gradient tracking)."""
        parts = []
        mask_t = torch.tensor(self.mask_reduced, device=x.device, dtype=x.dtype).unsqueeze(0)
        with torch.no_grad():
            if self.hparams.use_ps:
                _, ps = power_spectrum_batch(x)
                parts.append(ps)
            if self.hparams.use_hos:
                hos = compute_higher_order_statistics_batch(
                    x,
                    noise_std=NOISE_STD,
                    mask=mask_t,
                    n_scales=self.hparams.hos_n_scales,
                    pixel_arcmin=2.0,
                    n_bins=self.hparams.hos_n_bins,
                    l1_nbins=self.hparams.hos_l1_nbins,
                    min_snr=self.hparams.hos_min_snr,
                    max_snr=self.hparams.hos_max_snr,
                    l1_min_snr=self.hparams.hos_l1_min_snr,
                    l1_max_snr=self.hparams.hos_l1_max_snr,
                    normalize=True,
                )
                parts.append(hos)
            if self.hparams.use_scattering:
                if self.hparams.scattering_geometry == "full":
                    scat_x = _inverse_reshape_field(x)
                    scat_mask = torch.tensor(self.mask_full, device=x.device, dtype=x.dtype)
                else:
                    scat_x = x
                    scat_mask = mask_t.squeeze(0)
                scat = compute_scattering_batch(
                    scat_x,
                    J=self.hparams.scattering_J,
                    L=self.hparams.scattering_L,
                    normalize=True,
                    normalization=self.hparams.scattering_normalization,
                    mask=scat_mask,
                    mask_pooling=self.hparams.scattering_mask_pooling,
                    feature_pooling=self.hparams.scattering_feature_pooling,
                )
                parts.append(scat)
        return torch.cat(parts, dim=1)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Return the summary_dim-dimensional bottleneck vector for input maps x."""
        return self.compressor(self._compute_summaries(x))

    def forward(self, x: torch.Tensor):
        """Forward pass: returns (mean, scale) in normalised parameter space."""
        z = self.compress(x)
        out = self.head(z)
        mean = out[..., :2]
        scale = F.softplus(out[..., 2:]) + 0.001
        return mean, scale

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Adding noise BEFORE augmentation (consistent with RegressionModel)
        x = x + torch.randn_like(x) * NOISE_STD

        # Augmentations BEFORE masking so the survey footprint stays fixed
        batch_size = x.size(0)
        if self.hparams.augment_flip:
            flip_lr = torch.rand(batch_size, device=x.device) < 0.5
            x[flip_lr] = torch.flip(x[flip_lr], dims=[1])
            flip_ud = torch.rand(batch_size, device=x.device) < 0.5
            x[flip_ud] = torch.flip(x[flip_ud], dims=[2])
        if self.hparams.augment_shift:
            shift_x = torch.randint(low=0, high=x.size(1), size=(batch_size,), device=x.device)
            x = torch.stack([torch.roll(x[i], shifts=(shift_x[i].item(),), dims=(0,)) for i in range(batch_size)])
            shift_y = torch.randint(low=0, high=x.size(2), size=(batch_size,), device=x.device)
            x = torch.stack([torch.roll(x[i], shifts=(shift_y[i].item(),), dims=(1,)) for i in range(batch_size)])

        # Apply mask after augmentation
        x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

        mean, scale = self(x)
        # y is already normalised; model outputs are in the same normalised space
        loss = -torch.distributions.Normal(loc=mean, scale=scale).log_prob(y).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Single noise realisation with mask (consistent with RegressionModel)
        x = x + torch.randn_like(x) * NOISE_STD
        x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

        mean, scale = self(x)

        # Rescale to physical units
        mean = mean * torch.tensor(THETA_STD[:2], device=mean.device) + torch.tensor(THETA_MEAN[:2], device=mean.device)
        scale = scale * torch.tensor(THETA_STD[:2], device=scale.device)
        y = y[:, :2] * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)

        sq_error = (y - mean) ** 2
        score = -torch.sum(sq_error / scale**2 + torch.log(scale**2) + 1000.0 * sq_error, dim=1)
        score = torch.mean(score)
        mse = F.mse_loss(mean, y)

        self.log("val_score", score, prog_bar=True)
        self.log("val_mse", mse, prog_bar=True)
        return score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.max_lr, weight_decay=1e-5
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = total_steps // self.trainer.max_epochs
        step_size = self.hparams.decay_every_epochs * steps_per_epoch

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-10, end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        decay = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=self.hparams.decay_rate,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay],
            milestones=[self.hparams.warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


class HOSPosterior(L.LightningModule):
    """Stage-2 posterior: frozen HOSCompressor + conditional normalizing flow.

    Loads a trained HOSCompressor from a checkpoint and freezes all its parameters.
    A conditional Masked Autoregressive Flow (MAF) is then trained to model the
    posterior distribution p(theta | z), where z is the 8-dimensional summary
    vector produced by the frozen compressor.

    After training, sample_posterior(x, n_samples) returns posterior samples in
    physical parameter units (Omega_m, sigma_8), enabling the construction of
    cosmological parameter contours.
    """

    def __init__(
        self,
        compressor_checkpoint_path: str,
        # --- Normalizing flow ---
        flow_transforms: int = 8,
        flow_hidden_dim: int = 128,
        # --- Training ---
        warmup_steps: int = 500,
        max_lr: float = 5e-4,
        decay_rate: float = 0.90,
        decay_every_epochs: int = 1,
    ):
        super().__init__()
        # compressor_checkpoint_path stored outside hparams since it is a
        # runtime path, not an architecture hyperparameter.
        self.save_hyperparameters(ignore=["compressor_checkpoint_path"])
        self._compressor_checkpoint_path = compressor_checkpoint_path

        # Load and freeze the compressor
        compressor = HOSCompressor.load_from_checkpoint(compressor_checkpoint_path)
        for p in compressor.parameters():
            p.requires_grad_(False)
        compressor.eval()
        self.compressor = compressor

        # Reuse the mask from the compressor
        self.mask = compressor.mask

        # Build the conditional normalizing flow
        self.flow = build_flow(
            param_dim=2,
            context_dim=compressor.hparams.summary_dim,
            n_transforms=flow_transforms,
            hidden_dim=flow_hidden_dim,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Same noise + mask as compressor training (no augmentation: the compressor
        # already learned rotation/reflection invariances during stage 1).
        x = x + torch.randn_like(x) * NOISE_STD
        x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

        with torch.no_grad():
            z = self.compressor.compress(x)

        # y is already normalised; flow is trained in normalised space
        loss = -self.flow.log_prob(y, context=z).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = x + torch.randn_like(x) * NOISE_STD
        x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

        with torch.no_grad():
            z = self.compressor.compress(x)

        val_nll = -self.flow.log_prob(y, context=z).mean()

        # Approximate posterior mean via samples for a val_mse proxy
        with torch.no_grad():
            samples = self.flow.sample(128, context=z)  # (128, batch, 2)
            mean_norm = samples.mean(dim=0)
        mean = mean_norm * torch.tensor(THETA_STD[:2], device=x.device) + torch.tensor(THETA_MEAN[:2], device=x.device)
        y_phys = y[:, :2] * torch.tensor(THETA_STD[:2], device=x.device) + torch.tensor(THETA_MEAN[:2], device=x.device)
        val_mse = F.mse_loss(mean, y_phys)

        self.log("val_nll", val_nll, prog_bar=True)
        self.log("val_mse", val_mse, prog_bar=True)
        return val_nll

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.flow.parameters(), lr=self.hparams.max_lr, weight_decay=1e-5
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = total_steps // self.trainer.max_epochs
        step_size = self.hparams.decay_every_epochs * steps_per_epoch

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-10, end_factor=1.0,
            total_iters=self.hparams.warmup_steps,
        )
        decay = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=self.hparams.decay_rate,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay],
            milestones=[self.hparams.warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ------------------------------------------------------------------
    # Posterior inference
    # ------------------------------------------------------------------

    def sample_posterior(
        self,
        x: torch.Tensor,
        n_samples: int = 1000,
        n_noise_avg: int = 4,
    ) -> torch.Tensor:
        """Draw posterior samples p(Omega_m, sigma_8 | x) in physical units.

        Args:
            x: Noiseless input convergence map(s), shape (H, W) or (B, H, W).
               Noise is added internally for consistency with training.
            n_samples: Number of posterior samples to draw per map.
            n_noise_avg: Number of noise realisations to average the context
                         over for increased stability.

        Returns:
            Tensor of shape (n_samples, 2) in physical units (Omega_m, sigma_8),
            or (B, n_samples, 2) if B > 1.
        """
        self.eval()
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)  # (1, H, W)

        mask_t = torch.tensor(self.mask, device=x.device, dtype=x.dtype).unsqueeze(0)
        theta_std = torch.tensor(THETA_STD[:2], device=x.device, dtype=x.dtype)
        theta_mean_t = torch.tensor(THETA_MEAN[:2], device=x.device, dtype=x.dtype)

        with torch.no_grad():
            # Average context over noise realisations for stability
            B = x.size(0)
            z = torch.zeros(B, self.compressor.hparams.summary_dim, device=x.device)
            for _ in range(n_noise_avg):
                x_noisy = (x + torch.randn_like(x) * NOISE_STD) * mask_t
                z = z + self.compressor.compress(x_noisy)
            z = z / n_noise_avg

            # Sample from the flow for each observation in the batch
            results = []
            for i in range(B):
                z_i = z[i].unsqueeze(0).expand(n_samples, -1)  # (n_samples, summary_dim)
                samples_norm = self.flow.sample(n_samples, context=z_i)  # (n_samples, 2)
                samples_phys = samples_norm * theta_std + theta_mean_t
                results.append(samples_phys)

        if squeeze:
            return results[0]  # (n_samples, 2)
        return torch.stack(results, dim=0)  # (B, n_samples, 2)
