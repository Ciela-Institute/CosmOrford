import os
import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosmoford import THETA_MEAN, THETA_STD, SURVEY_MASK, NOISE_STD
from cosmoford.summaries import (power_spectrum_batch,
                                  compute_wavelet_peaks_batch,
                                  compute_higher_order_statistics_batch,
                                  compute_wavelet_l1_norms_batch,
                                  compute_scattering_batch,
                                  scattering_n_coefficients)
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b2, efficientnet_v2_s, efficientnet_v2_m


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


class RegressionModel(L.LightningModule):

  def __init__(self, backbone="efficientnet_b0",
               use_cnn: bool = True,
               use_ps: bool = True,
               use_hos: bool = False, hos_n_scales: int = 5, hos_n_bins: int = 31,
               hos_l1_nbins: int = 40, hos_min_snr: float = -4.0, hos_max_snr: float = 8.0,
               hos_l1_min_snr: float = -8.0, hos_l1_max_snr: float = 8.0,
               hos_compute_mono: bool = False, hos_l1_only: bool = False,
               hos_peaks_only: bool = False,
                use_scattering: bool = False, scattering_J: int = 5, scattering_L: int = 8,
                 scattering_normalization: str = "log1p_zscore",
                scattering_feature_pooling: str = "mean",
                scattering_mask_pooling: str = "soft",
                scattering_geometry: str = "reduced",
                augment_flip: bool = True,
                augment_shift: bool = True,
                 warmup_steps: int = 1000, max_lr: float = 0.256,
                decay_rate: float = 0.97, decay_every_epochs: int = 2, dropout_rate: float = 0.2,
                loss_type: str = "log_prob", freeze_backbone: bool = False,
                pretrained_checkpoint_path: str = None):
    super().__init__()
    self.save_hyperparameters(ignore=['pretrained_checkpoint_path'])

    # Validate loss_type parameter
    if loss_type not in ["log_prob", "score"]:
      raise ValueError(f"loss_type must be 'log_prob' or 'score', got '{loss_type}'")
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

    # CNN backbone (optional)
    if use_cnn:
      last_dim = 1280  # For efficientnet_b0
      if backbone == "efficientnet_b0":
        vision_model = efficientnet_b0()
      elif backbone == "efficientnet_b2":
        vision_model = efficientnet_b2()
        last_dim = 1408
      elif backbone == "efficientnet_v2_s":
        vision_model = efficientnet_v2_s()
      elif backbone == "efficientnet_v2_m":
        vision_model = efficientnet_v2_m()
      else:
        raise ValueError(f"Backbone {backbone} not supported.")
      self.model = vision_model.features
      self.reshape_head = nn.Sequential(
       nn.AdaptiveAvgPool2d(1),
       nn.Flatten(),
       nn.Dropout(p=self.hparams.dropout_rate, inplace=True),
      )
    else:
      last_dim = 0
      self.model = None
      self.reshape_head = None

    # Power spectrum head (optional; disabled when use_ps=False for ablation studies)
    if use_ps:
      self.ps_head = nn.Sequential(
       nn.Linear(10, 256),
       nn.LeakyReLU(),
       nn.Linear(256, 256),
       nn.LeakyReLU(),
       nn.Linear(256, 128),
       nn.LeakyReLU()
      )
      ps_processed_dim = 128
    else:
      self.ps_head = None
      ps_processed_dim = 0

    # Higher-order statistics head (optional)
    if use_hos:
      if hos_l1_only:
        hos_dim = hos_n_scales * hos_l1_nbins
      elif hos_peaks_only:
        hos_dim = hos_n_scales * hos_n_bins
      elif hos_compute_mono:
        hos_dim = hos_n_bins + hos_n_scales * (hos_n_bins + hos_l1_nbins)
      else:
        hos_dim = hos_n_scales * (hos_n_bins + hos_l1_nbins)
      self.hos_head = nn.Sequential(
        nn.Linear(hos_dim, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 128),
        nn.LeakyReLU(),
      )
      hos_processed_dim = 128
    else:
      self.hos_head = None
      hos_processed_dim = 0

    # Scattering transform head (optional)
    if use_scattering:
      scat_dim = scattering_n_coefficients(
        scattering_J,
        scattering_L,
        feature_pooling=scattering_feature_pooling,
      )
      self.scattering_head = nn.Sequential(
        nn.Linear(scat_dim, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 128),
        nn.LeakyReLU(),
      )
      scat_processed_dim = 128
    else:
      self.scattering_head = None
      scat_processed_dim = 0

    total_dim = last_dim + ps_processed_dim + hos_processed_dim + scat_processed_dim
    if total_dim == 0:
      raise ValueError(
        "At least one of use_cnn, use_ps, use_hos, or use_scattering must be enabled."
      )
    self.head = nn.Sequential(
     nn.Linear(total_dim, 128),
     nn.LeakyReLU(),
     nn.Linear(128, 2*2)  # mean and log-variance for (Omega_m, sigma_8)
    )

    # Load pretrained weights if checkpoint path is provided
    if pretrained_checkpoint_path is not None:
      self.load_pretrained_weights(pretrained_checkpoint_path)

    # Freeze backbone and power spectrum head if in fine-tuning mode
    if freeze_backbone:
      self.freeze_backbone_layers()

  def forward(self, x):

    x_orig = x  # keep for HOS/scattering (unpatched)

    # Power spectrum
    if self.hparams.use_ps:
      with torch.no_grad():
        _, ps_features = power_spectrum_batch(x_orig)
      ps_features = self.ps_head(ps_features)

    # Higher-order statistics
    if self.hparams.use_hos:
      with torch.no_grad():
        if self.hparams.hos_l1_only:
          hos_features = compute_wavelet_l1_norms_batch(
            x_orig,
            noise_std=NOISE_STD,
            mask=torch.from_numpy(self.mask).to(x.device),
            n_scales=self.hparams.hos_n_scales,
            pixel_arcmin=2.0,
            l1_nbins=self.hparams.hos_l1_nbins,
            l1_min_snr=self.hparams.hos_l1_min_snr,
            l1_max_snr=self.hparams.hos_l1_max_snr,
            normalize=True,
          )
        elif self.hparams.hos_peaks_only:
          hos_features = compute_wavelet_peaks_batch(
            x_orig,
            noise_std=NOISE_STD,
            mask=torch.from_numpy(self.mask).to(x.device),
            n_scales=self.hparams.hos_n_scales,
            pixel_arcmin=2.0,
            n_bins=self.hparams.hos_n_bins,
            min_snr=self.hparams.hos_min_snr,
            max_snr=self.hparams.hos_max_snr,
            normalize=True,
          )
        else:
          hos_features = compute_higher_order_statistics_batch(
            x_orig,
            noise_std=NOISE_STD,
            mask=torch.from_numpy(self.mask).to(x.device),
            n_scales=self.hparams.hos_n_scales,
            pixel_arcmin=2.0,
            n_bins=self.hparams.hos_n_bins,
            l1_nbins=self.hparams.hos_l1_nbins,
            min_snr=self.hparams.hos_min_snr,
            max_snr=self.hparams.hos_max_snr,
            l1_min_snr=self.hparams.hos_l1_min_snr,
            l1_max_snr=self.hparams.hos_l1_max_snr,
            compute_mono=self.hparams.hos_compute_mono,
            mono_smoothing_sigma=2.0,
            normalize=True,
          )
      hos_features = self.hos_head(hos_features)

    # Scattering transform
    if self.hparams.use_scattering:
      with torch.no_grad():
        if self.hparams.scattering_geometry == "full":
          scat_maps = _inverse_reshape_field(x_orig)
          survey_mask = torch.from_numpy(self.mask_full).to(x.device, dtype=x_orig.dtype)
        else:
          scat_maps = x_orig
          survey_mask = torch.from_numpy(self.mask_reduced).to(x.device, dtype=x_orig.dtype)
        scat_features = compute_scattering_batch(
          scat_maps,
          J=self.hparams.scattering_J,
          L=self.hparams.scattering_L,
          normalize=True,
          normalization=self.hparams.scattering_normalization,
          mask=survey_mask,
          mask_pooling=self.hparams.scattering_mask_pooling,
          feature_pooling=self.hparams.scattering_feature_pooling,
        )
      scat_features = self.scattering_head(scat_features)

    # CNN backbone (optional)
    if self.hparams.use_cnn:
      if x.dim() == 3:
        x = x.unsqueeze(1)
      if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
      x = F.pad(x, pad=(0, 0, 0, 14), mode='constant', value=0)
      x = x.reshape((-1, 3, 21, 88, 88))
      x = x.permute(0, 2, 1, 3, 4).reshape(-1, 3, 88, 88)

      if self.training:
        angles = torch.randint(0, 4, (x.size(0),), device=x.device)
        for k in range(4):
          mask = angles == k
          if mask.any():
            x[mask] = torch.rot90(x[mask], k=k, dims=(2, 3))

      cnn_features = self.model(x.float())
      cnn_features = cnn_features.reshape((-1, 21, cnn_features.size(1), cnn_features.size(2), cnn_features.size(3)))
      cnn_features = cnn_features.mean(dim=1)
      cnn_features = self.reshape_head(cnn_features)

      feature_parts = [cnn_features]
      if self.hparams.use_ps:
        feature_parts.append(ps_features)
    else:
      feature_parts = [ps_features] if self.hparams.use_ps else []

    if self.hparams.use_hos:
      feature_parts.append(hos_features)
    if self.hparams.use_scattering:
      feature_parts.append(scat_features)

    features = torch.cat(feature_parts, dim=1)
    x = self.head(features)
    return x[..., :2], F.softplus(x[..., 2:]) + 0.001  # mean and scale for (Omega_m, sigma_8)

  def load_pretrained_weights(self, checkpoint_path: str):
    """Load weights from a pretrained checkpoint.
    Only loads the model weights, not the hyperparameters or optimizer state."""
    print(f"\nLoading pretrained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Lightning checkpoints store the model state_dict under 'state_dict' key
    if 'state_dict' in checkpoint:
      state_dict = checkpoint['state_dict']
    else:
      state_dict = checkpoint

    # Load the weights
    self.load_state_dict(state_dict, strict=True)
    print("Pretrained weights loaded successfully!\n")

  def freeze_backbone_layers(self):
    """Freeze the CNN backbone and power spectrum head for fine-tuning."""
    if self.model is not None:
      for param in self.model.parameters():
        param.requires_grad = False
    for param in self.ps_head.parameters():
      param.requires_grad = False
    print("\nBackbone and ps_head frozen. Only reshape_head and head are trainable.")
    self.print_trainable_parameters()

  def print_trainable_parameters(self):
    """Print statistics about trainable vs frozen parameters"""
    trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in self.parameters())
    frozen_params = total_params - trainable_params

    print(f"{'='*60}")
    print(f"Parameter Statistics (Fine-tuning Mode)")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Frozen parameters:    {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
    print(f"{'='*60}\n")

  def training_step(self, batch, batch_idx):
    x, y = batch

    # Adding noise to the input convergence maps
    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise

    # Adding augmentations BEFORE masking so that the survey mask always covers
    # the same region of the map.  Applying augmentations after masking shifts
    # the zero-filled masked pixels to random positions, which creates spurious
    # mask-boundary features in the scattering transform (and any other
    # summary statistic that does not take an explicit mask argument).
    batch_size = x.size(0)
    if self.hparams.augment_flip:
      # Random flips along nx dimension (dim=1)
      flip_lr = torch.rand(batch_size, device=x.device) < 0.5
      x[flip_lr] = torch.flip(x[flip_lr], dims=[1])
      # Random flips along ny dimension (dim=2)
      flip_ud = torch.rand(batch_size, device=x.device) < 0.5
      x[flip_ud] = torch.flip(x[flip_ud], dims=[2])

    if self.hparams.augment_shift:
      # Random cyclic shifts (different for each sample) in nx and ny
      shift_x = torch.randint(low=0, high=x.size(1), size=(batch_size,), device=x.device)
      x = torch.stack([torch.roll(x[i], shifts=(shift_x[i].item(),), dims=(0,)) for i in range(batch_size)])
      shift_y = torch.randint(low=0, high=x.size(2), size=(batch_size,), device=x.device)
      x = torch.stack([torch.roll(x[i], shifts=(shift_y[i].item(),), dims=(1,)) for i in range(batch_size)])

    # Apply mask after augmentation so the survey footprint is always at the
    # same location in the (augmented) map.
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

    mean, std = self(x)

    if self.hparams.loss_type == "log_prob":
      loss = - torch.distributions.Normal(loc=mean, scale=std).log_prob(y)
      loss = loss.mean()
    elif self.hparams.loss_type == "score":
      # Rescaling back to original parameters
      mean = mean * torch.tensor(THETA_STD[:2], device=mean.device) + torch.tensor(THETA_MEAN[:2], device=mean.device)
      std = std * torch.tensor(THETA_STD[:2], device=std.device)
      y = y[:, :2] * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)

      # Use the Phase 1 score as the loss (negative because we minimize)
      sq_error = (y - mean) ** 2
      scale_factor = 1000.0
      score = -torch.sum(sq_error / std**2 + torch.log(std**2) + scale_factor * sq_error, dim=1)
      loss = -torch.mean(score)  # Negative because we want to maximize the score

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch

    # Adding noise to the input convergence maps and applying mask
    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

    mean, std = self(x)

    # Rescaling back to original parameters, for log_prob loss we do it here
    mean = mean * torch.tensor(THETA_STD[:2], device=mean.device) + torch.tensor(THETA_MEAN[:2], device=mean.device)
    std = std * torch.tensor(THETA_STD[:2], device=std.device)
    y = y[:, :2] * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)

    # Compute the Phase 1 score (torch version)
    sq_error = (y - mean) ** 2
    scale_factor = 1000.0
    score = -torch.sum(sq_error / std**2 + torch.log(std**2) + scale_factor * sq_error, dim=1)
    score = torch.mean(score)

    mse = F.mse_loss(mean, y)

    self.log('val_score', score, prog_bar=True)
    self.log('val_mse', mse, prog_bar=True)
    return score

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.max_lr, weight_decay=1e-5)

    # Calculate total steps
    total_steps = int(self.trainer.estimated_stepping_batches)
    warmup_steps = self.hparams.warmup_steps

    # Calculate step size for StepLR in terms of steps (not epochs)
    steps_per_epoch = total_steps // self.trainer.max_epochs
    step_size_in_steps = self.hparams.decay_every_epochs * steps_per_epoch

    # Linear warmup from 0 to max_lr
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-10,  # Start from nearly 0
        end_factor=1.0,      # End at max_lr
        total_iters=warmup_steps,
    )

    # Step decay after warmup (in steps, not epochs)
    decay = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size_in_steps,
        gamma=self.hparams.decay_rate,
    )

    # Combine warmup and decay
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",  # Both warmup and decay operate on steps
            "frequency": 1,
        },
    }
