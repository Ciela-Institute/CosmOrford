import os
import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosmoford import THETA_MEAN, THETA_STD, SURVEY_MASK, NOISE_STD
from cosmoford.summaries import (
  power_spectrum_batch,
  compute_wavelet_peaks_batch,
  compute_wavelet_l1_norms_batch,
  compute_higher_order_statistics_batch,
  compute_scattering_batch,
  scattering_n_coefficients,
)
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b2, efficientnet_v2_s, efficientnet_v2_m
from torchvision.models.resnet import resnet18, ResNet18_Weights
try:
  from peft import LoraConfig, get_peft_model
except ImportError:  # Optional dependency for LoRA workflows
  LoraConfig = None
  get_peft_model = None
try:
  from nflows.flows import Flow
  from nflows.distributions import StandardNormal
  from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform, RandomPermutation
except ImportError:  # Optional dependency unless flow training is enabled
  Flow = None
  StandardNormal = None
  CompositeTransform = None
  MaskedAffineAutoregressiveTransform = None
  RandomPermutation = None


def build_flow(param_dim=2, context_dim=8, n_transforms=4, hidden_dim=64):
  """Build a small conditional MAF: p(params | summaries)."""
  if Flow is None or StandardNormal is None:
    raise ImportError(
      "build_flow requires the optional 'nflows' dependency. "
      "Install nflows to enable posterior-flow training."
    )
  transforms = []
  for _ in range(n_transforms):
    transforms.append(RandomPermutation(features=param_dim))
    transforms.append(MaskedAffineAutoregressiveTransform(
      features=param_dim,
      hidden_features=hidden_dim,
      context_features=context_dim,
    ))
  return Flow(
    transform=CompositeTransform(transforms),
    distribution=StandardNormal([param_dim]),
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


def _stats_input_dim(
  use_ps: bool,
  use_hos: bool,
  hos_l1_only: bool,
  hos_peaks_only: bool,
  hos_n_scales: int,
  hos_n_bins: int,
  hos_l1_nbins: int,
  use_scattering: bool,
  scattering_J: int,
  scattering_L: int,
  scattering_feature_pooling: str,
) -> int:
  dim = 0
  if use_ps:
    dim += 10
  if use_hos:
    if hos_l1_only:
      dim += hos_n_scales * hos_l1_nbins
    elif hos_peaks_only:
      dim += hos_n_scales * hos_n_bins
    else:
      dim += hos_n_scales * (hos_n_bins + hos_l1_nbins)
  if use_scattering:
    dim += scattering_n_coefficients(
      scattering_J,
      scattering_L,
      feature_pooling=scattering_feature_pooling,
    )
  return dim


class StatsCompressorNoPatch(L.LightningModule):

  def __init__(
    self,
    summary_dim: int = 8,
    summary_hidden_dim: int = 512,
    summary_n_hidden: int = 3,
    summary_dropout_rate: float = 0.1,
    use_ps: bool = False,
    use_hos: bool = True,
    hos_l1_only: bool = False,
    hos_peaks_only: bool = False,
    hos_n_scales: int = 4,
    hos_n_bins: int = 51,
    hos_l1_nbins: int = 80,
    hos_min_snr: float = -3.0,
    hos_max_snr: float = 7.0,
    hos_l1_min_snr: float = -7.0,
    hos_l1_max_snr: float = 7.0,
    use_scattering: bool = False,
    scattering_J: int = 4,
    scattering_L: int = 8,
    scattering_normalization: str = "log1p_zscore",
    scattering_feature_pooling: str = "mean",
    scattering_mask_pooling: str = "soft",
    scattering_geometry: str = "reduced",
    augment_flip: bool = True,
    augment_shift: bool = True,
    warmup_steps: int = 500,
    max_lr: float = 1.0e-3,
    decay_rate: float = 0.85,
    decay_every_epochs: int = 1,
    loss_type: str = "log_prob",
    use_flow: bool = False,
    flow_transforms: int = 4,
    flow_hidden_dim: int = 64,
    lr_schedule: str = "step",
    total_steps: int = 0,
    n_val_noise: int = 1,
  ):
    super().__init__()
    self.save_hyperparameters()

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
    if hos_l1_only and hos_peaks_only:
      raise ValueError("hos_l1_only and hos_peaks_only cannot both be True.")
    if use_flow and Flow is None:
      raise ImportError(
        "use_flow=True requires the optional 'nflows' dependency. "
        "Install nflows or set use_flow=False."
      )

    self.mask_reduced = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])
    self.mask_full = SURVEY_MASK
    self.mask = self.mask_reduced

    input_dim = _stats_input_dim(
      use_ps=use_ps,
      use_hos=use_hos,
      hos_l1_only=hos_l1_only,
      hos_peaks_only=hos_peaks_only,
      hos_n_scales=hos_n_scales,
      hos_n_bins=hos_n_bins,
      hos_l1_nbins=hos_l1_nbins,
      use_scattering=use_scattering,
      scattering_J=scattering_J,
      scattering_L=scattering_L,
      scattering_feature_pooling=scattering_feature_pooling,
    )
    if input_dim == 0:
      raise ValueError("At least one of use_ps, use_hos, or use_scattering must be True.")

    layers = [
      nn.Linear(input_dim, summary_hidden_dim),
      nn.GELU(),
    ]
    for _ in range(max(summary_n_hidden - 1, 0)):
      layers.extend([
        nn.Linear(summary_hidden_dim, summary_hidden_dim),
        nn.GELU(),
        nn.Dropout(summary_dropout_rate),
      ])
    layers.append(nn.Linear(summary_hidden_dim, summary_dim))
    self.compressor = nn.Sequential(*layers)

    if use_flow:
      self.flow = build_flow(
        param_dim=2,
        context_dim=summary_dim,
        n_transforms=flow_transforms,
        hidden_dim=flow_hidden_dim,
      )
    self.head = nn.Sequential(
      nn.GELU(),
      nn.Linear(summary_dim, summary_dim * 4),
      nn.GELU(),
      nn.Linear(summary_dim * 4, 2 * 2),
    )

  def _compute_stats(self, x):
    parts = []
    mask_reduced_t = torch.tensor(self.mask_reduced, device=x.device, dtype=x.dtype)
    mask_full_t = torch.tensor(self.mask_full, device=x.device, dtype=x.dtype)
    with torch.no_grad():
      if self.hparams.use_ps:
        _, ps_features = power_spectrum_batch(x)
        parts.append(ps_features)

      if self.hparams.use_hos:
        if self.hparams.hos_l1_only:
          hos_features = compute_wavelet_l1_norms_batch(
            x,
            noise_std=NOISE_STD,
            mask=mask_reduced_t,
            n_scales=self.hparams.hos_n_scales,
            pixel_arcmin=2.0,
            l1_nbins=self.hparams.hos_l1_nbins,
            l1_min_snr=self.hparams.hos_l1_min_snr,
            l1_max_snr=self.hparams.hos_l1_max_snr,
            normalize=True,
          )
        elif self.hparams.hos_peaks_only:
          hos_features = compute_wavelet_peaks_batch(
            x,
            noise_std=NOISE_STD,
            mask=mask_reduced_t,
            n_scales=self.hparams.hos_n_scales,
            pixel_arcmin=2.0,
            n_bins=self.hparams.hos_n_bins,
            min_snr=self.hparams.hos_min_snr,
            max_snr=self.hparams.hos_max_snr,
            normalize=True,
          )
        else:
          hos_features = compute_higher_order_statistics_batch(
            x,
            noise_std=NOISE_STD,
            mask=mask_reduced_t,
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
        parts.append(hos_features)

      if self.hparams.use_scattering:
        if self.hparams.scattering_geometry == "full":
          scat_maps = _inverse_reshape_field(x)
          scat_mask = mask_full_t
        else:
          scat_maps = x
          scat_mask = mask_reduced_t
        scat_features = compute_scattering_batch(
          scat_maps,
          J=self.hparams.scattering_J,
          L=self.hparams.scattering_L,
          normalize=True,
          normalization=self.hparams.scattering_normalization,
          mask=scat_mask,
          mask_pooling=self.hparams.scattering_mask_pooling,
          feature_pooling=self.hparams.scattering_feature_pooling,
        )
        parts.append(scat_features)

    return torch.cat(parts, dim=1)

  def forward(self, x):
    summaries = self.compress(x)
    out = self.head(summaries)
    return out[..., :2], F.softplus(out[..., 2:]) + 0.001, summaries

  def compress(self, x):
    return self.compressor(self._compute_stats(x))

  def training_step(self, batch, batch_idx):
    x, y = batch

    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

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

    mean, std, summaries = self(x)
    if self.hparams.use_flow:
      loss = -self.flow.log_prob(y, context=summaries).mean()
    elif self.hparams.loss_type == "log_prob":
      loss = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
    else:
      mean = mean * torch.tensor(THETA_STD[:2], device=mean.device) + torch.tensor(THETA_MEAN[:2], device=mean.device)
      std = std * torch.tensor(THETA_STD[:2], device=std.device)
      y = y[:, :2] * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)
      sq_error = (y - mean) ** 2
      score = -torch.sum(sq_error / std**2 + torch.log(std**2) + 1000.0 * sq_error, dim=1)
      loss = -torch.mean(score)

    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    mask = torch.tensor(self.mask, device=x.device).unsqueeze(0)

    total_mean = 0.0
    total_std = 0.0
    total_summaries = 0.0
    for _ in range(self.hparams.n_val_noise):
      x_noisy = (x + torch.randn_like(x) * NOISE_STD) * mask
      m, s, summ = self(x_noisy)
      total_mean += m
      total_std += s
      total_summaries += summ
    mean = total_mean / self.hparams.n_val_noise
    std = total_std / self.hparams.n_val_noise
    summaries = total_summaries / self.hparams.n_val_noise

    if self.hparams.use_flow:
      nll = -self.flow.log_prob(y, context=summaries).mean()
      self.log("val_nll", nll, prog_bar=True)
      return nll

    loss_val = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
    self.log("val_log_prob", loss_val, prog_bar=True)

    mean = mean * torch.tensor(THETA_STD[:2], device=mean.device) + torch.tensor(THETA_MEAN[:2], device=mean.device)
    std = std * torch.tensor(THETA_STD[:2], device=std.device)
    y = y[:, :2] * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)

    sq_error = (y - mean) ** 2
    scale_factor = 1000.0
    score = -torch.sum(sq_error / std**2 + torch.log(std**2) + scale_factor * sq_error, dim=1)
    score = torch.mean(score)

    mse = F.mse_loss(mean, y)
    self.log("val_score", score, prog_bar=True)
    self.log("val_mse", mse, prog_bar=True)
    return score

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.max_lr, weight_decay=1e-5)
    if self.hparams.total_steps > 0:
      total_steps = self.hparams.total_steps
    else:
      total_steps = int(self.trainer.estimated_stepping_batches)
    warmup_steps = self.hparams.warmup_steps

    warmup = torch.optim.lr_scheduler.LinearLR(
      optimizer,
      start_factor=1e-10,
      end_factor=1.0,
      total_iters=warmup_steps,
    )

    if self.hparams.lr_schedule == "cosine":
      main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
      )
    else:
      steps_per_epoch = total_steps // self.trainer.max_epochs
      step_size_in_steps = self.hparams.decay_every_epochs * steps_per_epoch
      main_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size_in_steps,
        gamma=self.hparams.decay_rate,
      )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
      optimizer,
      schedulers=[warmup, main_scheduler],
      milestones=[warmup_steps],
    )

    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": scheduler,
        "interval": "step",
        "frequency": 1,
      },
    }


class RegressionModelNoPatch(L.LightningModule):

  def __init__(self, backbone="efficientnet_b0", summary_dim: int = 8,
               warmup_steps: int = 1000, max_lr: float = 0.256,
               decay_rate: float = 0.97, decay_every_epochs: int = 2, dropout_rate: float = 0.2,
               loss_type: str = "log_prob", freeze_backbone: bool = False,
               use_flow: bool = False, flow_transforms: int = 4, flow_hidden_dim: int = 64,
               pretrained_checkpoint_path: str = None,
               pretrained: bool = False, lr_schedule: str = "step",
               total_steps: int = 0,
               n_val_noise: int = 1,
               use_peft: bool = False, lora_r: int = 8, lora_alpha: int = 16,
               lora_dropout: float = 0.1, lora_target_modules: list = None):
    super().__init__()

    self.mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    last_dim = 1280  # For efficientnet_b0
    if backbone == "resnet18":
      vision_model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
      # ResNet: use all layers except avgpool and fc
      self.model = nn.Sequential(
        vision_model.conv1, vision_model.bn1, vision_model.relu, vision_model.maxpool,
        vision_model.layer1, vision_model.layer2, vision_model.layer3, vision_model.layer4,
      )
      last_dim = 512
    elif backbone == "efficientnet_b0":
      vision_model = efficientnet_b0()
      self.model = vision_model.features
    elif backbone == "efficientnet_b2":
      vision_model = efficientnet_b2()
      last_dim = 1408
      self.model = vision_model.features
    elif backbone == "efficientnet_v2_s":
      vision_model = efficientnet_v2_s()
      self.model = vision_model.features
    elif backbone == "efficientnet_v2_m":
      vision_model = efficientnet_v2_m()
      self.model = vision_model.features
    else:
      raise ValueError(f"Backbone {backbone} not supported.")

    # Adapt first conv layer from 3-channel to 1-channel input
    if pretrained:
      self._adapt_first_conv(backbone)

    # Store use_peft flag for later use
    self._use_peft = use_peft
    self._lora_r = lora_r
    self._lora_alpha = lora_alpha
    self._lora_dropout = lora_dropout
    self._lora_target_modules = lora_target_modules
    if use_flow and Flow is None:
      raise ImportError(
        "use_flow=True requires the optional 'nflows' dependency. "
        "Install nflows or set use_flow=False."
      )
    if use_peft and (LoraConfig is None or get_peft_model is None):
      raise ImportError(
        "use_peft=True requires the optional 'peft' dependency. "
        "Install peft or disable use_peft."
      )

    # Apply PEFT/LoRA if enabled (but only if we're not loading from a pretrained checkpoint)
    # If pretrained_checkpoint_path is provided, we'll apply LoRA after loading the weights
    if use_peft and pretrained_checkpoint_path is None:
      # Default target modules for EfficientNet (Conv2d layers in blocks)
      # We target only Conv2d layers with groups=1 (not depthwise convolutions)
      if lora_target_modules is None:
        lora_target_modules = []
        for name, module in self.model.named_modules():
          if isinstance(module, nn.Conv2d):
            # Skip depthwise convolutions (groups > 1) as they have restrictions with LoRA
            # For depthwise conv, the rank must be divisible by groups
            if module.groups == 1:
              # Only add standard convolutions
              parts = name.split('.')
              if len(parts) >= 2 and parts[-1] == '0':  # Conv2d is usually at position 0 in Sequential
                lora_target_modules.append('.'.join(parts))

        # Remove duplicates and keep only unique patterns - IMPORTANT: sort for determinism
        lora_target_modules = sorted(list(set(lora_target_modules)))
        print(f"Auto-detected {len(lora_target_modules)} standard Conv2d layers for LoRA (excluding depthwise convolutions)")

      lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,  # For non-text models
      )

      self.model = get_peft_model(self.model, lora_config)
      print(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
      print(f"Trainable parameters:")
      self.model.print_trainable_parameters()

      # CRITICAL: Save the actual target modules used so checkpoint loading works correctly
      # We need to save hyperparameters AFTER determining lora_target_modules
      lora_target_modules_used = lora_target_modules
    else:
      lora_target_modules_used = None

    # Save hyperparameters with the actual lora_target_modules that were used
    self.save_hyperparameters({
      'backbone': backbone,
      'warmup_steps': warmup_steps,
      'max_lr': max_lr,
      'decay_rate': decay_rate,
      'decay_every_epochs': decay_every_epochs,
      'dropout_rate': dropout_rate,
      'summary_dim': summary_dim,
      'loss_type': loss_type,
      'freeze_backbone': freeze_backbone,
      'use_flow': use_flow,
      'flow_transforms': flow_transforms,
      'flow_hidden_dim': flow_hidden_dim,
      'pretrained': pretrained,
      'lr_schedule': lr_schedule,
      'total_steps': total_steps,
      'n_val_noise': n_val_noise,
      'use_peft': use_peft,
      'lora_r': lora_r,
      'lora_alpha': lora_alpha,
      'lora_dropout': lora_dropout,
      'lora_target_modules': lora_target_modules_used  # Save the actual modules used, not None
    })

    self.reshape_head = nn.Sequential(
     nn.AdaptiveAvgPool2d(1),
     nn.Flatten(),
     nn.Dropout(p=self.hparams.dropout_rate, inplace=True),
    )
    self.compressor = nn.Linear(last_dim, summary_dim)
    if use_flow:
      self.flow = build_flow(param_dim=2, context_dim=summary_dim,
                             n_transforms=flow_transforms, hidden_dim=flow_hidden_dim)
    self.head = nn.Sequential(
     nn.GELU(),
     nn.Linear(summary_dim, summary_dim * 4),
     nn.GELU(),
     nn.Linear(summary_dim * 4, 2*2) # mean and log-std for Ω_m, S_8
    )

    # Load pretrained weights if checkpoint path is provided
    if pretrained_checkpoint_path is not None:
      self.load_pretrained_weights(pretrained_checkpoint_path)

    # Freeze backbone and power spectrum head if in fine-tuning mode
    if freeze_backbone:
      self.freeze_backbone_layers()

  def _adapt_first_conv(self, backbone):
    """Replace first conv layer from 3-channel to 1-channel input.
    Sums pretrained 3-channel weights into a single channel."""
    if backbone == "resnet18":
      old_conv = self.model[0]  # conv1
    elif backbone.startswith("efficientnet"):
      old_conv = self.model[0][0]  # features[0] is first ConvBNActivation block

    new_conv = nn.Conv2d(
      1, old_conv.out_channels,
      kernel_size=old_conv.kernel_size,
      stride=old_conv.stride,
      padding=old_conv.padding,
      bias=old_conv.bias is not None,
    )
    with torch.no_grad():
      new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
      if old_conv.bias is not None:
        new_conv.bias.copy_(old_conv.bias)

    if backbone == "resnet18":
      self.model[0] = new_conv
    elif backbone.startswith("efficientnet"):
      self.model[0][0] = new_conv

  def forward(self, x):
    # Add channel dimension if missing
    if x.dim() == 3:
      x = x.unsqueeze(1)
    # Repeat channels to match expected input size (only if not using pretrained single-channel adaptation)
    if x.size(1) == 1 and not self.hparams.pretrained:
      x = x.repeat(1, 3, 1, 1)

    # Compute features
    features = self.model(x.float())
    features = self.reshape_head(features)

    # Compress to summary statistics then predict Gaussian parameters
    summaries = self.compressor(features)
    x = self.head(summaries)
    return x[..., :2], F.softplus(x[..., 2:]) + 0.001, summaries  # mean, std, summaries

  def compress(self, x):
    """Return the compressed summary representation."""
    if x.dim() == 3:
      x = x.unsqueeze(1)
    if x.size(1) == 1 and not self.hparams.pretrained:
      x = x.repeat(1, 3, 1, 1)
    features = self.model(x.float())
    features = self.reshape_head(features)
    return self.compressor(features)

  def load_pretrained_weights(self, checkpoint_path: str):
    """Load weights from a pretrained checkpoint.
    Only loads the model weights, not the hyperparameters or optimizer state.
    If use_peft is enabled, applies LoRA after loading the base weights."""
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

    # Apply LoRA after loading pretrained weights if use_peft is enabled
    if self._use_peft:
      if LoraConfig is None or get_peft_model is None:
        raise ImportError(
          "use_peft=True requires the optional 'peft' dependency. "
          "Install peft or disable use_peft."
        )
      print("Applying LoRA to pretrained model...")
      lora_target_modules = self._lora_target_modules

      # Auto-detect target modules if not specified
      if lora_target_modules is None:
        lora_target_modules = []
        for name, module in self.model.named_modules():
          if isinstance(module, nn.Conv2d):
            # Skip depthwise convolutions (groups > 1)
            if module.groups == 1:
              parts = name.split('.')
              if len(parts) >= 2 and parts[-1] == '0':
                lora_target_modules.append('.'.join(parts))

        # Remove duplicates and sort for determinism
        lora_target_modules = sorted(list(set(lora_target_modules)))
        print(f"Auto-detected {len(lora_target_modules)} standard Conv2d layers for LoRA")

      lora_config = LoraConfig(
        r=self._lora_r,
        lora_alpha=self._lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=self._lora_dropout,
        bias="none",
        task_type=None,
      )

      self.model = get_peft_model(self.model, lora_config)
      print(f"Applied LoRA with r={self._lora_r}, alpha={self._lora_alpha}, dropout={self._lora_dropout}")
      print("Trainable parameters:")
      self.model.print_trainable_parameters()

      # Update hyperparameters with the actual lora_target_modules
      self.hparams.lora_target_modules = lora_target_modules

  def freeze_backbone_layers(self):
    """Freeze the backbone (vision model) and power spectrum head for fine-tuning.
    Only the final regression heads (reshape_head and head) remain trainable."""
    for param in self.model.parameters():
      param.requires_grad = False
    # Keep self.reshape_head and self.head trainable
    print("\nBackbone frozen. Only reshape_head and head are trainable.")
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

    # Adding noise to the input convergence maps and applying mask
    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

    # Adding augmentations, random left-right and up-down flips (per sample)
    batch_size = x.size(0)
    # Random flips along nx dimension (dim=1)
    flip_lr = torch.rand(batch_size, device=x.device) < 0.5
    x[flip_lr] = torch.flip(x[flip_lr], dims=[1])

    # Random flips along ny dimension (dim=2)
    flip_ud = torch.rand(batch_size, device=x.device) < 0.5
    x[flip_ud] = torch.flip(x[flip_ud], dims=[2])
    # Adding random cyclic shifts (different for each sample) in nx and ny
    shift_x = torch.randint(low=0, high=x.size(1), size=(batch_size,), device=x.device)
    x = torch.stack([torch.roll(x[i], shifts=(shift_x[i].item(),), dims=(0,)) for i in range(batch_size)])

    shift_y = torch.randint(low=0, high=x.size(2), size=(batch_size,), device=x.device)
    x = torch.stack([torch.roll(x[i], shifts=(shift_y[i].item(),), dims=(1,)) for i in range(batch_size)])

    mean, std, summaries = self(x)

    if self.hparams.use_flow:
      loss = -self.flow.log_prob(y, context=summaries).mean()
    elif self.hparams.loss_type == "log_prob":
      loss = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
    elif self.hparams.loss_type == "score":
      mean = mean * torch.tensor(THETA_STD[:2], device=mean.device) + torch.tensor(THETA_MEAN[:2], device=mean.device)
      std = std * torch.tensor(THETA_STD[:2], device=std.device)
      y = y[:, :2] * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)
      sq_error = (y - mean) ** 2
      score = -torch.sum(sq_error / std**2 + torch.log(std**2) + 1000.0 * sq_error, dim=1)
      loss = -torch.mean(score)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    mask = torch.tensor(self.mask, device=x.device).unsqueeze(0)

    # Multi-noise validation averaging for stable metrics
    total_mean = 0.0
    total_std = 0.0
    total_summaries = 0.0
    for _ in range(self.hparams.n_val_noise):
      x_noisy = (x + torch.randn_like(x) * NOISE_STD) * mask
      m, s, summ = self(x_noisy)
      total_mean += m
      total_std += s
      total_summaries += summ
    mean = total_mean / self.hparams.n_val_noise
    std = total_std / self.hparams.n_val_noise
    summaries = total_summaries / self.hparams.n_val_noise

    if self.hparams.use_flow:
      nll = -self.flow.log_prob(y, context=summaries).mean()
      self.log('val_nll', nll, prog_bar=True)
      return nll

    loss_val = -torch.distributions.Normal(loc=mean, scale=std).log_prob(y).mean()
    self.log('val_log_prob', loss_val, prog_bar=True)


    # Rescaling back to original parameters
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

    # Use fixed total_steps if provided, otherwise derive from trainer
    if self.hparams.total_steps > 0:
      total_steps = self.hparams.total_steps
    else:
      total_steps = int(self.trainer.estimated_stepping_batches)
    warmup_steps = self.hparams.warmup_steps

    # Linear warmup from 0 to max_lr
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-10,  # Start from nearly 0
        end_factor=1.0,      # End at max_lr
        total_iters=warmup_steps,
    )

    if self.hparams.lr_schedule == "cosine":
      main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, T_max=total_steps - warmup_steps
      )
    else:  # "step"
      # Calculate step size for StepLR in terms of steps (not epochs)
      steps_per_epoch = total_steps // self.trainer.max_epochs
      step_size_in_steps = self.hparams.decay_every_epochs * steps_per_epoch
      main_scheduler = torch.optim.lr_scheduler.StepLR(
          optimizer,
          step_size=step_size_in_steps,
          gamma=self.hparams.decay_rate,
      )

    # Combine warmup and main schedule
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, main_scheduler],
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
