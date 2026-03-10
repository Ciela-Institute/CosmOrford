import os
import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosmoford import THETA_MEAN, THETA_STD, SURVEY_MASK, NOISE_STD
from cosmoford.summaries import power_spectrum_batch
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b2, efficientnet_v2_s, efficientnet_v2_m
from torchvision.models.resnet import resnet18, ResNet18_Weights
from peft import LoraConfig, get_peft_model
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform, RandomPermutation


def build_flow(param_dim=2, context_dim=8, n_transforms=4, hidden_dim=64):
  """Build a small conditional MAF: p(params | summaries)."""
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
