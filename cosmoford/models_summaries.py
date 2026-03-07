import os
import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from s8ball import THETA_MEAN, THETA_STD, SURVEY_MASK, NOISE_STD
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b2, efficientnet_v2_s, efficientnet_v2_m
from nflows import transforms, distributions, flows
from typing import Optional
import matplotlib.pyplot as plt
import wandb

class RegressionModel(L.LightningModule):

  def __init__(self, backbone="efficientnet_b0", lr: float = 5e-3, milestone_interval: int = 1, gamma=0.75, neural_compression=False):
    super().__init__()
    self.save_hyperparameters()
    self.milestones = [milestone_interval * i for i in range(1, 6)]
    self.mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    self.neural_compression = neural_compression

    if self.neural_compression:
      self.num_targets = 1 # outputting only the mean
      self.loss_fn = self._mse_loss # only MSE loss for neural compression for now
    else:
      self.num_targets = 2  # outputting mean or mean and log-variance
      self.loss_fn = self._nll_loss

    last_dim = 1280  # For efficientnet_b0
    if backbone == "efficientnet_b0":
      vision_model = efficientnet_b0()
    elif backbone == "efficientnet_b2":
      vision_model = efficientnet_b2()
      last_dim = 1408
    elif backbone == "efficientnet_v2_s":
      vision_model = efficientnet_v2_s()
    else:
      raise ValueError(f"Backbone {backbone} not supported.")
    
    self.model = vision_model.features
    self.head = nn.Sequential(
     nn.AdaptiveAvgPool2d(1),
     nn.Flatten(),
     nn.Linear(last_dim, 128),
     nn.LeakyReLU(),
     nn.Linear(128, 5*self.num_targets) 
    )

  def forward(self, x):
    # Add channel dimension if missing
    if x.dim() == 3:
      x = x.unsqueeze(1)
    # Repeat channels to match expected input size (e.g., 3 for RGB)
    if x.size(1) == 1:
      x = x.repeat(1, 3, 1, 1)
    features = self.model(x.float())
    x = self.head(features)
    if self.num_targets == 1:
      return x[..., :5] # Return mean only
    else:
      return x[..., :5], F.softplus(x[..., 5:]+0.001)  # Return mean and scale

  def training_step(self, batch, batch_idx):
    x, y = batch

    # Adding noise to the input convergence maps and applying mask
    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

    out = self(x)
    if self.num_targets == 1: # neural compression
      mean = out
      std = torch.ones_like(mean)
    else: # direct regression
      mean, std = out

    loss = self.loss_fn(mean, std, y)
    loss = loss.mean()

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch

    # Adding noise to the input convergence maps and applying mask
    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

    out = self(x)
    if self.num_targets == 1: # neural compression
      mean = out
      std = torch.ones_like(mean)
    else: # direct regression
      mean, std = out

    # Rescaling back to original parameters
    mean = mean * torch.tensor(THETA_STD, device=mean.device) + torch.tensor(THETA_MEAN, device=mean.device)
    std = std * torch.tensor(THETA_STD, device=std.device)
    y = y * torch.tensor(THETA_STD, device=y.device) + torch.tensor(THETA_MEAN, device=y.device)

    # Only evaluating the results on the cosmological parameters
    mean = mean[:, :2]
    std = std[:, :2]
    y = y[:, :2]

    loss = self.loss_fn(mean, std, y)
    loss = loss.mean()

    self.log('val_loss', loss, prog_bar=True)

    if not self.neural_compression:
      # Compute the Phase 1 score (torch version)
      sq_error = (y - mean) ** 2
      scale_factor = 1000.0
      score = -torch.sum(sq_error / std**2 + torch.log(std**2) + scale_factor * sq_error, dim=1)
      score = torch.mean(score)

      # Compute the MSE as well for monitoring
      mse = F.mse_loss(mean, y)

      self.log('val_score', score, prog_bar=True)
      self.log('val_mse', mse, prog_bar=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.hparams.gamma)                
    return [optimizer], [scheduler]
  
  def _mse_loss(self, mean, std, y):
    return nn.MSELoss(reduction="none")(mean, y)

  def _nll_loss(self, mean, std, y):
    return - torch.distributions.Normal(loc=mean, scale=std).log_prob(y)


class AugmentedRegressionModel(L.LightningModule):

  def __init__(self, backbone="efficientnet_b0", warmup_steps: int = 1000, max_lr: float = 0.256,
               decay_rate: float = 0.97, decay_every_epochs: int = 2, neural_compression=False):
    super().__init__()
    self.save_hyperparameters()
    self.mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    self.neural_compression = neural_compression

    if self.neural_compression:
      self.num_targets = 1 # outputting only the mean
      self.loss_fn = self._mse_loss # only MSE loss for neural compression for now
    else:
      self.num_targets = 2  # outputting mean or mean and log-variance
      self.loss_fn = self._nll_loss

    last_dim = 1280  # For efficientnet_b0
    if backbone == "efficientnet_b0":
      vision_model = efficientnet_b0()
    elif backbone == "efficientnet_b2":
      vision_model = efficientnet_b2()
      last_dim = 1408
    elif backbone == "efficientnet_v2_s":
      vision_model = efficientnet_v2_s()
    else:
      raise ValueError(f"Backbone {backbone} not supported.")
    
    self.model = vision_model.features
    self.head = nn.Sequential(
     nn.AdaptiveAvgPool2d(1),
     nn.Flatten(),
     nn.Linear(last_dim, 128),
     nn.LeakyReLU(),
     nn.Linear(128, 5*self.num_targets) 
    )

  def forward(self, x):
    # Add channel dimension if missing
    if x.dim() == 3:
      x = x.unsqueeze(1)
    # Repeat channels to match expected input size (e.g., 3 for RGB)
    if x.size(1) == 1:
      x = x.repeat(1, 3, 1, 1)
    features = self.model(x.float())
    x = self.head(features)
    if self.num_targets == 1:
      return x[..., :5] # Return mean only
    else:
      return x[..., :5], F.softplus(x[..., 5:])+0.001  # Return mean and scale

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
    # Adding random cyclic shifts (different for each sample)
    shift_h = torch.randint(low=0, high=x.size(1), size=(batch_size,), device=x.device)
    # shift along height dimension
    idx = torch.arange(x.size(1), device=x.device).unsqueeze(0).repeat(batch_size, 1)
    idx = (idx - shift_h.unsqueeze(1)) % x.size(1)
    x = x.gather(1, idx.unsqueeze(2).expand(-1, -1, x.size(2)))

    out = self(x)
    if self.num_targets == 1: # neural compression
      mean = out
      std = torch.ones_like(mean)
    else: # direct regression
      mean, std = out

    loss = self.loss_fn(mean, std, y)
    loss = loss.mean()

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch

    # Adding noise to the input convergence maps and applying mask
    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

    out = self(x)
    if self.num_targets == 1: # neural compression
      mean = out
      std = torch.ones_like(mean)
    else: # direct regression
      mean, std = out

    # Rescaling back to original parameters
    mean = mean * torch.tensor(THETA_STD, device=mean.device) + torch.tensor(THETA_MEAN, device=mean.device)
    std = std * torch.tensor(THETA_STD, device=std.device)
    y = y * torch.tensor(THETA_STD, device=y.device) + torch.tensor(THETA_MEAN, device=y.device)

    # Only evaluating the results on the cosmological parameters
    mean = mean[:, :2]
    std = std[:, :2]
    y = y[:, :2]

    loss = self.loss_fn(mean, std, y)
    loss = loss.mean()

    self.log('val_loss', loss, prog_bar=True)

    if not self.neural_compression:
      # Compute the Phase 1 score (torch version)
      sq_error = (y - mean) ** 2
      scale_factor = 1000.0
      score = -torch.sum(sq_error / std**2 + torch.log(std**2) + scale_factor * sq_error, dim=1)
      score = torch.mean(score)

      # Compute the MSE as well for monitoring
      mse = F.mse_loss(mean, y)

      self.log('val_score', score, prog_bar=True)
      self.log('val_mse', mse, prog_bar=True)
    return loss

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
  
  def _mse_loss(self, mean, std, y):
    return nn.MSELoss(reduction="none")(mean, y)

  def _nll_loss(self, mean, std, y):
    return - torch.distributions.Normal(loc=mean, scale=std).log_prob(y)


class ProbabilisticModel(L.LightningModule):

  def __init__(
      self,
      backbone="efficientnet_b0",
      lr: float = 5e-3,
      milestone_interval: int = 1,
      gamma=0.75,
      summary_statistics=None,
      run_id: str = None,
      summary_stat_dim: int = 8,
      augmentation: bool = False,
      local_checkpoint_path: str = None,
      fancy_opti: bool = False,
      warmup_steps: int = 1000,
      max_lr: float = 0.256,
      decay_rate: float = 0.97,
      dropout_rate=0.1,
      decay_every_epochs: int = 2
  ):
    super().__init__()
    self.save_hyperparameters()
    self.mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    self.summary_statistics = summary_statistics

    if self.summary_statistics in ["mse", "vmim"]:
      print(f"Using neural compression with {self.summary_statistics} summary statistics.")
      if self.summary_statistics == "mse":
        # For MSE, we use RegressionModel directly
        self.compressor_fn = self._download_model(run_id, model_class="regression", local_checkpoint_path=local_checkpoint_path)[0].eval()
        self.context_features = self.compressor_fn.num_targets * 5
      else:  # vmim
        # For VMIM, we only want the neural network part of ProbabilisticModel
        model = self._download_model(run_id, model_class="probabilistic", local_checkpoint_path=local_checkpoint_path)[0].eval()
        # Extract only the feature extractor part
        self.compressor_fn = nn.Module()
        self.compressor_fn.model = model.model  # vision_model.features
        self.compressor_fn.head = model.head # the head producing summary statistics
        self.compressor_fn.forward = lambda x: model.forward(x)  # Use the same forward method
        self.context_features = model.context_features
    elif self.summary_statistics == None:
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
      
      # context features defined by summary_stat_dim parameter
      self.context_features = summary_stat_dim
      
      self.model = vision_model.features
      self.reshape_head = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Dropout(p=self.hparams.dropout_rate, inplace=True)
      )
      self.head = nn.Sequential(
      nn.Linear(last_dim, 128),
      nn.LeakyReLU(),
      nn.Linear(128, self.context_features) # Outputing mean and log-variance
      )
    else:
      raise ValueError(f"Summary statistics method {self.summary_statistics} not supported.") # could add PS or other manually crafted summary statistics

    # Define the flow
    flow_transform = transforms.CompositeTransform([
      transforms.MaskedAffineAutoregressiveTransform(features=2, hidden_features=128, context_features=self.context_features),
      transforms.ReversePermutation(features=2),
      transforms.MaskedAffineAutoregressiveTransform(features=2, hidden_features=128, context_features=self.context_features),
      transforms.ReversePermutation(features=2),
      transforms.MaskedAffineAutoregressiveTransform(features=2, hidden_features=128, context_features=self.context_features),
      transforms.ReversePermutation(features=2),
      transforms.MaskedAffineAutoregressiveTransform(features=2, hidden_features=128, context_features=self.context_features),
    ])
    self.flow = flows.Flow(transform=flow_transform, distribution=distributions.StandardNormal([2]))

  def forward(self, x):
    # Add channel dimension if missing
    if x.dim() == 3:
      x = x.unsqueeze(1)
    # Repeat channels to match expected input size (e.g., 3 for RGB)
    if x.size(1) == 1:
      x = x.repeat(1, 3, 1, 1)

    if self.summary_statistics == None:
      features = self.model(x.float())
      features = self.reshape_head(features)
      s = self.head(features) # Sufficient statistics
    else: # neural compression
      with torch.no_grad(): # avoid gradients through the compressor ? 
        s = self.compressor_fn(x.float())
    return s

  def training_step(self, batch, batch_idx):
    x, y = batch

    # Adding noise to the input convergence maps and applying mask
    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

    if self.hparams.augmentation == True: 
      print('Using augmentations in ProbabilisticModel training step.')
      # Adding augmentations, random left-right and up-down flips (per sample)
      batch_size = x.size(0)
      # Random flips along nx dimension (dim=1)
      flip_lr = torch.rand(batch_size, device=x.device) < 0.5
      x[flip_lr] = torch.flip(x[flip_lr], dims=[1])
      # Random flips along ny dimension (dim=2)
      flip_ud = torch.rand(batch_size, device=x.device) < 0.5
      x[flip_ud] = torch.flip(x[flip_ud], dims=[2])
      # Adding random cyclic shifts (different for each sample)
      shift_h = torch.randint(low=0, high=x.size(1), size=(batch_size,), device=x.device)
      # shift along height dimension
      idx = torch.arange(x.size(1), device=x.device).unsqueeze(0).repeat(batch_size, 1)
      idx = (idx - shift_h.unsqueeze(1)) % x.size(1)
      x = x.gather(1, idx.unsqueeze(2).expand(-1, -1, x.size(2)))

    print('x shape: ', x.shape)
    s = self(x)
    print('s shape: ', s.shape)

    log_prob = self.flow.log_prob(inputs=y.float()[:, :2], context=s)
    loss = - log_prob.mean()

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y = y.float()  # Ensure y is float32

    # Adding noise to the input convergence maps and applying mask
    noise = torch.randn_like(x) * NOISE_STD
    x = x + noise
    x = x * torch.tensor(self.mask, device=x.device).unsqueeze(0)

    s = self(x)

    log_prob = self.flow.log_prob(inputs=y.float()[:, :2], context=s)
    loss = - log_prob.mean()

    # Using the flow to sample 100 points per entries and compute their probability
    # Keep the maximum probablity point as our point estimate, and the std of the samples
    # as our uncertainty estimate
    samples, log_probs = self.flow.sample_and_log_prob(50000, context=s)  # [batch_size, 100, 2], [batch_size, 100]

    samples = samples.cpu().detach().numpy()
    log_probs = log_probs.cpu().detach().numpy()

    idx = log_probs.argmax(axis=1)
    y_hat = samples[np.arange(samples.shape[0]), idx, :]  # [batch_size, 2]
    std = samples.std(axis=1)    # [batch_size, 2]

    # Rescaling back to original parameters
    y_hat = y_hat * THETA_STD[:2] + THETA_MEAN[:2]
    std = std * THETA_STD[:2]
    y = y[:, :2].cpu().detach().numpy() * THETA_STD[:2] + THETA_MEAN[:2]

    # Compute the Phase 1 score (torch version)
    sq_error = (y - y_hat) ** 2
    scale_factor = 1000.0
    score = -np.sum(sq_error / std**2 + np.log(std**2) + scale_factor * sq_error, axis=1)
    score = np.mean(score)

    # Compute mse for logging
    mse = np.mean((y - y_hat) ** 2)

    self.log('val_score', score, prog_bar=True)
    self.log('val_mse', mse, prog_bar=True)
    self.log('val_loss', loss, prog_bar=True)
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
    
  def _download_model(
    self,
    run_id: str = None,
    project: str = "neurips-wl-challenge",
    entity: str = "cosmostat",
    model_class: str = "regression", 
    local_checkpoint_path: str = None
):
    if local_checkpoint_path is not None:
        checkpoint_path = local_checkpoint_path
    else:
        from wandb.apis.public import Api

        print(f"Downloading model from W&B for run {run_id} (no W&B run created)...")
        api = Api()
        art = api.artifact(f"{entity}/{project}/model-{run_id}:latest", type="model")
        artifact_dir = art.download()
        checkpoint_path = f"{artifact_dir}/model.ckpt"
    
    if model_class == "regression":
        model = RegressionModel.load_from_checkpoint(checkpoint_path).eval()
    elif model_class == "probabilistic":
        model = ProbabilisticModel.load_from_checkpoint(checkpoint_path).eval()
    else:
        raise ValueError(f"Model class {model_class} not supported.")
    
    print(f"Model loaded from {checkpoint_path}")
    return model, None  # keep signature if you rely on it elsewhere
