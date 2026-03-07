import torch
import lightning as L
from datasets import load_dataset, concatenate_datasets, Dataset
from torch.utils.data import DataLoader
from cosmoford import THETA_MEAN, THETA_STD
import numpy as np

__all__ = [
    "ChallengeDataModule",
    "reshape_field",
    "inverse_reshape_field",
    "reshape_field_numpy",
    "inverse_reshape_field_numpy",
]

def reshape_field(kappa):
  """ This function reshapes the field to remove most of the empty space.

  This turns the field from [1424, 176] to [1834,88]

  Note: this is not ideal, and far from the most optimal way to process the data
  but to first order it provides an easy way to get rid of most of the masks,
  at the cost of breaking some large scale modes.
  """
  return torch.concat([kappa[:, :, :88], kappa[:, 620:1030, 88:]], dim=1)

def reshape_field_numpy(kappa):
    """NumPy equivalent of reshape_field.

    Expects kappa with shape (B, 1424, 176) and returns (B, 1834, 88).
    """
    return np.concatenate([kappa[:, :, :88], kappa[:, 620:1030, 88:]], axis=1)

def inverse_reshape_field(kappa_reduced: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:

    B, H_red, W_red = kappa_reduced.shape
    # Split the reduced tensor into its two original parts
    part1 = kappa_reduced[:, :1424, :]      # corresponds to kappa[:, :, :88]
    part2 = kappa_reduced[:, 1424:, :]      # corresponds to kappa[:, 620:1030, 88:]

    # Allocate full-sized output and fill missing regions
    kappa_full = torch.full(
        (B, 1424, 176),
        fill_value,
        dtype=kappa_reduced.dtype,
        device=kappa_reduced.device,
    )

    # Restore left half (columns 0:88)
    kappa_full[:, :, :88] = part1

    # Restore the kept block of the right half (columns 88:176, rows 620:1030)
    kappa_full[:, 620:1030, 88:] = part2

    return kappa_full

def inverse_reshape_field_numpy(kappa_reduced, fill_value: float = 0.0):
    """NumPy equivalent of inverse_reshape_field.

    Expects kappa_reduced with shape (B, 1834, 88) and returns (B, 1424, 176).
    """
    B, H_red, W_red = kappa_reduced.shape
    part1 = kappa_reduced[:, :1424, :]
    part2 = kappa_reduced[:, 1424:, :]
    kappa_full = np.full((B, 1424, 176), fill_value, dtype=kappa_reduced.dtype)
    kappa_full[:, :, :88] = part1
    kappa_full[:, 620:1030, 88:] = part2
    return kappa_full


class ChallengeDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=8, train_on_full_data=False, dataset_mode="train",
                 max_train_samples: int = 0):
        """
        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_on_full_data: Legacy parameter - if True, uses train+validation for training
            dataset_mode: Which dataset to use for training. Options:
                - "lognormal": Use lognormal pretraining dataset
                - "train": Use the regular training set from neurips-wl-challenge-flat
                - "full": Use train + validation concatenated
            max_train_samples: If > 0, limit training set to this many samples
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_on_full_data = train_on_full_data
        self.dataset_mode = dataset_mode
        self.max_train_samples = max_train_samples

    def _collate_fn(self, batch):
        # Run the default collate function
        batch = torch.utils.data.dataloader.default_collate(batch)
        # Reshape the kappa fields to remove most of the empty space
        kappa = reshape_field(batch['kappa'])
        # Normalizing data with mean and std computed on the training set
        device = batch['theta'].device
        theta = (batch['theta'][:, :2] - torch.tensor(THETA_MEAN[:2], device=device)) / torch.tensor(THETA_STD[:2], device=device)
        # Convert to float32 to match model output dtype
        kappa = kappa.float()
        theta = theta.float()
        return kappa, theta

    def setup(self, stage=None):
        # Load the main dataset
        dset = load_dataset("cosmostat/neurips-wl-challenge-flat")
        dset = dset.with_format("torch")

        # Determine which dataset to use for training
        # Legacy: train_on_full_data overrides dataset_mode if True
        if self.train_on_full_data:
            self.train_dataset = concatenate_datasets([dset['train'], dset['validation']])
            self.val_dataset = dset['validation']

        elif self.dataset_mode == "gowerstreet":
            # Load gowerstreet pretraining dataset
            print("Loading Gower Street pretraining dataset...")
            dset_gowerstreet = Dataset.load_from_disk("gs://neurips-wl/datasets/gowerstreet_patches")
            dset_gowerstreet = dset_gowerstreet.shuffle(seed=42)
            dset_gowerstreet = dset_gowerstreet.with_format("torch")
            self.train_dataset = dset_gowerstreet
            self.val_dataset = dset['validation']

        elif self.dataset_mode == "gowerstreet-train":
            # Load gowerstreet pretraining dataset
            print("Loading Gower Street pretraining dataset...")
            dset_gowerstreet = Dataset.load_from_disk("gs://neurips-wl/datasets/gowerstreet_patches")
            dset_gowerstreet = dset_gowerstreet.shuffle(seed=42)
            dset_gowerstreet = dset_gowerstreet.with_format("torch")

            # Load regular training set
            train_dataset = dset['train']

            # Mix gower street and regular training set
            self.train_dataset = concatenate_datasets([dset_gowerstreet, train_dataset]).shuffle(seed=42)
            self.val_dataset = dset['validation']

        elif self.dataset_mode == "lognormal":
            # Load lognormal pretraining dataset
            dset_lognormal = Dataset.load_from_disk("gs://neurips-wl/datasets/lognormal")
            dset_lognormal = dset_lognormal.shuffle(seed=42)
            dset_lognormal = dset_lognormal.with_format("torch")
            self.train_dataset = dset_lognormal
            self.val_dataset = dset['validation']
        elif self.dataset_mode == "ot_emulated":
            # Load lognormal pretraining dataset
            dset_ot = Dataset.load_from_disk("gs://neurips-wl/datasets/ot_emulated")
            dset_ot = dset_ot.rename_column('maps', 'kappa')
            dset_ot = dset_ot.shuffle(seed=42)
            dset_ot = dset_ot.with_format("torch")
            self.train_dataset = dset_ot
            self.val_dataset = dset['validation']
        elif self.dataset_mode == "train":
            # Use regular training set
            self.train_dataset = dset['train']
            self.val_dataset = dset['validation']
        elif self.dataset_mode == "full":
            # Use train + validation concatenated
            self.train_dataset = concatenate_datasets([dset['train'], dset['validation']])
            self.val_dataset = dset['validation']
        else:
            raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}. Must be 'lognormal', 'train', or 'full'.")

        # Limit training set size if requested
        if self.max_train_samples > 0 and len(self.train_dataset) > self.max_train_samples:
            full_size = len(self.train_dataset)
            self.train_dataset = self.train_dataset.select(range(self.max_train_samples))
            print(f"Training set limited to {self.max_train_samples} samples (from {full_size})")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            drop_last=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            drop_last=True
            )

    def test_dataloader(self):
        # For this challenge, we'll use the validation set as a proxy for the test set
        return self.val_dataloader()
