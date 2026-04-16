"""Plot true vs predicted parameters for a trained compressor.

Produces two figures (validation set and holdout set), each with two subplots
(Omega_m and S_8). Noise is added only to the validation set (the holdout
dataset already contains noise).

Usage:
    python scripts/plot_compressor_regression.py \
        --checkpoint /path/to/compressor.ckpt \
        --validation_path /path/to/neurips-wl-challenge-flat \
        --holdout_path /path/to/neurips-wl-challenge-holdout \
        --output_dir /path/to/output \
        [--num_points 200]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk

from cosmoford import NOISE_STD, SURVEY_MASK, THETA_MEAN, THETA_STD
from cosmoford.dataset import reshape_field_numpy
from cosmoford.models_nopatch import RegressionModelNoPatch


def load_compressor(ckpt_path, device):
    compressor = RegressionModelNoPatch.load_from_checkpoint(ckpt_path, map_location=device)
    compressor.eval()
    compressor.to(device)
    for p in compressor.parameters():
        p.requires_grad = False
    return compressor


def predict(compressor, kappa_all, device, mask, add_noise):
    all_means = []
    all_stds = []

    with torch.no_grad():
        for i in range(len(kappa_all)):
            kappa_reshaped = reshape_field_numpy(kappa_all[i][np.newaxis])[0]

            if add_noise:
                noise = np.random.randn(*kappa_reshaped.shape).astype(np.float32) * NOISE_STD
                x_np = (kappa_reshaped + noise) * mask
            else:
                x_np = kappa_reshaped * mask

            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            mean, std, _ = compressor(x)
            all_means.append(mean.cpu().numpy())
            all_stds.append(std.cpu().numpy())

    all_means = np.array(all_means)[:, 0, :] * THETA_STD[:2] + THETA_MEAN[:2]
    all_stds = np.array(all_stds)[:, 0, :] * THETA_STD[:2]
    return all_means, all_stds


def plot_regression(true_theta, pred_means, pred_stds, title, output_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    labels = [r"$\Omega_m$", r"$S_8$"]
    for i, ax in enumerate(axs):
        ax.errorbar(true_theta[:, i], pred_means[:, i], yerr=pred_stds[:, i],
                     fmt=".", alpha=0.5, markersize=3)
        vmin = min(true_theta[:, i].min(), pred_means[:, i].min())
        vmax = max(true_theta[:, i].max(), pred_means[:, i].max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", label="Perfect")
        ax.set_xlabel(f"True {labels[i]}")
        ax.set_ylabel(f"Predicted {labels[i]}")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compressor regression plots (true vs predicted)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to compressor .ckpt")
    parser.add_argument("--validation_path", required=True,
                        help="Path to neurips-wl-challenge-flat (DatasetDict on disk)")
    parser.add_argument("--holdout_path", required=True,
                        help="Path to neurips-wl-challenge-holdout (DatasetDict on disk)")
    parser.add_argument("--output_dir", required=True, help="Directory for output figures")
    parser.add_argument("--num_points", type=int, default=200, help="Number of maps to evaluate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])
    compressor = load_compressor(args.checkpoint, device)

    # --- Validation set (add noise, data is clean) ---
    print("Loading validation set...")
    val_ds = load_from_disk(args.validation_path)
    val_ds = val_ds["validation"].with_format("numpy")
    n = min(args.num_points, len(val_ds))
    kappa_val = np.array(val_ds["kappa"][:n])
    theta_val = np.array(val_ds["theta"][:n])[:, :2]

    print(f"Running compressor on {n} validation maps (with noise)...")
    means_val, stds_val = predict(compressor, kappa_val, device, mask, add_noise=True)
    plot_regression(theta_val, means_val, stds_val,
                    "Validation set", output_dir / "regression_validation.png")

    # --- Holdout set (no noise, data already noisy) ---
    print("Loading holdout set...")
    holdout_ds = load_from_disk(args.holdout_path)
    holdout_ds = holdout_ds["train"].with_format("numpy")
    n = min(args.num_points, len(holdout_ds))
    kappa_holdout = np.array(holdout_ds["kappa"][:n])
    theta_holdout = np.array(holdout_ds["theta"][:n])[:, :2]

    print(f"Running compressor on {n} holdout maps (no noise added)...")
    means_ho, stds_ho = predict(compressor, kappa_holdout, device, mask, add_noise=False)
    plot_regression(theta_holdout, means_ho, stds_ho,
                    "Holdout set", output_dir / "regression_holdout.png")


if __name__ == "__main__":
    main()
