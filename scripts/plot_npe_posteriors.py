"""Plot 2D KDE posterior contours for fiducial holdout and validation maps.

Produces two figures:
  - npe_posteriors_fiducial.png: 4 fiducial holdout maps (no noise added)
  - npe_posteriors_validation.png: 4 validation maps (noise added)

Each figure is a 2x2 grid with 1/2-sigma KDE contours.

Usage:
    python scripts/plot_npe_posteriors.py \
        --compressor_checkpoint /path/to/compressor.ckpt \
        --npe_checkpoint /path/to/npe_results/budget-20200/npe_flow.pt \
        --holdout_path /path/to/neurips-wl-challenge-holdout \
        --validation_path /path/to/neurips-wl-challenge-flat \
        --output_dir /path/to/output \
        [--n_samples 10000]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import gaussian_kde

from cosmoford import NOISE_STD, SURVEY_MASK, THETA_MEAN, THETA_STD
from cosmoford.dataset import reshape_field_numpy
from cosmoford.models_nopatch import RegressionModelNoPatch, build_flow


def compress_maps(compressor, kappa_maps, mask, device, add_noise):
    summaries = []
    with torch.no_grad():
        for kappa_i in kappa_maps:
            kappa_reshaped = reshape_field_numpy(kappa_i[np.newaxis])[0]
            if add_noise:
                noise = np.random.randn(*kappa_reshaped.shape).astype(np.float32) * NOISE_STD
                x_np = (kappa_reshaped + noise) * mask
            else:
                x_np = kappa_reshaped * mask
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            summaries.append(compressor.compress(x))
    return summaries


def plot_kde_contours(ax, samples, true_om, true_s8):
    kde = gaussian_kde(samples.T)
    om_grid = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 200)
    s8_grid = np.linspace(samples[:, 1].min(), samples[:, 1].max(), 200)
    OM, S8 = np.meshgrid(om_grid, s8_grid)
    positions = np.vstack([OM.ravel(), S8.ravel()])
    Z = kde(positions).reshape(OM.shape)

    Z_sorted = np.sort(Z.ravel())[::-1]
    Z_cumsum = np.cumsum(Z_sorted) / Z_sorted.sum()
    level_1sigma = Z_sorted[np.searchsorted(Z_cumsum, 0.6827)]
    level_2sigma = Z_sorted[np.searchsorted(Z_cumsum, 0.9545)]

    ax.contourf(OM, S8, Z, levels=[level_2sigma, level_1sigma, Z.max()],
                colors=["#a6cee3", "#1f78b4"], alpha=0.7)
    ax.contour(OM, S8, Z, levels=[level_2sigma, level_1sigma],
               colors=["#1f78b4", "#08306b"], linewidths=[1, 1.5])

    ax.axvline(true_om, color="r", ls="--", lw=1, alpha=0.7)
    ax.axhline(true_s8, color="r", ls="--", lw=1, alpha=0.7)
    ax.plot(true_om, true_s8, "r+", ms=12, mew=2)
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$S_8$")


def plot_posterior_grid(flow, summaries, theta_all, indices, n_samples, device, title, output_path):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    with torch.no_grad():
        for idx, ax in zip(indices, axs.flat):
            s = summaries[idx]
            samples = flow.sample(n_samples, context=s)
            samples = samples.squeeze(0).cpu().numpy()
            samples_phys = samples * THETA_STD[:2] + THETA_MEAN[:2]

            true_om = theta_all[idx, 0]
            true_s8 = theta_all[idx, 1]

            plot_kde_contours(ax, samples_phys, true_om, true_s8)
            ax.set_title(f"Map {idx}")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D NPE posterior contours (KDE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--compressor_checkpoint", required=True, help="Path to compressor .ckpt")
    parser.add_argument("--npe_checkpoint", required=True, help="Path to NPE flow .pt")
    parser.add_argument("--holdout_path", required=True,
                        help="Path to neurips-wl-challenge-holdout (DatasetDict on disk)")
    parser.add_argument("--validation_path", required=True,
                        help="Path to neurips-wl-challenge-flat (DatasetDict on disk)")
    parser.add_argument("--output_dir", required=True, help="Directory for output figures")
    parser.add_argument("--n_samples", type=int, default=10_000, help="Posterior samples per map")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    # Load compressor
    compressor = RegressionModelNoPatch.load_from_checkpoint(
        args.compressor_checkpoint, map_location=device)
    compressor.eval()
    compressor.to(device)
    for p in compressor.parameters():
        p.requires_grad = False

    # Load NPE flow
    flow = build_flow(param_dim=2, context_dim=8).to(device)
    flow.load_state_dict(torch.load(args.npe_checkpoint, map_location=device, weights_only=False))
    flow.eval()

    # --- Fiducial holdout (no noise added) ---
    print("Loading fiducial holdout maps...")
    holdout = load_from_disk(args.holdout_path)
    fiducial = holdout["fiducial"].with_format("numpy")
    kappa_fid = np.array(fiducial["kappa"])
    theta_fid = np.array(fiducial["theta"])

    n_fid = min(4, len(kappa_fid))
    indices_fid = np.linspace(0, len(kappa_fid) - 1, n_fid, dtype=int)

    print(f"Compressing {n_fid} fiducial maps...")
    summaries_fid = compress_maps(compressor, kappa_fid, mask, device, add_noise=False)
    plot_posterior_grid(flow, summaries_fid, theta_fid, indices_fid, args.n_samples,
                        device, "NPE posteriors — fiducial holdout",
                        output_dir / "npe_posteriors_fiducial.png")

    # --- Validation set (noise added) ---
    print("Loading validation maps...")
    val_ds = load_from_disk(args.validation_path)
    val_ds = val_ds["validation"].with_format("numpy")
    n_val = min(4, len(val_ds))
    indices_val = np.linspace(0, len(val_ds) - 1, n_val, dtype=int)
    kappa_val = np.array(val_ds["kappa"])
    theta_val = np.array(val_ds["theta"])

    print(f"Compressing {n_val} validation maps (with noise)...")
    summaries_val = compress_maps(compressor, kappa_val, mask, device, add_noise=True)
    plot_posterior_grid(flow, summaries_val, theta_val, indices_val, args.n_samples,
                        device, "NPE posteriors — validation set",
                        output_dir / "npe_posteriors_validation.png")


if __name__ == "__main__":
    main()
