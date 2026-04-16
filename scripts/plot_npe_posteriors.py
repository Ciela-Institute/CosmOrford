"""Plot 2D posterior density for 4 fiducial holdout maps.

Loads a trained compressor and NPE flow, compresses 4 fiducial maps,
samples the posterior for each, and shows a 2x2 grid of 2D density plots
(Omega_m vs S_8) with the true fiducial values marked.

Usage:
    python scripts/plot_npe_posteriors.py \
        --compressor_checkpoint /path/to/compressor.ckpt \
        --npe_checkpoint /path/to/npe_results/budget-20200/npe_flow.pt \
        --holdout_path /path/to/neurips-wl-challenge-holdout \
        --output_dir /path/to/output \
        [--n_samples 10000]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk

from cosmoford import SURVEY_MASK, THETA_MEAN, THETA_STD
from cosmoford.dataset import reshape_field_numpy
from cosmoford.models_nopatch import RegressionModelNoPatch, build_flow


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D NPE posteriors on fiducial holdout maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--compressor_checkpoint", required=True, help="Path to compressor .ckpt")
    parser.add_argument("--npe_checkpoint", required=True, help="Path to NPE flow .pt")
    parser.add_argument("--holdout_path", required=True,
                        help="Path to neurips-wl-challenge-holdout (DatasetDict on disk)")
    parser.add_argument("--output_dir", required=True, help="Directory for output figure")
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

    # Load fiducial maps (already noisy)
    print("Loading fiducial holdout maps...")
    holdout = load_from_disk(args.holdout_path)
    fiducial = holdout["fiducial"].with_format("numpy")
    kappa_all = np.array(fiducial["kappa"])
    theta_all = np.array(fiducial["theta"])

    n_maps = min(4, len(kappa_all))
    indices = np.linspace(0, len(kappa_all) - 1, n_maps, dtype=int)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    with torch.no_grad():
        for idx, ax in zip(indices, axs.flat):
            kappa_reshaped = reshape_field_numpy(kappa_all[idx][np.newaxis])[0]
            x_np = kappa_reshaped * mask
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            s = compressor.compress(x)

            samples = flow.sample(args.n_samples, context=s)
            samples = samples.squeeze(0).cpu().numpy()
            samples_phys = samples * THETA_STD[:2] + THETA_MEAN[:2]

            true_om = theta_all[idx, 0]
            true_s8 = theta_all[idx, 1]

            ax.hist2d(samples_phys[:, 0], samples_phys[:, 1],
                      bins=60, cmap="Blues", density=True)
            ax.axvline(true_om, color="r", ls="--", lw=1.5)
            ax.axhline(true_s8, color="r", ls="--", lw=1.5)
            ax.plot(true_om, true_s8, "r+", ms=12, mew=2)
            ax.set_xlabel(r"$\Omega_m$")
            ax.set_ylabel(r"$S_8$")
            ax.set_title(f"Map {idx}")

    fig.suptitle("NPE posteriors on fiducial holdout maps", fontsize=14)
    fig.tight_layout()
    out_path = output_dir / "npe_posteriors_fiducial.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
