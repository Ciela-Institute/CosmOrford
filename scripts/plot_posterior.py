"""Plot posterior samples for each budget.

Usage:
    python scripts/plot_posteriors.py \\
        --npe_results_path /path/to/npe_results \\
        [--output posteriors.pdf]
"""
from pathlib import Path
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIDUCIAL = np.array([0.29022, 0.81345])  # Omega_m, S_8


def plot_posteriors(npe_results_path: Path, output_path: str):
    all_results = []
    for d in sorted(npe_results_path.iterdir()):
        rfile = d / "results.json"
        samples_file = d / "posterior_samples.npy"
        if rfile.exists() and samples_file.exists():
            r = json.loads(rfile.read_text())
            r["samples"] = np.load(samples_file)  # (n_maps, n_samples, 2)
            all_results.append(r)

    if not all_results:
        raise RuntimeError(f"No results with posterior samples found in {npe_results_path}")

    all_results = sorted(all_results, key=lambda x: x["budget"])
    n = len(all_results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for i, r in enumerate(all_results):
        ax = axes[i]
        samples = r["samples"]  # (n_maps, n_samples, 2)
        n_maps = samples.shape[0]
                                                                                                                                                    
        for j in range(1, n_maps):                                                                                                                                                                               
            ax.scatter(samples[j, :, 0], samples[j, :, 1],
                        alpha=0.02, s=1, color="grey", rasterized=True)
        ax.scatter(samples[0, :, 0], samples[0, :, 1],
                   alpha=0.02, s=1, color="C0", rasterized=True)

        ax.axvline(FIDUCIAL[0], color="red", linewidth=1.5, linestyle="--")
        ax.axhline(FIDUCIAL[1], color="red", linewidth=1.5, linestyle="--")
        ax.plot(*FIDUCIAL, "r*", markersize=12, zorder=10)

        ax.set_xlabel(r"$\Omega_m$", fontsize=12)
        ax.set_ylabel(r"$S_8$", fontsize=12)
        ax.set_title(f"Budget = {r['budget']} (FoM={r['fom_mean']:.1f})", fontsize=12)
        ax.grid(True, alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Posterior samples at fiducial cosmology", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot posteriors per budget")
    parser.add_argument("--npe_results_path", required=True)
    parser.add_argument("--output", default="posteriors.pdf")
    args = parser.parse_args()

    plot_posteriors(Path(args.npe_results_path), args.output)
