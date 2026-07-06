"""Plot FoM vs simulation budget from NPE results.

Usage (Modal):
    .venv/bin/modal run scripts/plot_fom_budget.py

Usage (local):
    python scripts/plot_fom_budget.py \\
        --experiments_dir /path/to/experiments \\
        [--output fom_budget_scan.pdf]
"""

from pathlib import Path

import json


def _plot_core(npe_results_path: Path, output_path: str):
    """Load results from disk, plot, and save to output_path."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_results = []
    if npe_results_path.exists():
        for d in sorted(npe_results_path.iterdir()):
            rfile = d / "results.json"
            if rfile.exists():
                all_results.append(json.loads(rfile.read_text()))

    if not all_results:
        raise RuntimeError(f"No results found in {npe_results_path}")

    for r in sorted(all_results, key=lambda x: x["budget"]):
        print(
            f"  budget={r['budget']:>6d}: FoM = {r['fom_mean']:.2f} ± {r['fom_std']:.2f} "
            f"(val_nll={r['best_val_nll']:.4f})"
        )

    def _mira(r):
        """Prefer the flat_val MIRA score (proper held-out test); fall back to npe_val."""
        calib = r.get("calibration", {})
        for tag in ("flat_val", "npe_val"):
            if tag in calib:
                return calib[tag]["mira_mean"], calib[tag]["mira_std"]
        return np.nan, np.nan

    budgets = np.array([r["budget"] for r in all_results])
    fom_means = np.array([r["fom_mean"] for r in all_results])
    fom_stds = np.array([r["fom_std"] for r in all_results])
    val_nlls = np.array([r["best_val_nll"] for r in all_results])
    mse_means = np.array([r.get("mse_mean", np.nan) for r in all_results])
    mse_stds = np.array([r.get("mse_std", np.nan) for r in all_results])
    score_means = np.array([r.get("score_mean", np.nan) for r in all_results])
    score_stds = np.array([r.get("score_std", np.nan) for r in all_results])
    mse_nf_means = np.array([r.get("mse_nf_mean", np.nan) for r in all_results])
    mse_nf_stds = np.array([r.get("mse_nf_std", np.nan) for r in all_results])
    score_nf_means = np.array([r.get("score_nf_mean", np.nan) for r in all_results])
    score_nf_stds = np.array([r.get("score_nf_std", np.nan) for r in all_results])
    mira_pairs = [_mira(r) for r in all_results]
    mira_means = np.array([m for m, _ in mira_pairs])
    mira_stds = np.array([s for _, s in mira_pairs])

    order = np.argsort(budgets)
    budgets = budgets[order]
    fom_means, fom_stds = fom_means[order], fom_stds[order]
    val_nlls = val_nlls[order]
    mse_means, mse_stds = mse_means[order], mse_stds[order]
    score_means, score_stds = score_means[order], score_stds[order]
    mse_nf_means, mse_nf_stds = mse_nf_means[order], mse_nf_stds[order]
    score_nf_means, score_nf_stds = score_nf_means[order], score_nf_stds[order]
    mira_means, mira_stds = mira_means[order], mira_stds[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        budgets,
        fom_means,
        yerr=fom_stds,
        fmt="o-",
        color="C0",
        markersize=8,
        capsize=4,
        linewidth=1.5,
        zorder=5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of compressor training simulations", fontsize=14)
    ax.set_ylabel("Figure of Merit (FoM)", fontsize=14)
    ax.set_title("Inference quality vs simulation budget", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes = axes.flatten()
    axes[0].errorbar(
        budgets,
        fom_means,
        yerr=fom_stds,
        fmt="o-",
        color="C0",
        markersize=8,
        capsize=4,
        linewidth=1.5,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Simulation budget", fontsize=13)
    axes[0].set_ylabel("Figure of Merit (FoM)", fontsize=13)
    axes[0].set_title("FoM vs budget", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(budgets, val_nlls, "o-", color="C1", markersize=8, linewidth=1.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Simulation budget", fontsize=13)
    axes[1].set_ylabel("Best val NLL", fontsize=13)
    axes[1].set_title("NPE val NLL vs budget", fontsize=14)
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(
        budgets,
        mse_means,
        yerr=mse_stds,
        fmt="o-",
        color="C2",
        markersize=8,
        capsize=4,
        linewidth=1.5,
    )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Simulation budget", fontsize=13)
    axes[2].set_ylabel("MSE", fontsize=13)
    axes[2].set_title("Compressor MSE vs budget", fontsize=14)
    axes[2].grid(True, alpha=0.3)

    axes[3].errorbar(
        budgets,
        score_means,
        yerr=score_stds,
        fmt="o-",
        color="C3",
        markersize=8,
        capsize=4,
        linewidth=1.5,
    )
    axes[3].set_xscale("log")
    axes[3].set_xlabel("Simulation budget", fontsize=13)
    axes[3].set_ylabel("Score", fontsize=13)
    axes[3].set_title("Compressor score vs budget", fontsize=14)
    axes[3].grid(True, alpha=0.3)

    axes[4].errorbar(
        budgets,
        mira_means,
        yerr=mira_stds,
        fmt="o-",
        color="C4",
        markersize=8,
        capsize=4,
        linewidth=1.5,
    )
    axes[4].set_xscale("log")
    axes[4].set_xlabel("Simulation budget", fontsize=13)
    axes[4].set_ylabel("MIRA score", fontsize=13)
    axes[4].set_title("MIRA calibration vs budget", fontsize=14)
    axes[4].grid(True, alpha=0.3)

    axes[5].errorbar(
        budgets,
        mse_nf_means,
        yerr=mse_nf_stds,
        fmt="o-",
        color="C5",
        markersize=8,
        capsize=4,
        linewidth=1.5,
    )
    axes[5].set_xscale("log")
    axes[5].set_xlabel("Simulation budget", fontsize=13)
    axes[5].set_ylabel("MSE (NF)", fontsize=13)
    axes[5].set_title("NF posterior MSE vs budget", fontsize=14)
    axes[5].grid(True, alpha=0.3)

    axes[6].errorbar(
        budgets,
        score_nf_means,
        yerr=score_nf_stds,
        fmt="o-",
        color="C6",
        markersize=8,
        capsize=4,
        linewidth=1.5,
    )
    axes[6].set_xscale("log")
    axes[6].set_xlabel("Simulation budget", fontsize=13)
    axes[6].set_ylabel("Score (NF)", fontsize=13)
    axes[6].set_title("NF posterior score vs budget", fontsize=14)
    axes[6].grid(True, alpha=0.3)

    axes[7].axis("off")

    for ax in axes:
        ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(output_path.replace(".pdf", "_full.pdf"), dpi=150)
    print(f"Plot saved to {output_path} and {output_path.replace('.pdf', '_full.pdf')}")

    # Saving data points for the plots above.
    if ".pdf" in output_path:
        output_path = output_path.split(".pdf")[0]

    np.savez(
        output_path + ".npz",
        budgets=budgets,
        fom_means=fom_means,
        fom_stds=fom_stds,
        mse_means=mse_means,
        mse_stds=mse_stds,
        score_means=score_means,
        score_stds=score_stds,
        mse_nf_means=mse_nf_means,
        mse_nf_stds=mse_nf_stds,
        score_nf_means=score_nf_means,
        score_nf_stds=score_nf_stds,
        mira_means=mira_means,
        mira_stds=mira_stds,
        val_nlls=val_nlls,
    )


# ── Modal entry point (only loaded when invoked via `modal run`) ──────────────
if __name__ != "__main__":
    import modal

    volume = modal.Volume.from_name("cosmoford-training", create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "numpy", "matplotlib"
    )
    app = modal.App("cosmoford-plot-fom", image=image)

    VOLUME_PATH = Path("/experiments")
    NPE_RESULTS_PATH = VOLUME_PATH / "npe_results"

    @app.function(volumes={VOLUME_PATH: volume}, timeout=60)
    def fetch_and_plot(output_path: str = "/tmp/fom_budget_scan.pdf") -> bytes:
        volume.reload()
        _plot_core(NPE_RESULTS_PATH, output_path)
        return Path(output_path).read_bytes()

    @app.local_entrypoint()
    def main(output: str = "fom_budget_scan.pdf"):
        pdf_bytes = fetch_and_plot.remote()
        Path(output).write_bytes(pdf_bytes)
        print(f"Plot saved to {output}")


# ── Local entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot FoM vs budget — local cluster mode"
    )
    parser.add_argument(
        "--experiments_dir",
        required=True,
        help="Root experiments directory (must contain npe_results/)",
    )
    parser.add_argument(
        "--output",
        default="fom_budget_scan.pdf",
        help="Output PDF path (default: fom_budget_scan.pdf)",
    )
    args = parser.parse_args()

    _plot_core(Path(args.experiments_dir) / "npe_results", args.output)
    print(f"Plot saved to {args.output}")

    if ".pdf" in args.output:
        data_path = args.output.split(".pdf")[0]
    else:
        data_path = args.output
    print(f"Data points saved to {data_path}.npz")
