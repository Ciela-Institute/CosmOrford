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
        print(f"  budget={r['budget']:>6d}: FoM = {r['fom_mean']:.2f} ± {r['fom_std']:.2f} "
              f"(val_nll={r['best_val_nll']:.4f})")

    budgets = np.array([r["budget"] for r in all_results])
    fom_means = np.array([r["fom_mean"] for r in all_results])
    fom_stds = np.array([r["fom_std"] for r in all_results])

    order = np.argsort(budgets)
    budgets, fom_means, fom_stds = budgets[order], fom_means[order], fom_stds[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(budgets, fom_means, yerr=fom_stds,
                fmt="o-", color="C0", markersize=8, capsize=4, linewidth=1.5, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of compressor training simulations", fontsize=14)
    ax.set_ylabel("Figure of Merit (FoM)", fontsize=14)
    ax.set_title("Inference quality vs simulation budget", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)

    if ".pdf" in output_path: 
        output_path = output_path.split(".pdf")[0]
    
    # Saving data points
    np.savez(
        output_path + ".npz",
        budgets = budgets, 
        fom_means = fom_means, 
        fom_stds = fom_stds 
    )


# ── Modal entry point (only loaded when invoked via `modal run`) ──────────────
if __name__ != "__main__":
    import modal

    volume = modal.Volume.from_name("cosmoford-training", create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "matplotlib")
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

    parser = argparse.ArgumentParser(description="Plot FoM vs budget — local cluster mode")
    parser.add_argument(
        "--experiments_dir", required=True,
        help="Root experiments directory (must contain npe_results/)",
    )
    parser.add_argument(
        "--output", default="fom_budget_scan.pdf",
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