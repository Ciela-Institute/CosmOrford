"""Plot FoM vs simulation budget from NPE results on Modal volume.

Usage:
    .venv/bin/modal run scripts/plot_fom_budget.py
"""
from pathlib import Path

import json

import modal

volume = modal.Volume.from_name("cosmoford-training", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "matplotlib")

app = modal.App("cosmoford-plot-fom", image=image)

VOLUME_PATH = Path("/experiments")
NPE_RESULTS_PATH = VOLUME_PATH / "npe_results"


@app.function(volumes={VOLUME_PATH: volume}, timeout=60)
def fetch_and_plot(output_path: str = "/tmp/fom_budget_scan.pdf") -> bytes:
    """Load results from volume, plot, and return the PDF bytes."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volume.reload()
    all_results = []
    if NPE_RESULTS_PATH.exists():
        for d in sorted(NPE_RESULTS_PATH.iterdir()):
            rfile = d / "results.json"
            if rfile.exists():
                all_results.append(json.loads(rfile.read_text()))

    if not all_results:
        raise RuntimeError("No results found on volume")

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

    return Path(output_path).read_bytes()


@app.local_entrypoint()
def main(output: str = "fom_budget_scan.pdf"):
    pdf_bytes = fetch_and_plot.remote()
    Path(output).write_bytes(pdf_bytes)
    print(f"Plot saved to {output}")
