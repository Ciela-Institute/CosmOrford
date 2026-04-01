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
    val_nlls = np.array([r["best_val_nll"] for r in all_results])                                                                                                                                                
    mse_means = np.array([r["mse_mean"] for r in all_results])                                                                                                                                                   
    mse_stds = np.array([r["mse_std"] for r in all_results]) 

    order = np.argsort(budgets)
    budgets = budgets[order]                                                                                                                                                                                     
    fom_means, fom_stds = fom_means[order], fom_stds[order]                                                                                                                                                      
    val_nlls = val_nlls[order]                                                                                                                                                                                   
    mse_means, mse_stds = mse_means[order], mse_stds[order] 

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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))                                                                                                                                                                                                                                                                                                                                                                
    axes[0].errorbar(budgets, fom_means, yerr=fom_stds,                                                                                                                                                          
                        fmt="o-", color="C0", markersize=8, capsize=4, linewidth=1.5)                                                                                                                               
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
                                                                                                                                                                                                            
    axes[2].errorbar(budgets, mse_means, yerr=mse_stds,                                                                                                                                                          
                        fmt="o-", color="C2", markersize=8, capsize=4, linewidth=1.5)                                                                                                                               
    axes[2].set_xscale("log")                                                                                                                                                                                    
    axes[2].set_xlabel("Simulation budget", fontsize=13)                                                                                                                                                         
    axes[2].set_ylabel("MSE", fontsize=13)                                                                                                                                                                       
    axes[2].set_title("Posterior MSE vs budget", fontsize=14)                                                                                                                                                    
    axes[2].grid(True, alpha=0.3)                                                                                                                                                                                
                                                                                                                                                                                                            
    for ax in axes:                                                                                                                                                                                              
        ax.tick_params(labelsize=11)  

    fig.tight_layout()
    fig.savefig(output_path.replace(".pdf", "_full.pdf"), dpi=150)
    print(f"Plot saved to {output_path} and {output_path.replace('.pdf', '_full.pdf')}")                                                                                                                                                                      
                                


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