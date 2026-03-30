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


def _collect_results(root: Path):
    all_results = []
    if root.exists():
        for d in sorted(root.iterdir()):
            rfile = d / "results.json"
            if rfile.exists():
                rec = json.loads(rfile.read_text())
                rec["_run_group"] = root.name
                rec["_budget_dir"] = d.name
                all_results.append(rec)
    return all_results


def _plot_core(npe_results_path: Path, output_path: str):
    """Load results from disk, plot, and save to output_path."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_results = _collect_results(npe_results_path)
    if not all_results and npe_results_path.exists():
        for child in sorted(npe_results_path.iterdir()):
            if child.is_dir():
                all_results.extend(_collect_results(child))

    if not all_results:
        raise RuntimeError(f"No results found in {npe_results_path}")

    for r in sorted(all_results, key=lambda x: (x.get("_run_group", ""), x["budget"])):
        group = r.get("_run_group", "")
        compressor = r.get("compressor_class_path", "unknown")
        print(
            f"  [{group}] budget={r['budget']:>6d}: FoM = {r['fom_mean']:.2f} ± {r['fom_std']:.2f} "
            f"(val_nll={r['best_val_nll']:.4f}, compressor={compressor})"
        )

    groups = sorted(set(r.get("_run_group", "default") for r in all_results))
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.get_cmap("tab10", max(len(groups), 1))
    for idx, group in enumerate(groups):
        rows = [r for r in all_results if r.get("_run_group", "default") == group]
        budgets = np.array([r["budget"] for r in rows])
        fom_means = np.array([r["fom_mean"] for r in rows])
        fom_stds = np.array([r["fom_std"] for r in rows])
        order = np.argsort(budgets)
        budgets, fom_means, fom_stds = budgets[order], fom_means[order], fom_stds[order]
        ax.errorbar(
            budgets,
            fom_means,
            yerr=fom_stds,
            fmt="o-",
            color=cmap(idx),
            markersize=6,
            capsize=3,
            linewidth=1.2,
            zorder=5,
            label=group,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Number of compressor training simulations", fontsize=14)
    ax.set_ylabel("Figure of Merit (FoM)", fontsize=14)
    ax.set_title("Inference quality vs simulation budget", fontsize=15)
    if len(groups) > 1:
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


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
