"""Plot constraining power vs simulation budget from wandb results.

Usage:
    python scripts/plot_budget_scan.py [--project PROJECT] [--entity ENTITY] [--output FILE]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import wandb


def main():
    parser = argparse.ArgumentParser(description="Plot budget scan results from wandb")
    parser.add_argument("--project", default="neurips-wl-challenge", help="wandb project name")
    parser.add_argument("--entity", default="cosmostat", help="wandb entity")
    parser.add_argument("--tag", default="budget-scan", help="wandb tag to filter runs")
    parser.add_argument("--output", default="budget_scan.pdf", help="output plot file")
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", filters={"tags": args.tag})

    budgets = []
    scores = []
    for run in runs:
        if run.state != "finished":
            print(f"Skipping {run.name} (state={run.state})")
            continue

        n_train = run.config.get("data", {}).get("init_args", {}).get("max_train_samples", 0)
        if n_train == 0:
            n_train = 20200  # full dataset

        hist = list(run.scan_history(keys=["val_score"]))
        if not hist:
            print(f"Skipping {run.name} (no val_score data)")
            continue

        best_score = max(h["val_score"] for h in hist if "val_score" in h)
        budgets.append(n_train)
        scores.append(best_score)
        print(f"{run.name}: n_train={n_train}, best_val_score={best_score:.2f}")

    if not budgets:
        print("No completed runs found.")
        return

    # Sort by budget
    order = np.argsort(budgets)
    budgets = np.array(budgets)[order]
    scores = np.array(scores)[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(budgets, scores, "o-", color="C0", markersize=8, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Number of training simulations", fontsize=14)
    ax.set_ylabel("Best validation score", fontsize=14)
    ax.set_title("Constraining power vs simulation budget", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
