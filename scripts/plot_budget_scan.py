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
    parser.add_argument("--after", default=None, help="only include runs created after this datetime (e.g. 2026-03-07T18:30)")
    parser.add_argument("--output", default="budget_scan.pdf", help="output plot file")
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", filters={"tags": args.tag})

    budgets_finished = []
    scores_finished = []
    budgets_crashed = []
    scores_crashed = []
    for run in runs:
        if args.after and run.created_at < args.after:
            continue
        # Extract budget from run name (e.g. "budget-100" -> 100)
        name = run.name
        if name.startswith("budget-"):
            n_train = int(name.split("-")[1])
        else:
            n_train = run.config.get("data", {}).get("init_args", {}).get("max_train_samples", 0)
            if n_train == 0:
                n_train = 20200

        hist = list(run.scan_history(keys=["val_score"]))
        if not hist:
            print(f"Skipping {run.name} (no val_score data)")
            continue

        best_score = max(h["val_score"] for h in hist if "val_score" in h)
        if run.state == "finished":
            budgets_finished.append(n_train)
            scores_finished.append(best_score)
        else:
            budgets_crashed.append(n_train)
            scores_crashed.append(best_score)
        print(f"{run.name} [{run.state}]: n_train={n_train}, best_val_score={best_score:.2f}")

    budgets = budgets_finished + budgets_crashed
    scores = scores_finished + scores_crashed
    if not budgets:
        print("No completed runs found.")
        return

    budgets_finished = np.array(budgets_finished)
    scores_finished = np.array(scores_finished)
    budgets_crashed = np.array(budgets_crashed)
    scores_crashed = np.array(scores_crashed)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Plot all points connected by a line (sorted)
    all_budgets = np.array(budgets)
    all_scores = np.array(scores)
    order = np.argsort(all_budgets)
    ax.plot(all_budgets[order], all_scores[order], "-", color="C0", linewidth=1.5, alpha=0.3)
    # Finished runs as solid markers
    if len(budgets_finished):
        ax.scatter(budgets_finished, scores_finished, s=80, color="C0", zorder=5, label="Finished")
    # Crashed runs as open markers
    if len(budgets_crashed):
        ax.scatter(budgets_crashed, scores_crashed, s=80, facecolors="none", edgecolors="C0",
                   linewidths=2, zorder=5, label="Crashed (partial)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of training simulations", fontsize=14)
    ax.set_ylabel("Best validation score", fontsize=14)
    ax.set_title("Constraining power vs simulation budget", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
