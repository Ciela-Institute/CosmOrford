#!/usr/bin/env python3
"""Monitor HOS-NPE ablation-transfer pipeline jobs and summarize artifacts."""

import argparse
import csv
import json
import pathlib
import subprocess
import sys
from collections import Counter, defaultdict


def read_manifest(path: pathlib.Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def sacct_state(job_id: str) -> str:
    if not job_id:
        return "NOJOB"
    cmd = ["sacct", "-n", "-P", "-j", job_id, "-o", "State"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return "UNKNOWN"
    for line in proc.stdout.splitlines():
        state = line.strip().split("|")[0]
        if state:
            return state
    return "UNKNOWN"


def summarize_results(root: pathlib.Path, run_name: str, budget: int):
    rdir = root / run_name / f"budget-{budget}"
    rjson = rdir / "results.json"
    if not rjson.exists():
        return {
            "results_exists": False,
            "results_dir": str(rdir),
            "fom_mean": None,
            "fom_std": None,
            "best_val_nll": None,
            "n_plots": 0,
            "n_contours": 0,
            "posterior_samples_norm": False,
            "posterior_samples_phys": False,
        }
    rec = json.loads(rjson.read_text())
    return {
        "results_exists": True,
        "results_dir": str(rdir),
        "fom_mean": rec.get("fom_mean"),
        "fom_std": rec.get("fom_std"),
        "best_val_nll": rec.get("best_val_nll"),
        "n_plots": len(rec.get("posterior_plot_files", []) or []),
        "n_contours": len(rec.get("posterior_contour_files", []) or []),
        "posterior_samples_norm": bool(rec.get("posterior_samples_norm_file")),
        "posterior_samples_phys": bool(rec.get("posterior_samples_phys_file")),
    }


def main():
    parser = argparse.ArgumentParser(description="Monitor HOS-NPE ablation-transfer batch")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", default=str(pathlib.Path.home() / "experiments" / "npe_results"))
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    manifest = pathlib.Path(args.manifest)
    rows = read_manifest(manifest)
    results_root = pathlib.Path(args.results_root)

    detail = []
    state_counts = Counter()
    per_regime = defaultdict(list)

    for row in rows:
        state = sacct_state(row.get("job_id", ""))
        state_counts[state] += 1
        budget = int(row.get("budget") or 20200)
        result = summarize_results(results_root, row["run_name"], budget)
        rec = {
            "regime_id": row["regime_id"],
            "seed": row["seed"],
            "job_id": row["job_id"],
            "state": state,
            "run_name": row["run_name"],
            "results_exists": result["results_exists"],
            "fom_mean": result["fom_mean"],
            "fom_std": result["fom_std"],
            "best_val_nll": result["best_val_nll"],
            "n_plots": result["n_plots"],
            "n_contours": result["n_contours"],
            "posterior_samples_norm": result["posterior_samples_norm"],
            "posterior_samples_phys": result["posterior_samples_phys"],
            "results_dir": result["results_dir"],
        }
        detail.append(rec)
        per_regime[row["regime_id"]].append(rec)

    print("State summary:")
    for k, v in sorted(state_counts.items()):
        print(f"  {k}: {v}")

    print("\nPer-regime status:")
    for regime_id in sorted(per_regime):
        group = per_regime[regime_id]
        done = sum(1 for x in group if str(x["state"]).startswith("COMPLETED"))
        with_results = sum(1 for x in group if x["results_exists"])
        foms = [float(x["fom_mean"]) for x in group if x["fom_mean"] is not None]
        if foms:
            mean_fom = sum(foms) / len(foms)
            print(
                f"  {regime_id}: completed={done}/{len(group)}, "
                f"results={with_results}/{len(group)}, mean_FoM={mean_fom:.3f}"
            )
        else:
            print(f"  {regime_id}: completed={done}/{len(group)}, results={with_results}/{len(group)}")

    out_csv = pathlib.Path(args.out_csv or manifest.with_name(manifest.stem + "_status.csv"))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "regime_id",
                "seed",
                "job_id",
                "state",
                "run_name",
                "results_exists",
                "fom_mean",
                "fom_std",
                "best_val_nll",
                "n_plots",
                "n_contours",
                "posterior_samples_norm",
                "posterior_samples_phys",
                "results_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(detail)
    print(f"\nWrote status CSV: {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
