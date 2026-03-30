#!/usr/bin/env python3
"""Aggregate representative HOS-NPE results from run manifests.

Reads stage-2 manifest produced by `scripts/run_hos_npe_representative.py` and
collects FoM/NLL metrics from each run's `results.json`.
"""

import argparse
import csv
import json
import pathlib
import statistics
from collections import defaultdict


def read_manifest(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_results_json(results_root: pathlib.Path, run_name: str, budget: int = 20200):
    path = results_root / run_name / f"budget-{budget}" / "results.json"
    if not path.exists():
        return None, path
    return json.loads(path.read_text()), path


def main():
    parser = argparse.ArgumentParser(description="Aggregate representative HOS-NPE results")
    parser.add_argument("--manifest", required=True, help="Stage-2 manifest CSV")
    parser.add_argument(
        "--results-root",
        default=str(pathlib.Path.home() / "experiments" / "npe_results"),
        help="Root directory containing per-run NPE result folders",
    )
    parser.add_argument("--budget", type=int, default=20200)
    parser.add_argument("--out-detail", default=None)
    parser.add_argument("--out-summary", default=None)
    args = parser.parse_args()

    manifest = pathlib.Path(args.manifest)
    rows = read_manifest(manifest)
    results_root = pathlib.Path(args.results_root)

    detail = []
    grouped = defaultdict(list)

    for row in rows:
        run_name = row["run_name"]
        name = row["name"]
        rec, path = load_results_json(results_root, run_name, budget=args.budget)
        if rec is None:
            detail.append(
                {
                    "name": name,
                    "run_name": run_name,
                    "job_id": row["job_id"],
                    "status": "missing_results",
                    "results_path": str(path),
                    "fom_mean": None,
                    "fom_std": None,
                    "best_val_nll": None,
                }
            )
            continue

        detail.append(
            {
                "name": name,
                "run_name": run_name,
                "job_id": row["job_id"],
                "status": "ok",
                "results_path": str(path),
                "fom_mean": rec.get("fom_mean"),
                "fom_std": rec.get("fom_std"),
                "best_val_nll": rec.get("best_val_nll"),
            }
        )
        if rec.get("fom_mean") is not None:
            grouped[name].append(float(rec["fom_mean"]))

    summary = []
    for name in sorted(grouped.keys() | {r["name"] for r in detail}):
        vals = grouped.get(name, [])
        summary.append(
            {
                "name": name,
                "n_runs": len(vals),
                "fom_mean_avg": statistics.mean(vals) if vals else None,
                "fom_mean_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0 if len(vals) == 1 else None,
            }
        )

    out_detail = pathlib.Path(args.out_detail or manifest.with_name(manifest.stem + "_results_detail.csv"))
    out_summary = pathlib.Path(args.out_summary or manifest.with_name(manifest.stem + "_results_summary.csv"))

    with out_detail.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "run_name",
                "job_id",
                "status",
                "results_path",
                "fom_mean",
                "fom_std",
                "best_val_nll",
            ],
        )
        w.writeheader()
        w.writerows(detail)

    with out_summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "n_runs", "fom_mean_avg", "fom_mean_std"])
        w.writeheader()
        w.writerows(summary)

    print(f"Wrote detail: {out_detail}")
    print(f"Wrote summary: {out_summary}")


if __name__ == "__main__":
    main()
