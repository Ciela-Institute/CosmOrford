#!/usr/bin/env python3
"""Submit and monitor representative analytical HOS->NPE runs.

This orchestrates:
1) Stage-1 compressor training jobs for representative configs
2) Stage-2 NPE+FoM jobs with matching NPE configs

Usage:
  python scripts/run_hos_npe_representative.py --mode submit-stage1
  python scripts/run_hos_npe_representative.py --mode submit-stage2 --manifest jobout/hos_npe_stage1_manifest_*.csv
  python scripts/run_hos_npe_representative.py --mode status --manifest jobout/hos_npe_stage2_manifest_*.csv
"""

import argparse
import csv
import datetime as dt
import pathlib
import re
import subprocess
import sys
from typing import Iterable

import yaml


def load_matrix(path: pathlib.Path) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    rows = data.get("hos_npe_representative", {}).get("compressors", [])
    if not rows:
        raise ValueError(f"No representative compressors found in {path}")
    return rows


def submit_stage1_job(config: str, run_name: str, budget: int = 20200, seed: int = 42) -> str:
    cmd = [
        "sbatch",
        "--export=ALL",
        "scripts/submit_hos_npe_compressor_job.sh",
        config,
        run_name,
        str(budget),
        str(seed),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout.strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {out}")
    return m.group(1)


def submit_stage2_job(npe_config: str, run_name: str) -> str:
    cmd = [
        "sbatch",
        "--export=ALL",
        "scripts/submit_hos_npe_job.sh",
        npe_config,
        run_name,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout.strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {out}")
    return m.group(1)


def write_manifest(rows: Iterable[dict], out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def sacct_state(job_id: str) -> str:
    cmd = ["sacct", "-n", "-P", "-j", job_id, "-o", "State"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return "UNKNOWN"
    for line in proc.stdout.splitlines():
        state = line.strip().split("|")[0]
        if state:
            return state
    return "UNKNOWN"


def run_stage1(matrix_path: pathlib.Path, seed: int, out: pathlib.Path):
    rows = load_matrix(matrix_path)
    matrix_data = yaml.safe_load(matrix_path.read_text())
    budget = int(matrix_data.get("hos_npe_representative", {}).get("budget", 20200))
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = out or pathlib.Path(f"jobout/hos_npe_stage1_manifest_{stamp}.csv")
    out_rows = []
    for row in rows:
        run_name = f"hos_npe_{row['name']}"
        job_id = submit_stage1_job(row["config"], run_name=run_name, budget=budget, seed=seed)
        out_rows.append(
            {
                "timestamp_utc": stamp,
                "name": row["name"],
                "run_name": run_name,
                "stage": "stage1",
                "config": row["config"],
                "npe_config": row["npe_config"],
                "wandb_tag": row.get("wandb_tag", ""),
                "budget": budget,
                "seed": seed,
                "job_id": job_id,
            }
        )
        print(f"[stage1] {row['name']} -> job {job_id}")
    write_manifest(out_rows, manifest)
    print(f"Wrote manifest: {manifest}")


def run_stage2(stage1_manifest: pathlib.Path, out: pathlib.Path):
    rows = read_manifest(stage1_manifest)
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = out or pathlib.Path(f"jobout/hos_npe_stage2_manifest_{stamp}.csv")
    out_rows = []

    for row in rows:
        stage1_state = sacct_state(row["job_id"])
        if not stage1_state.startswith("COMPLETED"):
            print(
                f"[stage2] skipping {row['name']} because stage1 job {row['job_id']} "
                f"is in state {stage1_state}"
            )
            continue

        job_id = submit_stage2_job(row["npe_config"], run_name=row["run_name"])
        out_rows.append(
            {
                "timestamp_utc": stamp,
                "name": row["name"],
                "run_name": row["run_name"],
                "stage": "stage2",
                "npe_config": row["npe_config"],
                "depends_on_stage1_job": row["job_id"],
                "job_id": job_id,
            }
        )
        print(f"[stage2] {row['name']} -> job {job_id}")

    if not out_rows:
        print("No stage2 jobs were submitted.")
        return
    write_manifest(out_rows, manifest)
    print(f"Wrote manifest: {manifest}")


def run_status(manifest_path: pathlib.Path):
    rows = read_manifest(manifest_path)
    for row in rows:
        state = sacct_state(row["job_id"])
        print(f"{row['stage']} {row['name']} job={row['job_id']} state={state}")


def main():
    parser = argparse.ArgumentParser(description="Representative HOS->NPE orchestration")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["submit-stage1", "submit-stage2", "status"],
    )
    parser.add_argument(
        "--matrix",
        default="configs/experiments/hos_npe_representative_matrix.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.mode == "submit-stage1":
        run_stage1(pathlib.Path(args.matrix), args.seed, pathlib.Path(args.out) if args.out else None)
        return

    if args.manifest is None:
        raise ValueError("--manifest is required for submit-stage2/status")

    manifest_path = pathlib.Path(args.manifest)
    if args.mode == "submit-stage2":
        run_stage2(manifest_path, pathlib.Path(args.out) if args.out else None)
    else:
        run_status(manifest_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
