#!/usr/bin/env python3
"""Resubmit failed/cancelled/timeout jobs from a HOS-NPE ablation manifest."""

import argparse
import csv
import datetime as dt
import pathlib
import re
import subprocess
import sys


FAIL_PREFIXES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")


def read_manifest(path: pathlib.Path):
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


def submit_row(row):
    cmd = [
        "sbatch",
        "--export=ALL",
        "scripts/submit_hos_npe_pipeline.sh",
        row["compressor_config"],
        row["npe_config"],
        row["run_name"],
        row["budget"],
        row["seed"],
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout.strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {out}")
    return m.group(1), out


def main():
    parser = argparse.ArgumentParser(description="Resubmit failed HOS-NPE ablation jobs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default=None, help="Output manifest for resubmitted jobs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    in_manifest = pathlib.Path(args.manifest)
    rows = read_manifest(in_manifest)
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_manifest = pathlib.Path(args.out or in_manifest.with_name(in_manifest.stem + f"_rerun_{stamp}.csv"))

    reruns = []
    for row in rows:
        job_id = row.get("job_id", "")
        state = sacct_state(job_id) if job_id else "NOJOB"
        if not state.startswith(FAIL_PREFIXES):
            continue
        if args.dry_run:
            new_job_id = ""
            submit_output = (
                f"sbatch --export=ALL scripts/submit_hos_npe_pipeline.sh "
                f"{row['compressor_config']} {row['npe_config']} {row['run_name']} {row['budget']} {row['seed']}"
            )
        else:
            new_job_id, submit_output = submit_row(row)
        out = dict(row)
        out["previous_job_id"] = job_id
        out["previous_state"] = state
        out["job_id"] = new_job_id
        out["submit_output"] = submit_output
        out["timestamp_utc"] = stamp
        reruns.append(out)
        print(f"rerun {row['regime_id']} seed={row['seed']} prev={job_id}({state}) -> {new_job_id or '[dry-run]'}")

    if not reruns:
        print("No failed/cancelled/timeout jobs to rerun.")
        return

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(reruns[0].keys()))
        writer.writeheader()
        writer.writerows(reruns)
    print(f"Wrote rerun manifest: {out_manifest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
