#!/usr/bin/env python3
import argparse
import csv
import pathlib
import re
import subprocess
import sys
from collections import Counter


def read_manifest(path: pathlib.Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _job_id_as_int(job_id: str) -> int:
    try:
        return int(job_id)
    except (TypeError, ValueError):
        return -1


def latest_attempt_rows(rows):
    latest = {}
    for row in rows:
        key = (row.get("config", ""), row.get("seed", ""))
        jid = _job_id_as_int(row.get("job_id", ""))
        prev = latest.get(key)
        if prev is None or jid > _job_id_as_int(prev.get("job_id", "")):
            latest[key] = row
    return list(latest.values())


def sacct_state(job_id: str) -> str:
    cmd = ["sacct", "-n", "-P", "-j", job_id, "-o", "State"]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("sacct command not found; run this script on a SLURM login node.") from exc
    except subprocess.CalledProcessError:
        return "UNKNOWN"
    for line in proc.stdout.splitlines():
        state = line.strip().split("|")[0]
        if state:
            return state
    return "UNKNOWN"


def extract_last_val_score(log_path: pathlib.Path):
    if not log_path.exists():
        return None
    score_re = re.compile(r"val_score:\s*([-+]?\d*\.?\d+)")
    last = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = score_re.search(line)
        if m:
            last = float(m.group(1))
    return last


def main():
    parser = argparse.ArgumentParser(description="Monitor submitted ablation jobs.")
    parser.add_argument("--manifest", required=True, help="CSV produced by submit_ablation_batch.py")
    parser.add_argument("--jobout-dir", default="jobout")
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--all-attempts", action="store_true", help="Report all manifest rows instead of latest attempt per (config, seed).")
    args = parser.parse_args()

    rows = read_manifest(pathlib.Path(args.manifest))
    if not args.all_attempts:
        rows = latest_attempt_rows(rows)
    jobout_dir = pathlib.Path(args.jobout_dir)
    states = Counter()
    failed = []

    for row in rows:
        job_id = row["job_id"]
        if not job_id:
            continue
        state = sacct_state(job_id)
        states[state] += 1
        if state.startswith("FAILED") or state.startswith("CANCELLED") or state.startswith("TIMEOUT"):
            failed.append((job_id, row["config"], row["seed"]))

    print("State summary:")
    for k, v in sorted(states.items()):
        print(f"  {k}: {v}")

    print("\nCompleted job snapshots:")
    for row in rows:
        job_id = row["job_id"]
        if not job_id:
            continue
        state = sacct_state(job_id)
        if state.startswith("COMPLETED"):
            log_path = jobout_dir / f"cosmoford_{job_id}.out"
            last_score = extract_last_val_score(log_path)
            if last_score is None:
                print(f"  job={job_id} config={row['config']} seed={row['seed']} last_val_score=NA")
            else:
                print(
                    f"  job={job_id} config={row['config']} seed={row['seed']} "
                    f"last_val_score={last_score:.4f}"
                )

    if args.show_failures and failed:
        print("\nFailures:")
        for job_id, cfg, seed in failed:
            print(f"  job={job_id} config={cfg} seed={seed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
