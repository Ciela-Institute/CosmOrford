#!/usr/bin/env python3
import argparse
import csv
import pathlib
import re
import statistics
import subprocess
import sys
from collections import defaultdict


VAL_SCORE_RE = re.compile(r"val_score:\s*([-+]?\d*\.?\d+)")
VAL_MSE_RE = re.compile(r"val_mse:\s*([-+]?\d*\.?\d+)")


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
    try:
        proc = subprocess.run(
            ["sacct", "-n", "-P", "-j", job_id, "-o", "State"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return "UNKNOWN"
    for line in proc.stdout.splitlines():
        state = line.strip().split("|")[0]
        if state:
            return state
    return "UNKNOWN"


def parse_log_metrics(path: pathlib.Path):
    if not path.exists():
        return None, None
    best_score = None
    best_mse = None
    for line in path.read_text(errors="ignore").splitlines():
        ms = VAL_SCORE_RE.search(line)
        if ms:
            score = float(ms.group(1))
            if best_score is None or score > best_score:
                best_score = score
        mm = VAL_MSE_RE.search(line)
        if mm:
            mse = float(mm.group(1))
            if best_mse is None or mse < best_mse:
                best_mse = mse
    return best_score, best_mse


def main():
    parser = argparse.ArgumentParser(description="Aggregate ablation batch results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--jobout-dir", default="jobout")
    parser.add_argument("--out-detail", default=None)
    parser.add_argument("--out-summary", default=None)
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--all-attempts", action="store_true", help="Aggregate all manifest rows instead of latest attempt per (config, seed).")
    args = parser.parse_args()

    manifest = pathlib.Path(args.manifest)
    rows = list(csv.DictReader(manifest.open()))
    if not args.all_attempts:
        rows = latest_attempt_rows(rows)
    jobout_dir = pathlib.Path(args.jobout_dir)

    detail = []
    grouped_scores = defaultdict(list)
    grouped_mses = defaultdict(list)

    for row in rows:
        job_id = row["job_id"]
        cfg = row["config"]
        seed = int(row["seed"])
        state = sacct_state(job_id) if job_id else "NOJOB"
        log_path = jobout_dir / f"cosmoford_{job_id}.out"
        best_score, best_mse = parse_log_metrics(log_path)
        detail.append(
            {
                "config": cfg,
                "seed": seed,
                "job_id": job_id,
                "state": state,
                "best_val_score": best_score,
                "best_val_mse": best_mse,
                "log_path": str(log_path),
            }
        )
        if best_score is not None and (not args.completed_only or state.startswith("COMPLETED")):
            grouped_scores[cfg].append(best_score)
        if best_mse is not None and (not args.completed_only or state.startswith("COMPLETED")):
            grouped_mses[cfg].append(best_mse)

    summary = []
    for cfg in sorted(set(r["config"] for r in detail)):
        scores = grouped_scores.get(cfg, [])
        mses = grouped_mses.get(cfg, [])
        score_mean = statistics.mean(scores) if scores else None
        score_std = statistics.pstdev(scores) if len(scores) > 1 else 0.0 if len(scores) == 1 else None
        mse_mean = statistics.mean(mses) if mses else None
        summary.append(
            {
                "config": cfg,
                "n_scored": len(scores),
                "score_mean": score_mean,
                "score_std": score_std,
                "mse_mean": mse_mean,
            }
        )

    out_detail = pathlib.Path(args.out_detail or manifest.with_name(manifest.stem + "_detail.csv"))
    out_summary = pathlib.Path(args.out_summary or manifest.with_name(manifest.stem + "_summary.csv"))

    with out_detail.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "seed",
                "job_id",
                "state",
                "best_val_score",
                "best_val_mse",
                "log_path",
            ],
        )
        w.writeheader()
        w.writerows(detail)

    with out_summary.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["config", "n_scored", "score_mean", "score_std", "mse_mean"],
        )
        w.writeheader()
        w.writerows(summary)

    print(f"Wrote detail: {out_detail}")
    print(f"Wrote summary: {out_summary}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
