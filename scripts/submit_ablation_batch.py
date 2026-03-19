#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import pathlib
import re
import subprocess
import sys
from typing import Iterable, List

import yaml


def parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def load_repeats_config(path: pathlib.Path) -> tuple[list[str], list[int]]:
    data = yaml.safe_load(path.read_text())
    configs = data.get("repeats", {}).get("key_configs", [])
    seeds = data.get("repeats", {}).get("seeds", [])
    if not configs or not seeds:
        raise ValueError(f"Invalid repeats config: {path}")
    return configs, [int(s) for s in seeds]


def iter_jobs(configs: Iterable[str], seeds: Iterable[int]):
    for cfg in configs:
        for seed in seeds:
            yield cfg, int(seed)


def submit_one(config: str, seed: int, dry_run: bool) -> tuple[str, str]:
    cmd = [
        "sbatch",
        "--export=ALL,CONFIG=" + config + ",SEED=" + str(seed),
        "scripts/submit_job.sh",
    ]
    if dry_run:
        return "", " ".join(cmd)

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout.strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {out}")
    return m.group(1), out


def main():
    parser = argparse.ArgumentParser(description="Submit ablation jobs in batch.")
    parser.add_argument("--repeats-config", default="configs/experiments/ablation_repeats.yaml")
    parser.add_argument("--configs", nargs="*", default=None, help="Optional explicit config list")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds; overrides repeats config")
    parser.add_argument("--manifest-out", default=None, help="Output CSV path for submitted jobs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rep_path = pathlib.Path(args.repeats_config)
    configs_cfg, seeds_cfg = load_repeats_config(rep_path)
    configs = args.configs if args.configs else configs_cfg
    seeds = parse_csv_ints(args.seeds) if args.seeds else seeds_cfg

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = pathlib.Path(args.manifest_out or f"jobout/ablation_manifest_{timestamp}.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg, seed in iter_jobs(configs, seeds):
        job_id, raw = submit_one(cfg, seed, args.dry_run)
        rows.append({
            "timestamp_utc": timestamp,
            "config": cfg,
            "seed": seed,
            "job_id": job_id,
            "dry_run_cmd_or_output": raw,
        })
        print(f"{cfg} seed={seed} -> {job_id or '[dry-run]'}")

    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp_utc", "config", "seed", "job_id", "dry_run_cmd_or_output"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
