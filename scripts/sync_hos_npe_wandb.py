#!/usr/bin/env python3
"""Sync offline W&B runs for HOS-NPE compression and inference pages."""

import argparse
import subprocess
import sys
from pathlib import Path


def _sync_path(path: Path, project: str, entity: str, dry_run: bool) -> None:
    if not path.exists():
        print(f"Skip missing path: {path}")
        return
    cmd = [
        "wandb",
        "sync",
        "--include-offline",
        "--mark-synced",
        "--entity",
        entity,
        "--project",
        project,
        str(path),
    ]
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync offline W&B runs for HOS-NPE")
    parser.add_argument("--entity", default="cosmostat")
    parser.add_argument("--compression-project", default="neurips-wl-challenge-hos-compression")
    parser.add_argument("--inference-project", default="neurips-wl-challenge-hos-npe")
    parser.add_argument("--compression-root", default=str(Path.home() / "experiments" / "checkpoints"))
    parser.add_argument("--inference-root", default=str(Path.home() / "experiments" / "npe_results"))
    parser.add_argument("--run-glob", default="hos_npe*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sync-inference-backfill",
        action="store_true",
        help="Create offline NPE wandb runs from results.json + posterior artifacts before sync",
    )
    parser.add_argument("--backfill-overwrite", action="store_true")
    args = parser.parse_args()

    compression_root = Path(args.compression_root)
    inference_root = Path(args.inference_root)

    matched_compression = sorted(compression_root.glob(f"{args.run_glob}/**/wandb"))
    matched_inference = sorted(inference_root.glob(f"{args.run_glob}/**/wandb"))

    if args.sync_inference_backfill:
        backfill_cmd = [
            "python",
            "scripts/backfill_hos_npe_inference_wandb.py",
            "--results-root",
            str(inference_root),
            "--run-glob",
            args.run_glob,
            "--entity",
            args.entity,
            "--project",
            args.inference_project,
        ]
        if args.backfill_overwrite:
            backfill_cmd.append("--overwrite")
        print("+", " ".join(backfill_cmd))
        if not args.dry_run:
            subprocess.run(backfill_cmd, check=True)
        matched_inference = sorted(inference_root.glob(f"{args.run_glob}/**/wandb"))

    if not matched_compression and not matched_inference:
        print("No W&B offline directories found.")
        return

    for p in matched_compression:
        _sync_path(p, args.compression_project, args.entity, args.dry_run)
    for p in matched_inference:
        _sync_path(p, args.inference_project, args.entity, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed with exit {exc.returncode}", file=sys.stderr)
        raise
