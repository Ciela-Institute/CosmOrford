#!/usr/bin/env python3
"""Backfill NPE W&B offline runs from completed results artifacts."""

import argparse
import json
from pathlib import Path

import wandb


def _iter_result_dirs(results_root: Path, run_glob: str):
    for run_dir in sorted(results_root.glob(run_glob)):
        for budget_dir in sorted(run_dir.glob("budget-*")):
            if (budget_dir / "results.json").exists():
                yield run_dir, budget_dir


def _as_image(path: Path):
    return wandb.Image(str(path)) if path.exists() else None


def backfill_result(run_dir: Path, budget_dir: Path, entity: str, project: str, overwrite: bool) -> bool:
    marker = budget_dir / ".wandb_npe_backfill_done"
    if marker.exists() and not overwrite:
        return False

    payload = json.loads((budget_dir / "results.json").read_text())
    run_name = f"{run_dir.name}-{budget_dir.name}-backfill"
    tags = ["hos-npe", "inference", "backfill", f"budget-{payload.get('budget', 'unknown')}"]

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        mode="offline",
        dir=str(budget_dir),
        tags=tags,
        config={
            "run_name": run_dir.name,
            "budget_dir": budget_dir.name,
            "budget": payload.get("budget"),
            "compressor_class_path": payload.get("compressor_class_path"),
            "checkpoint_metric_name": payload.get("checkpoint_metric_name"),
            "checkpoint_mode": payload.get("checkpoint_mode"),
            "flow_transforms": payload.get("flow_transforms"),
            "flow_hidden_dim": payload.get("flow_hidden_dim"),
            "n_noise_realizations": payload.get("n_noise_realizations"),
            "n_fiducial_maps": payload.get("n_fiducial_maps"),
            "n_posterior_samples": payload.get("n_posterior_samples"),
            "posterior_contour_backend": payload.get("posterior_contour_backend"),
        },
    )

    run.log(
        {
            "fom/mean": float(payload.get("fom_mean", 0.0)),
            "fom/std": float(payload.get("fom_std", 0.0)),
            "npe/best_val_nll": float(payload.get("best_val_nll", 0.0)),
        }
    )

    image_payload = {}
    for idx, fname in enumerate(payload.get("posterior_plot_files", []) or []):
        p = budget_dir / fname
        img = _as_image(p)
        if img is not None:
            image_payload[f"posterior/density_obs{idx:02d}"] = img

    for idx, fname in enumerate(payload.get("posterior_contour_files", []) or []):
        p = budget_dir / fname
        img = _as_image(p)
        if img is not None:
            image_payload[f"posterior/contour_obs{idx:02d}"] = img

    if image_payload:
        run.log(image_payload)

    run.summary["results_dir"] = str(budget_dir)
    run.summary["fom_mean"] = float(payload.get("fom_mean", 0.0))
    run.summary["fom_std"] = float(payload.get("fom_std", 0.0))
    run.summary["best_val_nll"] = float(payload.get("best_val_nll", 0.0))
    run.finish()

    marker.write_text("done\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Backfill offline W&B NPE runs from saved results")
    parser.add_argument("--results-root", default=str(Path.home() / "experiments" / "npe_results"))
    parser.add_argument("--run-glob", default="hos_npe*")
    parser.add_argument("--entity", default="cosmostat")
    parser.add_argument("--project", default="neurips-wl-challenge-hos-npe")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    total = 0
    created = 0
    for run_dir, budget_dir in _iter_result_dirs(results_root, args.run_glob):
        total += 1
        if backfill_result(
            run_dir=run_dir,
            budget_dir=budget_dir,
            entity=args.entity,
            project=args.project,
            overwrite=args.overwrite,
        ):
            created += 1

    print(f"Result dirs scanned: {total}")
    print(f"Backfill runs created: {created}")


if __name__ == "__main__":
    main()
