#!/usr/bin/env python3
"""Submit a representative HOS-NPE batch from successful ablation regimes.

For each selected regime and seed:
1) Generate a StatsCompressorNoPatch config mirroring summary-stat choices
2) Generate a matching NPE/FoM config
3) Submit one end-to-end pipeline SLURM job via submit_hos_npe_pipeline.sh
"""

import argparse
import csv
import datetime as dt
import pathlib
import re
import subprocess
import sys
from typing import Dict, List

import yaml


DEFAULT_REGIMES: Dict[str, dict] = {
    "ps_fixed": {
        "source_config": "configs/experiments/ablation_ps_fixed.yaml",
        "family": "PS",
        "notes": "PS-only fixed-normalization baseline",
    },
    "scattering_J4": {
        "source_config": "configs/experiments/ablation_scattering_J4.yaml",
        "family": "Scattering",
        "notes": "Scattering-only reference J=4",
    },
    "l1_ns4": {
        "source_config": "configs/experiments/ablation_l1_ns4.yaml",
        "family": "L1",
        "notes": "L1-only baseline ns=4",
    },
    "l1_only_l1_wide": {
        "source_config": "configs/experiments/ablation_l1_only_l1_wide.yaml",
        "family": "L1",
        "notes": "L1-only wide SNR",
    },
    "hos_ns4": {
        "source_config": "configs/experiments/ablation_hos_ns4.yaml",
        "family": "HOS",
        "notes": "Full HOS baseline ns=4",
    },
    "hos_l1_wide": {
        "source_config": "configs/experiments/ablation_hos_l1_wide.yaml",
        "family": "HOS",
        "notes": "Full HOS with wide L1 SNR",
    },
    "hos_scat_J4": {
        "source_config": "configs/experiments/ablation_hos_scat_J4.yaml",
        "family": "HOS+Scat",
        "notes": "HOS+Scattering baseline J=4",
    },
    "snr_full_wide": {
        "source_config": "configs/experiments/ablation_snr_full_wide.yaml",
        "family": "HOS+Scat",
        "notes": "HOS+Scat with wide peaks+L1 SNR",
    },
    "ps_hos_fixed": {
        "source_config": "configs/experiments/ablation_ps_hos_fixed.yaml",
        "family": "PS+HOS",
        "notes": "PS+HOS fixed-normalization",
    },
    "ps_l1_fixed": {
        "source_config": "configs/experiments/ablation_ps_l1_fixed.yaml",
        "family": "PS+L1",
        "notes": "PS+L1 fixed-normalization",
    },
    "ps_hos_scat_full_wide": {
        "source_config": "configs/experiments/ablation_ps_hos_scat_full_wide.yaml",
        "family": "PS+HOS+Scat",
        "notes": "PS+HOS+Scat with wide peaks+L1 SNR",
    },
    "snr_full_wide_meanstd_fullgeom_noflip_noshift": {
        "source_config": "configs/experiments/ablation_snr_full_wide_meanstd_fullgeom_noflip_noshift.yaml",
        "family": "HOS+Scat",
        "notes": "Best WST-audited full-wide setting",
    },
}


STATS_KEYS = [
    "use_ps",
    "use_hos",
    "use_scattering",
    "hos_l1_only",
    "hos_peaks_only",
    "hos_n_scales",
    "hos_n_bins",
    "hos_l1_nbins",
    "hos_min_snr",
    "hos_max_snr",
    "hos_l1_min_snr",
    "hos_l1_max_snr",
    "scattering_J",
    "scattering_L",
    "scattering_normalization",
    "scattering_feature_pooling",
    "scattering_mask_pooling",
    "scattering_geometry",
    "augment_flip",
    "augment_shift",
]


def _parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _load_yaml(path: pathlib.Path):
    return yaml.safe_load(path.read_text())


def _write_yaml(path: pathlib.Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _build_compressor_cfg(
    source_cfg_path: pathlib.Path,
    regime_id: str,
    budget: int,
):
    source = _load_yaml(source_cfg_path)
    model_args = source.get("model", {}).get("init_args", {})
    data_args = source.get("data", {}).get("init_args", {})
    trainer_cfg = source.get("trainer", {})

    init_args = {
        "summary_dim": 8,
        "summary_hidden_dim": 512,
        "summary_n_hidden": 3,
        "summary_dropout_rate": 0.1,
        "use_ps": False,
        "use_hos": False,
        "use_scattering": False,
        "augment_flip": False,
        "augment_shift": False,
        "warmup_steps": int(model_args.get("warmup_steps", 500)),
        "max_lr": float(model_args.get("max_lr", 0.001)),
        "decay_rate": float(model_args.get("decay_rate", 0.85)),
        "decay_every_epochs": int(model_args.get("decay_every_epochs", 1)),
        "loss_type": "log_prob",
        "lr_schedule": str(model_args.get("lr_schedule", "step")),
        "n_val_noise": int(model_args.get("n_val_noise", 8)),
    }
    for k in STATS_KEYS:
        if k in model_args:
            init_args[k] = model_args[k]

    if not (init_args.get("use_ps") or init_args.get("use_hos") or init_args.get("use_scattering")):
        raise ValueError(f"Regime {regime_id} has no active summary family in {source_cfg_path}")

    max_epochs = int(trainer_cfg.get("max_epochs", 30))
    batch_size = int(data_args.get("batch_size", 256))
    num_workers = int(data_args.get("num_workers", 12))

    payload = {
        "model": {
            "class_path": "cosmoford.models_nopatch.StatsCompressorNoPatch",
            "init_args": init_args,
        },
        "data": {
            "class_path": "cosmoford.dataset.ChallengeDataModule",
            "init_args": {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "max_train_samples": int(budget),
            },
        },
        "trainer": {
            "max_epochs": max_epochs,
            "accelerator": "gpu",
            "precision": "16-mixed",
            "log_every_n_steps": 1,
            "check_val_every_n_epoch": 1,
            "callbacks": [
                {
                    "class_path": "LearningRateMonitor",
                    "init_args": {"logging_interval": "step"},
                },
                {"class_path": "cosmoford.trainer.EMAWeightAveraging"},
                {
                    "class_path": "ModelCheckpoint",
                    "init_args": {
                        "monitor": "val_log_prob",
                        "mode": "min",
                        "save_top_k": 3,
                        "save_last": True,
                        "filename": "epoch={epoch:02d}-step={step}-val_log_prob={val_log_prob:.4f}",
                    },
                },
            ],
            "logger": {
                "class_path": "WandbLogger",
                "init_args": {
                    "name": f"hos_npe_{regime_id}_budget{budget}",
                    "project": "neurips-wl-challenge",
                    "entity": "cosmostat",
                    "tags": ["hos-npe", "ablation-transfer", regime_id, f"budget-{budget}"],
                    "log_model": True,
                },
            },
        },
    }
    return payload


def _build_npe_cfg(regime_id: str, budget: int, fiducial_maps: int):
    return {
        "budgets": [int(budget)],
        "compressor_class_path": "cosmoford.models_nopatch.StatsCompressorNoPatch",
        "checkpoint_metric_name": "val_log_prob",
        "checkpoint_mode": "min",
        "wandb_entity": "cosmostat",
        "wandb_project": "neurips-wl-challenge",
        "wandb_budget_tag": f"hos-npe-{regime_id}",
        "n_noise_realizations": 16,
        "npe_epochs": 500,
        "npe_lr": 1.0e-3,
        "npe_batch_size": 512,
        "npe_patience": 50,
        "npe_seeds": 5,
        "flow_transforms": 4,
        "flow_hidden_dim": 64,
        "n_posterior_samples": 10000,
        "n_fiducial_maps": int(fiducial_maps),
        "save_posterior_samples": True,
        "save_posterior_plots": True,
        "posterior_plot_bins": 80,
    }


def _submit_pipeline_job(compressor_cfg: pathlib.Path, npe_cfg: pathlib.Path, run_name: str, budget: int, seed: int):
    cmd = [
        "sbatch",
        "--export=ALL",
        "scripts/submit_hos_npe_pipeline.sh",
        str(compressor_cfg),
        str(npe_cfg),
        run_name,
        str(budget),
        str(seed),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout.strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {out}")
    return m.group(1), out


def main():
    parser = argparse.ArgumentParser(description="Submit HOS-NPE ablation-transfer batch.")
    parser.add_argument("--budget", type=int, default=20200)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--fiducial-maps", type=int, default=10)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--regimes", nargs="*", default=None, help="Optional subset of regime IDs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    seeds = _parse_seeds(args.seeds)
    selected = args.regimes if args.regimes else list(DEFAULT_REGIMES.keys())
    unknown = sorted(set(selected) - set(DEFAULT_REGIMES.keys()))
    if unknown:
        raise ValueError(f"Unknown regime IDs: {unknown}")

    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = pathlib.Path(args.manifest_out or f"jobout/hos_npe_ablation_manifest_{stamp}.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)

    generated_dir = root / "configs" / "experiments" / "generated_hos_npe"
    rows = []
    for regime_id in selected:
        meta = DEFAULT_REGIMES[regime_id]
        source_cfg = root / meta["source_config"]
        if not source_cfg.exists():
            raise FileNotFoundError(f"Missing source config for regime {regime_id}: {source_cfg}")

        compressor_cfg = generated_dir / f"{regime_id}_compressor.yaml"
        npe_cfg = generated_dir / f"{regime_id}_npe.yaml"
        _write_yaml(compressor_cfg, _build_compressor_cfg(source_cfg, regime_id, args.budget))
        _write_yaml(npe_cfg, _build_npe_cfg(regime_id, args.budget, args.fiducial_maps))

        for seed in seeds:
            run_name = f"hos_npe_{regime_id}_s{seed}"
            if args.dry_run:
                job_id = ""
                submit_out = (
                    f"sbatch --export=ALL scripts/submit_hos_npe_pipeline.sh "
                    f"{compressor_cfg} {npe_cfg} {run_name} {args.budget} {seed}"
                )
            else:
                job_id, submit_out = _submit_pipeline_job(compressor_cfg, npe_cfg, run_name, args.budget, seed)
            rows.append(
                {
                    "timestamp_utc": stamp,
                    "regime_id": regime_id,
                    "family": meta["family"],
                    "notes": meta["notes"],
                    "source_config": str(source_cfg.relative_to(root)),
                    "compressor_config": str(compressor_cfg.relative_to(root)),
                    "npe_config": str(npe_cfg.relative_to(root)),
                    "seed": seed,
                    "budget": args.budget,
                    "run_name": run_name,
                    "job_id": job_id,
                    "submit_output": submit_out,
                }
            )
            print(f"{regime_id} seed={seed} -> {job_id or '[dry-run]'}")

    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "regime_id",
                "family",
                "notes",
                "source_config",
                "compressor_config",
                "npe_config",
                "seed",
                "budget",
                "run_name",
                "job_id",
                "submit_output",
            ],
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
