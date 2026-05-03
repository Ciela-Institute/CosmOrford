#!/usr/bin/env python3
"""
Generate compressor exploration configs for L1-norms (320d → 8d).

Tests: flow loss vs Gaussian, architecture depth/width, training length,
dropout, and bottleneck dimension. All in configs/experiments/compressor_explore/.
Submission script chains compressor → NPE via SLURM dependencies.
"""
import pathlib, yaml, textwrap

CONF_DIR = pathlib.Path("configs/experiments/compressor_explore")
CONF_DIR.mkdir(exist_ok=True)

EXPLORE_BASE = "$HOME/experiments/compressor_explore"

# ── Common base for all compressor configs ─────────────────────────────────
BASE_MODEL = {
    "class_path": "cosmoford.models_nopatch.StatsCompressorNoPatch",
    "init_args": {
        "summary_dim": 8,
        "summary_hidden_dim": 512,
        "summary_n_hidden": 3,
        "summary_dropout_rate": 0.1,
        "use_ps": False,
        "use_hos": True,
        "use_scattering": False,
        "hos_l1_only": True,
        "hos_n_scales": 4,
        "hos_l1_nbins": 80,
        "hos_l1_min_snr": -7.0,
        "hos_l1_max_snr": 7.0,
        "augment_flip": False,
        "augment_shift": False,
        "warmup_steps": 500,
        "max_lr": 0.002,
        "decay_rate": 0.85,
        "decay_every_epochs": 1,
        "loss_type": "log_prob",
        "use_flow": False,
        "flow_transforms": 4,
        "flow_hidden_dim": 64,
        "lr_schedule": "step",
        "n_val_noise": 8,
        "use_fixed_normalization": True,
    },
}

BASE_DATA = {
    "class_path": "cosmoford.dataset.ChallengeDataModule",
    "init_args": {"batch_size": 256, "num_workers": 12, "max_train_samples": 20200},
}

BASE_TRAINER_CALLBACKS = [
    {"class_path": "LearningRateMonitor", "init_args": {"logging_interval": "step"}},
    {"class_path": "cosmoford.trainer.EMAWeightAveraging"},
]

def make_ckpt_callback(monitor):
    return {
        "class_path": "ModelCheckpoint",
        "init_args": {
            "monitor": monitor,
            "mode": "min",
            "save_top_k": 3,
            "save_last": True,
            "filename": f"epoch={{epoch:02d}}-step={{step}}-{monitor}={{{monitor}:.4f}}",
        },
    }

def make_wandb_logger(name, tags):
    return {
        "class_path": "WandbLogger",
        "init_args": {
            "name": name,
            "project": "neurips-wl-challenge-hos-compression",
            "entity": "cosmostat",
            "tags": tags,
            "log_model": True,
        },
    }

# ── Experiment definitions ─────────────────────────────────────────────────
# Each entry: (name, overrides_to_model_init_args, max_epochs, ckpt_metric, long_job)
# long_job=True → request 2:30 wall time instead of 1:30
EXPERIMENTS = [
    ("baseline_repro", {}, 30, "val_log_prob", False),
    ("flow_s",  {"use_flow": True, "flow_transforms": 4, "flow_hidden_dim": 64}, 30, "val_nll", False),
    ("flow_m",  {"use_flow": True, "flow_transforms": 8, "flow_hidden_dim": 128}, 30, "val_nll", False),
    ("flow_l",  {"use_flow": True, "flow_transforms": 8, "flow_hidden_dim": 256}, 30, "val_nll", False),
    ("flow_s_long", {
        "use_flow": True, "flow_transforms": 4, "flow_hidden_dim": 64,
        "max_lr": 0.001, "lr_schedule": "cosine",
    }, 60, "val_nll", True),
    ("deep_flow_s", {
        "use_flow": True, "flow_transforms": 4, "flow_hidden_dim": 64,
        "summary_n_hidden": 5, "summary_hidden_dim": 1024, "summary_dropout_rate": 0.05,
        "max_lr": 0.001, "lr_schedule": "cosine",
    }, 60, "val_nll", True),
    ("deep_flow_m", {
        "use_flow": True, "flow_transforms": 8, "flow_hidden_dim": 128,
        "summary_n_hidden": 5, "summary_hidden_dim": 1024, "summary_dropout_rate": 0.05,
        "max_lr": 0.001, "lr_schedule": "cosine",
    }, 60, "val_nll", True),
    ("deep_gauss", {
        "summary_n_hidden": 5, "summary_hidden_dim": 1024, "summary_dropout_rate": 0.05,
        "max_lr": 0.001, "lr_schedule": "cosine",
    }, 60, "val_log_prob", True),
    ("nodrop_flow", {
        "use_flow": True, "flow_transforms": 4, "flow_hidden_dim": 64,
        "summary_dropout_rate": 0.0, "max_lr": 0.001, "lr_schedule": "cosine",
    }, 60, "val_nll", True),
    ("score_loss", {
        "loss_type": "score", "max_lr": 0.001, "lr_schedule": "cosine",
    }, 60, "val_log_prob", True),
    ("dim16_flow", {
        "use_flow": True, "flow_transforms": 4, "flow_hidden_dim": 64, "summary_dim": 16,
    }, 30, "val_nll", False),
    ("dim32_flow", {
        "use_flow": True, "flow_transforms": 4, "flow_hidden_dim": 64, "summary_dim": 32,
    }, 30, "val_nll", False),
]

# ── NPE flow architectures ─────────────────────────────────────────────────
# For 8d context: p5 (4T×64h). For 16d/32d: medflow (4T×128h).
def npe_flow_arch(summary_dim):
    if summary_dim <= 8:
        return {"flow_transforms": 4, "flow_hidden_dim": 64}
    else:
        return {"flow_transforms": 4, "flow_hidden_dim": 128}

generated = []

for name, overrides, max_epochs, ckpt_metric, long_job in EXPERIMENTS:
    import copy
    model_args = copy.deepcopy(BASE_MODEL["init_args"])
    model_args.update(overrides)
    summary_dim = model_args["summary_dim"]

    # ── Compressor config ──────────────────────────────────────────────────
    comp_cfg = {
        "model": {
            "class_path": BASE_MODEL["class_path"],
            "init_args": model_args,
        },
        "data": BASE_DATA,
        "trainer": {
            "max_epochs": max_epochs,
            "accelerator": "gpu",
            "precision": "16-mixed",
            "log_every_n_steps": 1,
            "check_val_every_n_epoch": 1,
            "callbacks": (BASE_TRAINER_CALLBACKS if not model_args.get("use_flow") else
                          [BASE_TRAINER_CALLBACKS[0]]) + [make_ckpt_callback(ckpt_metric)],
            "logger": make_wandb_logger(
                f"compressor_explore_l1_{name}_budget20200",
                ["compressor-explore", "l1", name],
            ),
        },
    }
    comp_path = CONF_DIR / f"l1_{name}_compressor.yaml"
    with open(comp_path, "w") as f:
        yaml.dump(comp_cfg, f, default_flow_style=False, sort_keys=False)

    # ── NPE config ────────────────────────────────────────────────────────
    npe_arch = npe_flow_arch(summary_dim)
    npe_cfg = {
        "budgets": [20200],
        "compressor_class_path": "cosmoford.models_nopatch.StatsCompressorNoPatch",
        "checkpoint_metric_name": ckpt_metric,
        "checkpoint_mode": "min",
        "wandb_entity": "cosmostat",
        "wandb_project": "neurips-wl-challenge-hos-compression",
        "wandb_budget_tag": f"compressor-explore-l1-{name}",
        "log_npe_to_wandb": True,
        "wandb_npe_entity": "cosmostat",
        "wandb_npe_project": "neurips-wl-challenge-hos-npe",
        "wandb_npe_tags": ["compressor-explore", "l1", name],
        "wandb_npe_log_images": True,
        "n_noise_realizations": 1,
        "n_holdout_train_maps": 0,
        "npe_epochs": 100,
        "npe_lr": 1e-3,
        "npe_batch_size": 512,
        "npe_patience": 5,
        "npe_seeds": 5,
        "flow_transforms": npe_arch["flow_transforms"],
        "flow_hidden_dim": npe_arch["flow_hidden_dim"],
        "normalize_context": True,
        "use_raw_stats_context": False,
        "n_posterior_samples": 10000,
        "n_fiducial_maps": 10,
        "save_posterior_samples": True,
        "save_posterior_plots": True,
        "posterior_plot_bins": 80,
        "use_getdist_contours": True,
    }
    npe_path = CONF_DIR / f"l1_{name}_npe.yaml"
    with open(npe_path, "w") as f:
        yaml.dump(npe_cfg, f, default_flow_style=False, sort_keys=False)

    generated.append((name, long_job, ckpt_metric, summary_dim))
    print(f"  {comp_path.name}  +  {npe_path.name}")

print(f"\nTotal: {len(generated)} pairs")

# ── Submission script with SLURM dependencies ──────────────────────────────
lines = ["#!/bin/bash", "set -euo pipefail", ""]
lines.append(f'EXPLORE_BASE="{EXPLORE_BASE}"')
lines.append("")

for name, long_job, ckpt_metric, summary_dim in generated:
    run_name = f"l1_{name}"
    comp_cfg = f"configs/experiments/compressor_explore/l1_{name}_compressor.yaml"
    npe_cfg  = f"configs/experiments/compressor_explore/l1_{name}_npe.yaml"
    ckpt_dir = f"{EXPLORE_BASE}/checkpoints/{run_name}"
    npe_dir  = f"{EXPLORE_BASE}/npe_results/{run_name}"
    summ_dir = f"{EXPLORE_BASE}/summaries_cache/{run_name}"
    time_str = "00-02:30" if long_job else "00-01:30"

    lines.append(f"# ── {name} ──")
    lines.append(
        f'COMP_JOB=$(CHECKPOINTS_PATH="{ckpt_dir}" sbatch --parsable '
        f'--time={time_str} --gpus-per-node=h100:1 '
        f'scripts/submit_compressor_only.sh {comp_cfg} {run_name})'
    )
    lines.append(f'echo "{name}: compressor job $COMP_JOB"')
    lines.append(
        f'NPE_RESULTS_PATH="{npe_dir}" '
        f'SUMMARIES_CACHE_PATH="{summ_dir}" '
        f'sbatch --parsable '
        f'--dependency=afterok:$COMP_JOB '
        f'--gpus-per-node=h100:1 '
        f'scripts/submit_npe_light.sh '
        f'{npe_cfg} {run_name} {ckpt_dir}'
    )
    lines.append("")

submit_path = pathlib.Path("scripts/submit_compressor_explore.sh")
with open(submit_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote {submit_path}")
