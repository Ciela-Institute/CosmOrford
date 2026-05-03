#!/usr/bin/env python3
"""
Generate VMIM compressor exploration configs for L1-norms (320d → 8d).

VMIM (Variational Mutual Information Maximization) uses a RealNVP (affine
coupling) flow instead of MAF as the compressor training loss, with a piecewise
constant LR schedule that decays by 0.7x at regular intervals — mimicking the
original JAX/sbi_lens implementation in PyTorch.

All runs go in configs/experiments/vmim_explore/ and
$HOME/experiments/vmim_explore/ to keep them separate from compressor_explore.
"""
import copy
import pathlib
import yaml

CONF_DIR = pathlib.Path("configs/experiments/vmim_explore")
CONF_DIR.mkdir(exist_ok=True)

EXPLORE_BASE = "$HOME/experiments/vmim_explore"

# ── Common base ────────────────────────────────────────────────────────────
BASE_MODEL = {
    "class_path": "cosmoford.models_nopatch.StatsCompressorNoPatch",
    "init_args": {
        "summary_dim": 8,
        "summary_hidden_dim": 512,
        "summary_n_hidden": 3,
        "summary_dropout_rate": 0.0,   # VMIM default: no dropout
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
        "max_lr": 1e-4,                # VMIM default LR
        "loss_type": "log_prob",
        "use_flow": True,
        "flow_type": "realnvp",        # RealNVP (VMIM) instead of MAF
        "flow_transforms": 4,
        "flow_hidden_dim": 128,
        "lr_schedule": "piecewise_constant",
        "piecewise_decay_rate": 0.7,
        "piecewise_n_decays": 9,
        "n_val_noise": 8,
        "use_fixed_normalization": True,
    },
}

BASE_DATA = {
    "class_path": "cosmoford.dataset.ChallengeDataModule",
    "init_args": {"batch_size": 128, "num_workers": 12, "max_train_samples": 20200},
}

BASE_TRAINER_CALLBACKS = [
    {"class_path": "LearningRateMonitor", "init_args": {"logging_interval": "step"}},
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
# (name, overrides, max_epochs, ckpt_metric, long_job)
# All use flow_type="realnvp" (VMIM) by default from BASE_MODEL
EXPERIMENTS = [
    # ── Direct comparison to the best MAF variants ──────────────────────
    # vmim_s: same as flow_s but with RealNVP + piecewise LR + lr=1e-4
    ("vmim_s", {
        "flow_transforms": 4, "flow_hidden_dim": 64,
    }, 30, "val_nll", False),

    # vmim_m: RealNVP 8T×128h, 30ep (equivalent to flow_m)
    ("vmim_m", {
        "flow_transforms": 8, "flow_hidden_dim": 128,
    }, 30, "val_nll", False),

    # vmim_s_long: RealNVP 4T×64h, 60ep piecewise (equivalent to flow_s_long)
    ("vmim_s_long", {
        "flow_transforms": 4, "flow_hidden_dim": 64,
    }, 60, "val_nll", True),

    # vmim_m_long: RealNVP 8T×128h, 60ep piecewise (best MAF equivalent)
    ("vmim_m_long", {
        "flow_transforms": 8, "flow_hidden_dim": 128,
    }, 60, "val_nll", True),

    # vmim_l_long: RealNVP 8T×256h, 60ep piecewise (was unstable with MAF)
    ("vmim_l_long", {
        "flow_transforms": 8, "flow_hidden_dim": 256,
    }, 60, "val_nll", True),

    # vmim_deep_m: deeper MLP + RealNVP 8T×128h (equivalent to deep_flow_m)
    ("vmim_deep_m", {
        "summary_n_hidden": 5, "summary_hidden_dim": 1024,
        "summary_dropout_rate": 0.05,
        "flow_transforms": 8, "flow_hidden_dim": 128,
    }, 60, "val_nll", True),

    # vmim_cosine: same as vmim_m_long but cosine LR instead of piecewise
    # (to isolate the LR schedule effect from the flow architecture effect)
    ("vmim_cosine", {
        "flow_transforms": 8, "flow_hidden_dim": 128,
        "lr_schedule": "cosine", "max_lr": 1e-3,
    }, 60, "val_nll", True),

    # vmim_lr3: same as vmim_m_long but with lr=1e-3 (like best MAF variants)
    ("vmim_lr3", {
        "flow_transforms": 8, "flow_hidden_dim": 128,
        "max_lr": 1e-3,
    }, 60, "val_nll", True),
]


def npe_flow_arch(summary_dim):
    if summary_dim <= 8:
        return {"flow_transforms": 4, "flow_hidden_dim": 64}
    else:
        return {"flow_transforms": 4, "flow_hidden_dim": 128}


generated = []

for name, overrides, max_epochs, ckpt_metric, long_job in EXPERIMENTS:
    model_args = copy.deepcopy(BASE_MODEL["init_args"])
    model_args.update(overrides)
    summary_dim = model_args["summary_dim"]

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
            "callbacks": BASE_TRAINER_CALLBACKS + [make_ckpt_callback(ckpt_metric)],
            "logger": make_wandb_logger(
                f"vmim_explore_l1_{name}_budget20200",
                ["vmim-explore", "l1", "realnvp", name],
            ),
        },
    }
    comp_path = CONF_DIR / f"l1_{name}_compressor.yaml"
    with open(comp_path, "w") as f:
        yaml.dump(comp_cfg, f, default_flow_style=False, sort_keys=False)

    npe_arch = npe_flow_arch(summary_dim)
    npe_cfg = {
        "budgets": [20200],
        "compressor_class_path": "cosmoford.models_nopatch.StatsCompressorNoPatch",
        "checkpoint_metric_name": ckpt_metric,
        "checkpoint_mode": "min",
        "wandb_entity": "cosmostat",
        "wandb_project": "neurips-wl-challenge-hos-compression",
        "wandb_budget_tag": f"vmim-explore-l1-{name}",
        "log_npe_to_wandb": True,
        "wandb_npe_entity": "cosmostat",
        "wandb_npe_project": "neurips-wl-challenge-hos-npe",
        "wandb_npe_tags": ["vmim-explore", "l1", "realnvp", name],
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

# ── Submission script ──────────────────────────────────────────────────────
lines = ["#!/bin/bash", "set -euo pipefail", ""]
lines.append(f'VMIM_BASE="{EXPLORE_BASE}"')
lines.append("")

for name, long_job, ckpt_metric, summary_dim in generated:
    run_name = f"l1_{name}"
    comp_cfg = f"configs/experiments/vmim_explore/l1_{name}_compressor.yaml"
    npe_cfg  = f"configs/experiments/vmim_explore/l1_{name}_npe.yaml"
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

submit_path = pathlib.Path("scripts/submit_vmim_explore.sh")
with open(submit_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote {submit_path}")
