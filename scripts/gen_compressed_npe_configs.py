#!/usr/bin/env python3
"""Generate compressed NPE configs (MLP compressor → 8d context → flow)."""
import pathlib, yaml

OUT = pathlib.Path("configs/experiments/generated_hos_npe")

COMBOS = {
    "l1_only_l1_wide": {
        "budget_tag": "hos-npe-l1_only_l1_wide",
        "tags": ["hos-npe", "l1_only_l1_wide", "compressed"],
        "ckpt": "hos_npe_l1_only_l1_wide_s42",
    },
    "hos_ns4": {
        "budget_tag": "hos-npe-hos_ns4",
        "tags": ["hos-npe", "hos_ns4", "compressed"],
        "ckpt": "hos_npe_hos_ns4_s42",
    },
    "hos_scat_J4": {
        "budget_tag": "hos-npe-hos_scat_J4",
        "tags": ["hos-npe", "hos_scat_J4", "compressed"],
        "ckpt": "hos_npe_hos_scat_J4_s42",
    },
    "snr_full_wide": {
        "budget_tag": "hos-npe-snr_full_wide",
        "tags": ["hos-npe", "snr_full_wide", "compressed"],
        "ckpt": "hos_npe_snr_full_wide_s42",
    },
    "snr_full_wide_meanstd_fullgeom_noflip_noshift": {
        "budget_tag": "hos-npe-snr_full_wide_meanstd_fullgeom_noflip_noshift",
        "tags": ["hos-npe", "snr_full_wide_meanstd_fullgeom_noflip_noshift", "compressed"],
        "ckpt": "hos_npe_snr_full_wide_meanstd_fullgeom_noflip_noshift_s42",
    },
}

# For 8d compressed context: smallflow/p5/medflow are all appropriate
# (transforms, hidden_dim, lr, patience)
SETUPS = {
    "smallflow": (2, 64, 1e-3, 5),
    "p5":        (4, 64, 1e-3, 5),
    "medflow":   (4, 128, 1e-3, 5),
}

generated = []
for combo, info in COMBOS.items():
    for setup, (transforms, hidden, lr, patience) in SETUPS.items():
        fname = f"{combo}_compressed_{setup}_npe.yaml"
        npe_run_name = f"hos_npe_{combo}_s42_compressed_{setup}"
        cfg = {
            "budgets": [20200],
            "compressor_class_path": "cosmoford.models_nopatch.StatsCompressorNoPatch",
            "checkpoint_metric_name": "val_log_prob",
            "checkpoint_mode": "min",
            "wandb_entity": "cosmostat",
            "wandb_project": "neurips-wl-challenge-hos-compression",
            "wandb_budget_tag": info["budget_tag"],
            "log_npe_to_wandb": True,
            "wandb_npe_entity": "cosmostat",
            "wandb_npe_project": "neurips-wl-challenge-hos-npe",
            "wandb_npe_tags": info["tags"] + [setup],
            "wandb_npe_log_images": True,
            "n_noise_realizations": 1,
            "n_holdout_train_maps": 0,
            "npe_epochs": 100,
            "npe_lr": lr,
            "npe_batch_size": 512,
            "npe_patience": patience,
            "npe_seeds": 5,
            "flow_transforms": transforms,
            "flow_hidden_dim": hidden,
            "normalize_context": True,
            "use_raw_stats_context": False,
            "n_posterior_samples": 10000,
            "n_fiducial_maps": 10,
            "save_posterior_samples": True,
            "save_posterior_plots": True,
            "posterior_plot_bins": 80,
            "use_getdist_contours": True,
        }
        path = OUT / fname
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        ckpt_path = f"/home/tersenov/experiments/checkpoints/{info['ckpt']}"
        generated.append((fname, npe_run_name, ckpt_path))
        print(f"  {fname}")

print(f"\nTotal: {len(generated)} configs")

COMPRESSED_BASE = "$HOME/experiments/npe_results/compressed_hos_npe"
with open("scripts/submit_compressed_npe.sh", "w") as f:
    f.write("#!/bin/bash\nset -euo pipefail\n\n")
    for fname, run_name, ckpt_path in generated:
        f.write(
            f'NPE_RESULTS_PATH="{COMPRESSED_BASE}/{run_name}" '
            f'sbatch --gpus-per-node=h100:1 scripts/submit_npe_light.sh '
            f'configs/experiments/generated_hos_npe/{fname} '
            f'{run_name} '
            f'{ckpt_path}\n'
        )
print("Wrote scripts/submit_compressed_npe.sh")
