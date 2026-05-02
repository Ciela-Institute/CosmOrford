#!/usr/bin/env python3
"""Generate raw-stats NPE configs for best-performing statistic combinations."""
import pathlib, yaml

OUT = pathlib.Path("configs/experiments/generated_hos_npe")

COMBOS = {
    "l1_only_l1_wide": {
        "budget_tag": "hos-npe-l1_only_l1_wide",
        "tags": ["hos-npe", "l1_only_l1_wide", "rawstats", "no-compression"],
        "ckpt": "hos_npe_l1_only_l1_wide_s42",
    },
    "hos_l1_wide": {
        "budget_tag": "hos-npe-hos_l1_wide",
        "tags": ["hos-npe", "hos_l1_wide", "rawstats", "no-compression"],
        "ckpt": "hos_npe_hos_l1_wide_s42",
    },
    "hos_scat_J4": {
        "budget_tag": "hos-npe-hos_scat_J4",
        "tags": ["hos-npe", "hos_scat_J4", "rawstats", "no-compression"],
        "ckpt": "hos_npe_hos_scat_J4_s42",
    },
    "snr_full_wide": {
        "budget_tag": "hos-npe-snr_full_wide",
        "tags": ["hos-npe", "snr_full_wide", "rawstats", "no-compression"],
        "ckpt": "hos_npe_snr_full_wide_s42",
    },
    "ps_hos_scat_full_wide": {
        "budget_tag": "hos-npe-ps_hos_scat_full_wide",
        "tags": ["hos-npe", "ps_hos_scat_full_wide", "rawstats", "no-compression"],
        "ckpt": "hos_npe_ps_hos_scat_full_wide_s42",
    },
    "snr_full_wide_meanstd_fullgeom_noflip_noshift": {
        "budget_tag": "hos-npe-snr_full_wide_meanstd_fullgeom_noflip_noshift",
        "tags": ["hos-npe", "snr_full_wide_meanstd_fullgeom_noflip_noshift", "rawstats", "no-compression"],
        "ckpt": "hos_npe_snr_full_wide_meanstd_fullgeom_noflip_noshift_s42",
    },
}

# (name, transforms, hidden_dim, lr, patience)
SETUPS = {
    "p5":           (4, 128, 1e-3, 5),
    "widecond":     (4, 256, 1e-3, 5),
    "smallflow":    (2,  64, 1e-3, 5),
}

# For very high-dim contexts only submit widecond + smallflow
HIGH_DIM = {"snr_full_wide_meanstd_fullgeom_noflip_noshift"}

generated = []
for combo, info in COMBOS.items():
    setups = SETUPS if combo not in HIGH_DIM else {k: v for k, v in SETUPS.items() if k in ("widecond", "smallflow")}
    for setup, (transforms, hidden, lr, patience) in setups.items():
        fname = f"{combo}_rawstats_{setup}_npe.yaml"
        run_name = f"hos_npe_{info['ckpt'].replace('hos_npe_', '')}rawstats_{setup}"
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
            "use_raw_stats_context": True,
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
        npe_run_name = f"hos_npe_{info['ckpt'].replace('hos_npe_', '')}_rawstats_{setup}"
        generated.append((fname, npe_run_name, ckpt_path))
        print(f"  {fname}")

print(f"\nTotal: {len(generated)} configs")
# Write submission script
with open("scripts/submit_combo_rawstats_npe.sh", "w") as f:
    f.write("#!/bin/bash\nset -euo pipefail\n\n")
    for fname, run_name, ckpt_path in generated:
        f.write(
            f'sbatch --gpus-per-node=h100:1 scripts/submit_npe_light.sh '
            f'configs/experiments/generated_hos_npe/{fname} '
            f'{run_name} '
            f'{ckpt_path}\n'
        )
print("Wrote scripts/submit_combo_rawstats_npe.sh")
