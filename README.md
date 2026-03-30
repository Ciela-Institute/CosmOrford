<div align="center">

# CosmOrford

*How to build optimal summary statistics for weak gravitational lensing cosmology under a limited simulation budget?*

[![Challenge](https://img.shields.io/badge/Challenge-FAIR%20Universe%20WL-blue)](https://www.codabench.org/competitions/8934/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Datasets-CosmoStat-FFD21E)](https://huggingface.co/CosmoStat)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

This repository investigates how to build optimal summary statistics for weak gravitational lensing cosmology under a limited simulation budget. This work distills lessons learned from participating in the [FAIR Universe - Weak Lensing ML Uncertainty Challenge](https://www.codabench.org/competitions/8934/).

We compare different strategies for building summary statistics — analytical, neural without pre-training, and neural with pre-training on cheaper simulations — within a unified evaluation framework.

---

## 📐 Evaluation framework

All summary strategies are evaluated through the same three-step pipeline, which ensures a fair comparison across approaches.

**Step 1 — Compression to 8D.**
Every summary (analytical or neural) is compressed into an 8-dimensional vector. This shared dimensionality puts all approaches on equal footing for the downstream posterior estimation.

**Step 2 — Neural Posterior Estimation (NPE).**
A Masked Autoregressive Flow (MAF) is trained on (summary, θ) pairs drawn from the holdout dataset — with noise augmentation applied to the maps before compression — to approximate the posterior p(Ω_m, S_8 | summary).

**Step 3 — Figure of Merit (FoM).**
Posterior samples are drawn for maps from the `fiducial` split of the holdout dataset (Ω_m = 0.29, S_8 = 0.81). The FoM = 1 / sqrt(det Cov(Ω_m, S_8)) measures how tightly the posterior constrains the parameters.

**Scripts:**

| Script | Description |
|---|---|
| `cosmoford/models_nopatch.py` | Compressor model, trained via `cosmoford/trainer.py` |
| `scripts/run_npe_budget_scan.py` | Trains the NPE flow and computes FoM, sweeping over simulation budgets; supports pluggable compressor classes |
| `scripts/plot_fom_budget.py` | Plots FoM vs. simulation budget from saved results |

**Datasets:**

| Dataset | Split | Used for |
|---|---|---|
| [`CosmoStat/neurips-wl-challenge-flat`](https://huggingface.co/datasets/CosmoStat/neurips-wl-challenge-flat) | `train` / `validation` | Compressor training and validation |
| [`CosmoStat/neurips-wl-challenge-holdout`](https://huggingface.co/datasets/CosmoStat/neurips-wl-challenge-holdout) | `train` | NPE training (summaries precomputed with noise augmentation) |
| [`CosmoStat/neurips-wl-challenge-holdout`](https://huggingface.co/datasets/CosmoStat/neurips-wl-challenge-holdout) | `fiducial` | FoM evaluation |

⚠️ See below for how to access the datasets.

---

## 🗂️ Summary statistics strategies

### 🔢 Option A — Analytical summaries

Physically motivated statistics computed directly from the masked convergence maps, such as peak counts, wavelet ℓ₁-norm, power spectrum, and scattering features.  
To align with the neural-summary NPE pipeline, analytical summaries now use `StatsCompressorNoPatch` (`cosmoford/models_nopatch.py`) with the same `compress(x) -> 8D` interface and `val_log_prob` checkpoint convention.

Representative stage-1 configs (fixed budget 20200):

- `configs/experiments/hos_npe_compressor_ps.yaml`
- `configs/experiments/hos_npe_compressor_l1.yaml`
- `configs/experiments/hos_npe_compressor_hos.yaml`
- `configs/experiments/hos_npe_compressor_hos_scat.yaml`

Representative stage-2 NPE configs:

- `configs/experiments/hos_npe_budget_ps.yaml`
- `configs/experiments/hos_npe_budget_l1.yaml`
- `configs/experiments/hos_npe_budget_hos.yaml`
- `configs/experiments/hos_npe_budget_hos_scat.yaml`

Orchestration helpers:

- `scripts/run_hos_npe_representative.py` (submit/monitor stage1 and stage2)
- `scripts/submit_hos_npe_compressor_job.sh`
- `scripts/submit_hos_npe_job.sh`
- `scripts/analyze_hos_npe_results.py`

Example flow:

```bash
# Stage 1 (compressors)
python scripts/run_hos_npe_representative.py --mode submit-stage1

# Stage 2 (NPE/FoM) once stage-1 jobs complete
python scripts/run_hos_npe_representative.py --mode submit-stage2 --manifest jobout/hos_npe_stage1_manifest_<STAMP>.csv

# Aggregate FoM outputs
python scripts/analyze_hos_npe_results.py --manifest jobout/hos_npe_stage2_manifest_<STAMP>.csv
```

**Dataset:** [`CosmoStat/neurips-wl-challenge-flat`](https://huggingface.co/datasets/CosmoStat/neurips-wl-challenge-flat)

---

### 🧠 Option B — Neural compressor (no pre-training)

An EfficientNetV2-S network trained directly on the N-body simulations, compressing each convergence map to 8 summary statistics by maximizing the Gaussian log-likelihood.

**Training script:** `trainer fit -c configs/experiments/efficientnet_v2_s_logp_.yaml`

**Dataset:** [`CosmoStat/neurips-wl-challenge-flat`](https://huggingface.co/datasets/CosmoStat/neurips-wl-challenge-flat)

---

### 🚀 Option C — Neural compressor with pre-training

Same EfficientNetV2-S architecture, but first pre-trained on a larger set of cheaper simulations to reduce overfitting when the N-body budget is small, then fine-tuned on the N-body dataset. The compressor is trained with a Gaussian log-likelihood loss.

**Fine-tuning script:** `trainer fit -c configs/finetune_from_pretrain_nopatch_logp.yaml`
> Update `pretrained_checkpoint_path` in the config to point to your pre-trained checkpoint.

**Fine-tuning dataset:** [`CosmoStat/neurips-wl-challenge-flat`](https://huggingface.co/datasets/CosmoStat/neurips-wl-challenge-flat)

Available pre-training datasets and their configs:

| Simulation type | Local dataset | Pre-training config |
|---|---|---|
| Gaussian Random Field (GRF) | `CosmoStat/GRF_HF` | `None` |
| LogNormal | `CosmoStat/lognormal` | `configs/experiments/pretrain_lognormal_nopatch_logp.yaml` |
| Gower Street | `CosmoStat/gowerstreet-train` | `configs/experiments/pretrain_gowerstreet_nopatch_logp.yaml` |
| OT-emulated (from LogNormal) | `CosmoStat/ot_emulated` | `configs/pretrain_otemulated_nopatch_logp.yaml` |
| OT-emulated from TBD | output of the emulator (see below) | TBD |

```bash
# Example: pre-train on LogNormal, then fine-tune on challenge data
trainer fit -c configs/experiments/pretrain_lognormal_nopatch_logp.yaml
trainer fit -c configs/finetune_from_pretrain_nopatch_logp.yaml
```

---

### ⚙️ Building the OT-emulated dataset

To bridge the gap between cheap simulations and the N-body distribution, a UNet emulator is trained using conditional optimal-transport flow matching (COT-FM). It maps LogNormal (or Gower Street) convergence maps to the distribution of N-body maps, conditioned on cosmological parameters. The emulated maps are then used as pre-training data for Option C.

**Training script:** `cosmoford/emulator/cot_fm.py`
**UNet configs:** `configs/unet_condition_small.yaml` / `configs/unet_condition_large.yaml`
**Build HF dataset from emulated maps:** `scripts/hf_emulated_dataset.py`

| Dataset | Role |
|---|---|
| `CosmoStat/GRF_HF` | Cheap simulations to be corrected (GRF) |
| `CosmoStat/lognormal` | Cheap simulations to be corrected (LogNormal) |
| PM source | To be generated |
| [`CosmoStat/neurips-wl-challenge-flat`](https://huggingface.co/datasets/CosmoStat/neurips-wl-challenge-flat) | N-body target distribution for the emulator |

```bash
python cosmoford/emulator/cot_fm.py \
    --config_yaml configs/unet_condition_large.yaml \
    --dataset_dir_nbody <path/to/neurips-wl-challenge-flat> \
    --dataset_dir_logn_train <path/to/GRF_HF> \
    --num_epochs 100

# Build the emulated HF dataset
python scripts/hf_emulated_dataset.py
```

---

## 🔧 Installation

```bash
pip install -e .
```

Requires Python ≥ 3.8. Key dependencies: `torch`, `lightning`, `diffusers`, `torchdyn`, `nflows`, `datasets`, `wandb`.

---

## 📦 Dataset loading

By default, datasets are loaded **locally** from `/project/rrg-lplevass/shared/wl_chall_data/` (on the Rorqual cluster). The expected directory structure is:

```
/project/rrg-lplevass/shared/wl_chall_data/
├── neurips-wl-challenge-flat/   # Main challenge dataset (train/validation splits)
├── lognormal/                   # LogNormal pretraining data
├── gowerstreet-train/           # Gower Street pretraining data
├── ot_emulated/                 # OT-emulated pretraining data
└── GRF_HF/                     # Gaussian Random Field pretraining data
```

To **load from HuggingFace Hub / GCS** instead (e.g. when running outside the cluster), set `use_hub: true` in your config:

```yaml
data:
  init_args:
    use_hub: true
```

To use a **different local directory**, set `data_dir`:

```yaml
data:
  init_args:
    data_dir: /path/to/your/datasets
```

All options can also be passed as CLI overrides:

```bash
# Default: just pick a dataset mode (loads locally from the default path)
trainer fit -c configs/experiments/efficientnet_v2_s.yaml --data.dataset_mode=lognormal

# Load from HuggingFace Hub
trainer fit -c configs/experiments/efficientnet_v2_s.yaml --data.use_hub=true

# Load from a custom local path
trainer fit -c configs/experiments/efficientnet_v2_s.yaml --data.data_dir=/scratch/datasets
```

Available `dataset_mode` values: `train`, `full`, `lognormal`, `gowerstreet`, `gowerstreet-train`, `ot_emulated`, `grf`.

---

## 👥 Team

| | | |
|---|---|---|
| [@AndreasTersenov](https://github.com/AndreasTersenov) | [@ASKabalan](https://github.com/ASKabalan) | [@b-remy](https://github.com/b-remy) |
| [@EiffL](https://github.com/EiffL) | [@noe-dia](https://github.com/noe-dia) | [@JuliaLinhart](https://github.com/JuliaLinhart) |
| [@Justinezgh](https://github.com/Justinezgh) | [@LaurencePeanuts](https://github.com/LaurencePeanuts) | [@SammyS15](https://github.com/SammyS15) |
| [@sachaguer](https://github.com/sachaguer) | [@rouzib](https://github.com/rouzib) | |

---

## 📝 License

See [LICENSE](LICENSE) file for details.
