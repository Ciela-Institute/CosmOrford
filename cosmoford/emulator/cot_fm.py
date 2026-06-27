import argparse
from pathlib import Path
import os
from datasets import load_from_disk, load_dataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import ot
import wandb
from cosmoford.emulator.torch_models import build_unet2d_condition_with_y
from cosmoford.dataset import reshape_field_numpy
from cosmoford.emulator.utils import (
    preprocess_batch,
    split_rng,
    augmentation_data_numpy,
    apply_mask,
    iter_microbatches,
    pqm_evaluate,
    power_spectrum_distance,
    hos_peaks_distance,
    hos_l1_distance,
    scattering_distance,
)
from cosmoford.emulator.neural_ode import solve_ode_forward
from cosmoford import NOISE_STD
from cosmoford.summaries import (
    power_spectrum_batch,
    compute_wavelet_peaks_batch,
    compute_wavelet_l1_norms_batch,
)

plt.style.use("seaborn-v0_8")
# Improve default figure quality for logged images (crisper text and colorbars)
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
        # Larger fonts for readability after downscaling in W&B UI
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_config", type=str, required=True, help="Path to experiment YAML"
)
parser.add_argument(
    "--sim_budget",
    type=int,
    default=None,
    help="Number of N-body simulations to train on (null = full dataset)",
)
cli = parser.parse_args()

with open(cli.exp_config) as f:
    _cfg = yaml.safe_load(f)
_cfg.pop("exp_name", None)

_defaults = {"seed": 42, "dataset_dir_pqm_nbody": None}
_defaults.update(_cfg)
args = argparse.Namespace(**_defaults)

# Checkpoint-selection metric(s): best_ckpt_metric is either a single name
# (e.g. "val_loss", "pqm_chi2", "power_spectrum" -- backward compatible) or a
# {name: weight} mapping. The "best" checkpoint is the one minimizing the
# weighted mean of whichever of these metrics are available at each eval
# step, with weights chosen by the user (no automatic normalization, since
# metrics live on very different scales -- e.g. val_loss ~O(1) vs
# pqm_chi2 ~O(100)).
_raw_ckpt_metric = getattr(args, "best_ckpt_metric", "val_loss")
if isinstance(_raw_ckpt_metric, str):
    ckpt_metric_weights = {_raw_ckpt_metric: 1.0}
else:
    ckpt_metric_weights = dict(_raw_ckpt_metric)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

print("--Set Config--")

print("--Load Dataset--")

if args.dataset_dir_nbody is not None:
    dataset_nbody = load_from_disk(args.dataset_dir_nbody)
else:
    dataset_nbody = load_dataset("cosmostat/neurips-wl-challenge-flat")
dset_nbody = dataset_nbody.with_format("numpy")
test_dataset_nbody = dset_nbody["validation"]

# Subset N-body training set for simulation budget scan.
# Use first N samples (range) to match the compressor budget scan convention.
_train_nbody_full = dset_nbody["train"]
sim_budget = cli.sim_budget
if sim_budget is not None:
    _n_full = len(_train_nbody_full)
    train_dataset_nbody = _train_nbody_full.select(
        range(sim_budget), keep_in_memory=True
    ).with_format("numpy")
    print(f"N-body budget: {sim_budget} / {_n_full} sims")
else:
    train_dataset_nbody = _train_nbody_full

# Sorted N-body dataset for PQMass: 256 entries × (101 cosmologies, H, W).
# Each entry is one nuisance realization; the 101-axis gives unique cosmologies.
# For fully i.i.d. samples we draw an independent nuisance index per cosmology.
if args.dataset_dir_pqm_nbody is not None:
    pqm_nbody_dataset = load_from_disk(args.dataset_dir_pqm_nbody)["train"].with_format(
        "numpy"
    )
else:
    pqm_nbody_dataset = load_dataset(
        "cosmostat/neurips-wl-challenge", split="train"
    ).with_format("numpy")

dataset_lognormal = load_from_disk(args.dataset_dir_logn_train)
n = len(dataset_lognormal)
perm = np.random.default_rng(2).permutation(n).tolist()
n_test = int(0.2 * n)
train_dataset_lognormal = dataset_lognormal.select(
    perm[n_test:], keep_in_memory=True
).with_format("numpy")
test_dataset_lognormal = dataset_lognormal.select(
    perm[:n_test], keep_in_memory=True
).with_format("numpy")

print("train_dataset_lognormal", train_dataset_lognormal)
print("test_dataset_lognormal", test_dataset_lognormal)
print("train_dataset_pm", train_dataset_nbody)
print("test_dataset_pm", test_dataset_nbody)

# Use NumPy reshape helper to infer reduced field sizes without torch conversions
_kappa_sample = test_dataset_nbody[:10]["kappa"]  # (10, H, W) numpy
_kappa_rs = reshape_field_numpy(_kappa_sample)  # (10, H_red, W_red)
field_npix_x = _kappa_rs.shape[-1]
field_npix_y = _kappa_rs.shape[-2]
# Align y dimension at init with training-time y (lognormal theta[:,1:])
nb_of_params_to_infer = 3


# just because I cannot access Noe's dataset directly
def get_iterable_dataset(dataset, batch_size, seed):
    # Ensure shuffling writes indices to a writable directory under the W&B run folder
    split_cache_dir = os.path.join(str(run_dir), "hf_indices")
    os.makedirs(split_cache_dir, exist_ok=True)
    idx_cache = os.path.join(
        split_cache_dir, f"shuffle_{seed}_{len(dataset)}.idx.arrow"
    )
    return dataset.shuffle(seed=seed, indices_cache_file_name=idx_cache).iter(
        batch_size=batch_size, drop_last_batch=True
    )


print("--Set OT--")


def compute_cost_matrix(
    x0: torch.Tensor, x1: torch.Tensor, yx0: torch.Tensor, yx1: torch.Tensor, eps: float
) -> torch.Tensor:
    C_x = ot.dist(x0, x1)
    C_y = ot.dist(yx0, yx1)
    M = eps * C_x + C_y
    return M / M.max()


def get_paired_data(tp: np.ndarray, n_x0: int, rng: np.random.Generator):
    p = tp.flatten()
    p = p / p.sum()
    choices = rng.choice(tp.size, size=n_x0, replace=False, p=p)
    x0_paired, x1_paired = np.divmod(choices, tp.shape[1])
    return x0_paired, x1_paired


def sample_ot_plan(
    x0: np.ndarray,
    x1: np.ndarray,
    yx0: np.ndarray,
    yx1: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    device: torch.device,
    reg: float,
    ot_method: str = "sinkhorn",
):
    x0_t = torch.from_numpy(x0.reshape(x0.shape[0], -1)).float().to(device)
    x1_t = torch.from_numpy(x1.reshape(x1.shape[0], -1)).float().to(device)
    yx0_t = torch.from_numpy(np.array(yx0)).float().to(device)
    yx1_t = torch.from_numpy(np.array(yx1)).float().to(device)
    n, m = x0_t.shape[0], x1_t.shape[0]
    a = torch.full((n,), 1.0 / n, device=device, dtype=torch.float32)
    b = torch.full((m,), 1.0 / m, device=device, dtype=torch.float32)
    M = compute_cost_matrix(x0_t, x1_t, yx0_t, yx1_t, eps)
    if ot_method == "emd":
        # ot.emd requires CPU tensors (LP solver runs on CPU)
        tp = ot.emd(a.cpu(), b.cpu(), M.cpu()).to(device)
    else:
        # sinkhorn_log: log-domain arithmetic, numerically stable for small reg
        # numItermax=50000 ensures convergence at ot_reg=1e-3
        tp = ot.sinkhorn(a, b, M, reg=reg, method="sinkhorn_log", numItermax=50000)
    return get_paired_data(tp.cpu().numpy(), n, rng)


print("--Define get sample--")


def sample_time(n_samples: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n_samples, 1, device=device)


def get_ot_batch(
    batch,
    rng: np.random.Generator,
    eps: float,
    device: torch.device,
    ot_reg: float,
    ot_method: str = "sinkhorn",
):
    # Split parent RNG for independent, reproducible streams
    rng_pre, rng_x0, rng_x1, rng_plan = split_rng(rng, 4)

    batch = preprocess_batch(batch, rng_pre)
    batch_logn, batch_nbody = batch
    x0 = batch_logn["maps"]
    x1 = batch_nbody["maps"]
    theta_x0 = batch_logn["theta"]
    theta_x1 = batch_nbody["theta"]
    batch_size = len(batch_logn["theta"])

    t = sample_time(batch_size, device)
    # Augment before OT pairing

    x0, vmask0, hmask0 = augmentation_data_numpy(x0, rng_x0)
    x1, vmask1, hmask1 = augmentation_data_numpy(x1, rng_x1)

    inds_x0, inds_x1 = sample_ot_plan(
        x0, x1, theta_x0, theta_x1, rng_plan, eps, device, ot_reg, ot_method=ot_method
    )
    x0_paired = x0[inds_x0]
    x1_paired = x1[inds_x1]
    theta_x0_paired = theta_x0[inds_x0]
    theta_x1_paired = theta_x1[inds_x1]
    vmask1 = vmask1[inds_x1]
    hmask1 = hmask1[inds_x1]
    x0_paired = apply_mask(x0_paired, vmask1, hmask1)

    # Convert to torch tensors with desired shapes
    # Ensure channel-last shape then to torch (B, 1, H, W)
    if x0_paired.ndim == 3:
        x0_paired = x0_paired[..., None]
    if x1_paired.ndim == 3:
        x1_paired = x1_paired[..., None]
    x0_t = torch.from_numpy(x0_paired.transpose(0, 3, 1, 2)).float().to(device)
    x1_t = torch.from_numpy(x1_paired.transpose(0, 3, 1, 2)).float().to(device)
    theta_x0_t = torch.from_numpy(theta_x0_paired).float().to(device)
    theta_x1_t = torch.from_numpy(theta_x1_paired).float().to(device)
    return {
        "x0": x0_t,
        "x1": x1_t,
        "t": t,
        "theta_x0": theta_x0_t,
        "theta_x1": theta_x1_t,
    }


print("--Flow matching training fun--")


def compute_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return x1 - x0


def sample_conditional_pt(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma: float
) -> torch.Tensor:
    # Broadcast t: (B,1,1,1)
    t_b = t.view(-1, 1, 1, 1)
    x = t_b * x1 + (1.0 - t_b) * x0
    return x + sigma * torch.randn_like(x)


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # Sum over spatial dims, mean over batch
    return ((y_pred - y_true) ** 2).sum(dim=(2, 3)).mean()


def flow_matching_loss(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    yx0: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    v = compute_velocity(x0=x0, x1=x1)
    x_t = sample_conditional_pt(x0=x0, x1=x1, t=t, sigma=sigma)
    # model expects (B, C, H, W)
    t_in = t.view(-1)  # (B,)
    y_in = yx0  # (B, y_dim)
    # Provide encoder_hidden_states only if the UNet expects cross-attention
    enc = None
    cross_dim = getattr(model.config, "cross_attention_dim", None)
    if cross_dim is not None:
        enc = torch.zeros(x_t.size(0), 1, cross_dim, device=x_t.device, dtype=x_t.dtype)
    out = model(x_t, t_in, encoder_hidden_states=enc, y=y_in)
    v_pred = out.sample if hasattr(out, "sample") else out
    return mse_loss(v_pred, v)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x0: torch.Tensor,
    x1: torch.Tensor,
    yx0: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = flow_matching_loss(model, x0, x1, yx0, t, sigma)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


print("--Define the unet model--")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg_path = Path(args.config_yaml)
with open(cfg_path, "r") as f:
    config = yaml.safe_load(f)
config_source = str(cfg_path)

# Ensure sample_size matches the reshaped sizes (H, W)
config["sample_size"] = [field_npix_x, field_npix_y]
unet = build_unet2d_condition_with_y(config).to(device)

print("--Set optimizer--")
optimizer = Adam(unet.parameters(), lr=args.base_lr)
scheduler = ExponentialLR(optimizer, gamma=args.gamma)

print("--Init WandB--")

# Initialize Weights & Biases (always enabled)
run_suffix = f"{np.random.randint(100, 1000)}"
auto_run_name = f"emulator_training_{run_suffix}"
_base_run_name = args.wandb_run_name or auto_run_name
_run_name = (
    f"{_base_run_name}/budget_{sim_budget}"
    if sim_budget is not None
    else _base_run_name
)
wandb_kwargs = dict(
    project=args.wandb_project,
    name=_run_name,
    config={
        "eps": args.eps,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "sigma": args.sigma,
        "micro_batch_size": args.micro_batch_size,
        "base_lr": args.base_lr,
        "gamma": args.gamma,
        "ot_reg": args.ot_reg,
        "ot_method": args.ot_method,
        "sim_budget": sim_budget,
        "model_config": config,
        "config_source": config_source,
    },
)
if args.wandb_entity:
    wandb_kwargs["entity"] = args.wandb_entity
# mode can be "online" or "offline"
wandb_kwargs["mode"] = args.wandb_mode
run = wandb.init(**wandb_kwargs)
# Parameter counts
try:
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    wandb.summary["model/total_params"] = total_params
    wandb.summary["model/trainable_params"] = trainable_params
except Exception:
    pass
# Save code snapshot for reproducibility
try:
    # Log only the current module directory (emulator), not the whole cosmoford package
    wandb.run.log_code(root=str(Path(__file__).resolve().parent))
except Exception:
    pass

# Keep all run outputs inside the W&B run directory
run_dir = Path(wandb.run.dir).resolve()
fig_dir = run_dir / "fig"
ckpt_dir = run_dir / "checkpoints"

# Ensure the chosen directories exist
fig_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Persist resolved config to the W&B run directory and log it as an artifact
resolved_cfg_path = run_dir / "config_used.yaml"
try:
    with open(resolved_cfg_path, "w") as f:
        yaml.safe_dump(config, f)
    # Update wandb config with the saved file path for reference
    try:
        wandb.config.update(
            {"config_file": str(resolved_cfg_path)}, allow_val_change=True
        )
    except Exception:
        pass
    # Log config artifact
    cfg_art = wandb.Artifact("unet-config", type="config")
    cfg_art.add_file(str(resolved_cfg_path))
    wandb.log_artifact(cfg_art)
except Exception:
    pass

print("--Map-based evaluation (power spectrum / PDF / HOS)--")

_PIXSIZE_PS  = 2.0 / 60 / 180 * np.pi
_KEDGE_PS    = np.logspace(2, 4, 16)
_K_CENTER_PS = np.sqrt(_KEDGE_PS[:-1] * _KEDGE_PS[1:])
_K_NEAR_EVAL = 50  # lognormal maps to generate per eval step

# Cache N-body flat-validation thetas so we don't re-read them every eval step
_thetas_nb_all   = None
_unique_cosmo_nb = None


def _get_nb_cosmologies():
    global _thetas_nb_all, _unique_cosmo_nb
    if _thetas_nb_all is None:
        _thetas_nb_all   = np.array(test_dataset_nbody["theta"], dtype=np.float32)
        _unique_cosmo_nb = np.unique(_thetas_nb_all[:, :2].round(5), axis=0)
    return _thetas_nb_all, _unique_cosmo_nb


def _run_map_eval(model, label, seed, fig_path, key_prefix, extra_log=None):
    """Per-cosmology map evaluation.

    1. Pick one random N-body cosmology from the flat validation set.
    2. Load all ~56 N-body maps at that cosmology.
    3. Find _K_NEAR_EVAL nearest lognormal maps in (Omega_m, S8) space.
    4. Generate emulated maps by running those through the ODE.
    5. Compare power spectrum, pixel PDF, and HOS (peaks + L1 norms).
    6. Return scalar distances for checkpoint selection.
    """
    try:
        rng_eval = np.random.default_rng(seed)

        # ── 1. Pick one N-body cosmology ─────────────────────────────────────
        thetas_nb_all, unique_cosmo = _get_nb_cosmologies()
        idx_cosmo = rng_eval.integers(len(unique_cosmo))
        theta_ref = unique_cosmo[idx_cosmo]   # (2,)  [Omega_m, S8]

        # ── 2. Load all N-body maps at that cosmology (~56) ──────────────────
        mask_nb = np.all(thetas_nb_all[:, :2].round(5) == theta_ref, axis=1)
        idx_nb  = np.where(mask_nb)[0].tolist()
        kappa_nb = np.array(
            test_dataset_nbody.select(idx_nb)["kappa"], dtype=np.float32
        )  # (n_nb, H, W)
        maps_ref = reshape_field_numpy(kappa_nb)  # (n_nb, H_red, W_red)

        # ── 3. Find k nearest lognormal maps in (Omega_m, S8) ───────────────
        thetas_logn = np.array(test_dataset_lognormal["theta"], dtype=np.float32)
        thetas_logn_2d = thetas_logn[:, :2]
        std_logn = thetas_logn_2d.std(axis=0).clip(min=1e-8)
        dists    = np.sum(((thetas_logn_2d - theta_ref) / std_logn) ** 2, axis=1)
        idx_near = np.argsort(dists)[:_K_NEAR_EVAL].tolist()

        batch_logn  = test_dataset_lognormal.select(idx_near)
        kappa_logn  = np.array(batch_logn["kappa"], dtype=np.float32)  # (k, H, W)
        theta_logn  = thetas_logn[idx_near]                             # (k, 3)

        # ── 4. Generate maps via ODE ─────────────────────────────────────────
        x0_logn = reshape_field_numpy(kappa_logn)               # (k, H_red, W_red)
        x0_t    = torch.from_numpy(x0_logn[:, None]).float()    # (k, 1, H_red, W_red)
        theta_t = torch.from_numpy(theta_logn[:, :2]).float()   # (k, 2)

        gen_chunks = []
        for start in range(0, x0_t.shape[0], micro_bs):
            with torch.no_grad():
                pred = solve_ode_forward(
                    x0_t[start : start + micro_bs].to(device),
                    model,
                    theta_t[start : start + micro_bs].to(device),
                    device,
                )
            gen_chunks.append(pred[-1])   # (chunk, H_red, W_red) numpy
        maps_gen = np.concatenate(gen_chunks, axis=0)  # (k, H_red, W_red)

        metrics = {}
        log     = {}
        stem    = fig_path.stem
        title   = rf"$\Omega_m={theta_ref[0]:.3f},\ S_8={theta_ref[1]:.3f}$ — {label}"

        # ── 5a. Power spectrum ───────────────────────────────────────────────
        _, ps_nb  = power_spectrum_batch(
            torch.from_numpy(maps_ref), pixsize=_PIXSIZE_PS, kedge=_KEDGE_PS, normalize=False
        )
        _, ps_gen = power_spectrum_batch(
            torch.from_numpy(maps_gen), pixsize=_PIXSIZE_PS, kedge=_KEDGE_PS, normalize=False
        )
        ps_nb_np  = ps_nb.cpu().numpy()
        ps_gen_np = ps_gen.cpu().numpy()

        fig_ps, ax = plt.subplots(figsize=(7, 4))
        mn_nb, sd_nb   = ps_nb_np.mean(0),  ps_nb_np.std(0)
        mn_gen, sd_gen = ps_gen_np.mean(0), ps_gen_np.std(0)
        ax.fill_between(_K_CENTER_PS, mn_nb - sd_nb,  mn_nb + sd_nb,  alpha=0.3, color="C0")
        ax.plot(_K_CENTER_PS, mn_nb,  color="C0", label=f"N-body ({len(maps_ref)})")
        ax.fill_between(_K_CENTER_PS, mn_gen - sd_gen, mn_gen + sd_gen, alpha=0.3, color="C1")
        ax.plot(_K_CENTER_PS, mn_gen, color="C1", label=f"Generated ({len(maps_gen)})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$P(\ell)$")
        ax.set_title(title); ax.legend()
        fig_ps.tight_layout()
        ps_path = fig_path.parent / f"ps_{stem}.png"
        fig_ps.savefig(ps_path, dpi=150); plt.close(fig_ps)
        log[f"{key_prefix}/ps_plot"] = wandb.Image(str(ps_path), caption=f"PS {label}")

        # ── 5b. Pixel PDF ────────────────────────────────────────────────────
        bins = np.linspace(-0.15, 0.4, 150)
        bw   = bins[1] - bins[0]
        px_nb  = maps_ref.ravel()
        px_gen = maps_gen.ravel()
        h_nb,  _ = np.histogram(px_nb,  bins=bins)
        h_gen, _ = np.histogram(px_gen, bins=bins)
        pdf_nb  = h_nb  / (len(px_nb)  * bw)
        pdf_gen = h_gen / (len(px_gen) * bw)

        fig_pdf, axes_pdf = plt.subplots(1, 2, figsize=(12, 4))
        for ax, yscale in zip(axes_pdf, ["linear", "log"]):
            ax.stairs(pdf_nb,  bins, fill=True, alpha=0.4, color="C0",
                      label=f"N-body ({len(maps_ref)})")
            ax.stairs(pdf_gen, bins, fill=True, alpha=0.4, color="C1",
                      label=f"Generated ({len(maps_gen)})")
            ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("Density")
            ax.set_yscale(yscale); ax.set_title(f"Pixel PDF ({yscale})"); ax.legend()
        fig_pdf.suptitle(title)
        fig_pdf.tight_layout()
        pdf_path = fig_path.parent / f"pdf_{stem}.png"
        fig_pdf.savefig(pdf_path, dpi=150); plt.close(fig_pdf)
        log[f"{key_prefix}/pdf_plot"] = wandb.Image(str(pdf_path), caption=f"PDF {label}")

        # ── 5c. HOS (wavelet peaks + L1 norms) ──────────────────────────────
        peaks_nb  = compute_wavelet_peaks_batch(
            torch.from_numpy(maps_ref), noise_std=NOISE_STD, normalize=False
        ).cpu().numpy()
        peaks_gen = compute_wavelet_peaks_batch(
            torch.from_numpy(maps_gen), noise_std=NOISE_STD, normalize=False
        ).cpu().numpy()
        l1_nb  = compute_wavelet_l1_norms_batch(
            torch.from_numpy(maps_ref), noise_std=NOISE_STD, normalize=False
        ).cpu().numpy()
        l1_gen = compute_wavelet_l1_norms_batch(
            torch.from_numpy(maps_gen), noise_std=NOISE_STD, normalize=False
        ).cpu().numpy()

        fig_hos, axes_hos = plt.subplots(1, 2, figsize=(12, 4))
        for ax, nb, gen, ttl in zip(
            axes_hos,
            [peaks_nb, l1_nb], [peaks_gen, l1_gen],
            ["Wavelet peak counts", "Wavelet L1 norms"],
        ):
            x = np.arange(nb.shape[1])
            mn, sd = nb.mean(0), nb.std(0)
            mg, sg = gen.mean(0), gen.std(0)
            ax.plot(x, mn, color="C0", label=f"N-body ({len(nb)})")
            ax.fill_between(x, mn - sd, mn + sd, alpha=0.3, color="C0")
            ax.plot(x, mg, color="C1", label=f"Generated ({len(gen)})")
            ax.fill_between(x, mg - sg, mg + sg, alpha=0.3, color="C1")
            ax.set_xlabel("Coefficient index"); ax.set_ylabel("Mean value")
            ax.set_title(ttl); ax.legend()
        fig_hos.suptitle(title)
        fig_hos.tight_layout()
        hos_path = fig_path.parent / f"hos_{stem}.png"
        fig_hos.savefig(hos_path, dpi=150); plt.close(fig_hos)
        log[f"{key_prefix}/hos_plot"] = wandb.Image(str(hos_path), caption=f"HOS {label}")

        # ── 6. Scalar distances for checkpoint selection ─────────────────────
        if ckpt_metric_weights.get("power_spectrum", 0):
            metrics["power_spectrum"] = power_spectrum_distance(maps_ref, maps_gen)
            log[f"{key_prefix}/power_spectrum_dist"] = metrics["power_spectrum"]
            print(f"Power spectrum [{label}]: {metrics['power_spectrum']:.4g}")

        if ckpt_metric_weights.get("hos_peaks", 0):
            metrics["hos_peaks"] = hos_peaks_distance(maps_ref, maps_gen)
            log[f"{key_prefix}/hos_peaks_dist"] = metrics["hos_peaks"]
            print(f"HOS peaks [{label}]: {metrics['hos_peaks']:.4g}")

        if ckpt_metric_weights.get("hos_l1", 0):
            metrics["hos_l1"] = hos_l1_distance(maps_ref, maps_gen)
            log[f"{key_prefix}/hos_l1_dist"] = metrics["hos_l1"]
            print(f"HOS L1-norms [{label}]: {metrics['hos_l1']:.4g}")

        if ckpt_metric_weights.get("scattering", 0):
            metrics["scattering"] = scattering_distance(maps_ref, maps_gen)
            log[f"{key_prefix}/scattering_dist"] = metrics["scattering"]
            print(f"Scattering [{label}]: {metrics['scattering']:.4g}")

        if extra_log:
            log.update(extra_log)
        if log:
            wandb.log(log)
        return metrics
    except Exception as e:
        print(f"Map eval [{label}] failed: {e}")
        import traceback; traceback.print_exc()
        return {}


print("--Training--")

eps = args.eps
max_steps = args.max_steps
batch_size = min(args.batch_size, len(train_dataset_nbody))
sigma = args.sigma
micro_bs = int(args.micro_batch_size)

num_training_steps_total = max_steps
nb_checkpoints = num_training_steps_total // 5
pqm_step_interval = max(1, num_training_steps_total // max(1, args.n_eval_stats))
step = 0
latest_metrics: dict[str, float] = {}
best_combined_score = float("inf")

n_val_samples = 500


def _run_validation_loss(model, n_samples, seed):
    """Validation MSE over a fixed held-out set of n_samples, chunked through
    the model in micro_bs pieces to avoid OOM."""
    rng_eval = np.random.default_rng(seed)
    ds_test_lognormal = get_iterable_dataset(test_dataset_lognormal, n_samples, 1)
    ds_test_nbody = get_iterable_dataset(test_dataset_nbody, n_samples, 0)
    batch_lognormal = next(ds_test_lognormal)
    batch_nbody = next(ds_test_nbody)
    batch = get_ot_batch(
        [batch_lognormal, batch_nbody],
        rng_eval,
        eps,
        device,
        args.ot_reg,
        ot_method=args.ot_method,
    )

    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for mb in iter_microbatches(batch, micro_bs):
            n_mb = mb["x0"].shape[0]
            loss_mb = flow_matching_loss(
                model, mb["x0"], mb["x1"], mb["theta_x0"], mb["t"], sigma
            )
            total_loss += float(loss_mb.detach().cpu().item()) * n_mb
            total_n += n_mb
    return total_loss / total_n


def _save_best_ckpt():
    best_ckpt = str(ckpt_dir / "unet_best.pth")
    torch.save(unet.state_dict(), best_ckpt)
    try:
        art = wandb.Artifact("unet-best", type="model")
        art.add_file(best_ckpt)
        wandb.log_artifact(art)
    except Exception:
        pass


def _maybe_update_best(epoch):
    """Weighted mean of whichever ckpt_metric_weights entries are currently
    known (lower is better for val_loss / pqm_chi2 / power_spectrum); save a
    new best checkpoint whenever this combined score improves."""
    global best_combined_score
    used = {k: v for k, v in latest_metrics.items() if ckpt_metric_weights.get(k, 0)}
    if not used:
        return
    score = float(
        np.average(list(used.values()), weights=[ckpt_metric_weights[k] for k in used])
    )
    wandb.log({"ckpt/combined_score": score, "epoch": epoch})
    if score < best_combined_score:
        best_combined_score = score
        _save_best_ckpt()


epoch = 0
pbar = tqdm(total=max_steps, desc="training")
while step < max_steps:
    ds_train_logn = get_iterable_dataset(
        train_dataset_lognormal, batch_size, int((epoch + 1) * 1000)
    )
    ds_train_nbody = get_iterable_dataset(train_dataset_nbody, batch_size, epoch)
    rng_epoch = np.random.default_rng(args.seed + epoch)

    for batch_logn, batch_nbody in zip(ds_train_logn, ds_train_nbody):
        batch = get_ot_batch(
            [batch_logn, batch_nbody],
            rng_epoch,
            eps,
            device,
            args.ot_reg,
            ot_method=args.ot_method,
        )

        for mb in iter_microbatches(batch, micro_bs):
            if step >= max_steps:
                break
            loss = train_step(
                unet,
                optimizer,
                mb["x0"],
                mb["x1"],
                mb["theta_x0"],
                mb["t"],
                sigma,
            )
            step += 1
            pbar.update(1)
            wandb.log(
                {
                    "train_loss": loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )
            if step % 500 == 0:
                # basic scheduler step per epoch
                scheduler.step()

            if step % nb_checkpoints == 0:
                ds_test_lognormal = get_iterable_dataset(
                    test_dataset_lognormal, micro_bs, 0
                )
                ds_test_nbody = get_iterable_dataset(test_dataset_nbody, micro_bs, 0)
                batch_lognormal = next(ds_test_lognormal)
                batch_nbody = next(ds_test_nbody)
                batch_test = get_ot_batch(
                    [batch_lognormal, batch_nbody],
                    rng_epoch,
                    eps,
                    device,
                    args.ot_reg,
                    ot_method=args.ot_method,
                )

                x_1_pred = solve_ode_forward(
                    batch_test["x0"], unet, batch_test["theta_x0"], device
                )

                # Higher DPI for crisper images in W&B; moderate figsize to reduce browser downscale blur
                fig = plt.figure(figsize=(12, 6), dpi=300, constrained_layout=True)
                # Prepare numpy arrays for plotting (avoid cuda tensors in np.concatenate)
                x0_np = batch_test["x0"][0].detach().cpu().squeeze(0).numpy()  # (H,W)
                x1_np = batch_test["x1"][0].detach().cpu().squeeze(0).numpy()  # (H,W)
                pred_np = x_1_pred[-1, 0]  # (H,W) numpy
                # Use consistent color scaling across panels to aid visual comparison
                all_vals = np.concatenate(
                    [x0_np.ravel(), pred_np.ravel(), x1_np.ravel()]
                )
                vmin, vmax = np.percentile(all_vals, [1, 99])

                ax1 = fig.add_subplot(5, 1, 1)
                im1 = ax1.imshow(x0_np.T, cmap="viridis", vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im1, ax=ax1)
                cbar.ax.tick_params(labelsize=12)
                ax1.set_title("Beginning of ODE", fontsize=16)
                ax1.axis("off")

                ax2 = fig.add_subplot(5, 1, 2)
                im2 = ax2.imshow(pred_np.T, cmap="viridis", vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im2, ax=ax2)
                cbar.ax.tick_params(labelsize=12)
                ax2.set_title("End of ODE", fontsize=16)
                ax2.axis("off")

                ax3 = fig.add_subplot(5, 1, 3)
                im3 = ax3.imshow(x1_np.T, cmap="viridis", vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im3, ax=ax3)
                cbar.ax.tick_params(labelsize=12)
                ax3.set_title("Truth", fontsize=16)
                ax3.axis("off")

                ax4 = fig.add_subplot(5, 1, 4)
                im4 = ax4.imshow((pred_np - x1_np).T, cmap="viridis")
                cbar = fig.colorbar(im4, ax=ax4)
                cbar.ax.tick_params(labelsize=12)
                ax4.set_title("Residuals", fontsize=16)
                ax4.axis("off")

                ax5 = fig.add_subplot(5, 1, 5)
                im5 = ax5.imshow((x0_np - pred_np).T, cmap="viridis")
                cbar = fig.colorbar(im5, ax=ax5)
                cbar.ax.tick_params(labelsize=12)
                ax5.set_title("Learned correction", fontsize=16)
                ax5.axis("off")

                # Save to disk at high resolution to avoid any in-memory backend inconsistencies
                img_path = fig_dir / f"fm_diag_epoch_{epoch}.png"
                fig.savefig(img_path, dpi=300, bbox_inches="tight")
                log_dict = {
                    "fm_diagnostics": wandb.Image(
                        str(img_path), caption=f"fm_diagnostics epoch {epoch}"
                    ),
                    "epoch": epoch,
                }
                try:
                    wandb.log(log_dict)
                except Exception:
                    pass
                plt.close(fig)

            # Validation loss + map-based eval (PQMass / power spectrum) at the
            # same interval, so all requested metrics are available together
            # for weighted-combination best-checkpoint tracking.
            if step > 0 and step % pqm_step_interval == 0:
                val_loss = _run_validation_loss(unet, n_val_samples, args.seed + 1234)
                wandb.log(
                    {
                        "val_loss": val_loss,
                        "train_loss_epoch": float(loss),
                        "epoch": epoch,
                    }
                )
                latest_metrics["val_loss"] = val_loss

                latest_metrics.update(
                    _run_map_eval(
                        unet,
                        f"step {step}",
                        step,
                        fig_dir / f"pqm_unet_step_{step}.png",
                        "eval/unet",
                        extra_log={"epoch": epoch},
                    )
                )
                _maybe_update_best(epoch)

    epoch += 1
pbar.close()

# Final trained model
try:
    final_ckpt = str(ckpt_dir / "unet_FINAL.pth")
    torch.save(unet.state_dict(), final_ckpt)
    art = wandb.Artifact("unet-final", type="model")
    art.add_file(final_ckpt)
    wandb.log_artifact(art)
except Exception:
    pass

# Post-training map eval: evaluate both the last and the combined-score-best checkpoint
_run_map_eval(unet, "last", 42, fig_dir / "pqm_final_last.png", "eval/last")

best_ckpt_path = ckpt_dir / "unet_best.pth"
if best_ckpt_path.exists():
    unet_best = build_unet2d_condition_with_y(config).to(device)
    unet_best.load_state_dict(torch.load(str(best_ckpt_path), map_location=device))
    unet_best.eval()
    _run_map_eval(unet_best, "best", 42, fig_dir / "pqm_final_best.png", "eval/best")

wandb.finish()
