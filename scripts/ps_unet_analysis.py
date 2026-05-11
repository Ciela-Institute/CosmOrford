#!/usr/bin/env python
"""
Power spectrum analysis: N-body vs LogNormal vs UNet.

Transforms InvestigateLogNormal.ipynb into a script and adds UNet ODE-based
map generation for comparison. For each of the 101 N-body cosmologies:
  - computes the mean power spectrum of the 100 N-body maps
  - computes the mean power spectrum of the 10 nearest lognormal maps
  - runs the UNet ODE on those lognormal maps and computes the resulting PS

Produces one figure per checkpoint (last / best) with two subplots:
  (P / P_nbody - 1) coloured by Omega_m and S8.

Usage:
    python ps_unet_analysis.py \\
        --run_name p1_emd \\
        --checkpoint_last <path/unet_FINAL.pth> \\
        --checkpoint_best <path/unet_best.pth> \\
        --config_yaml /lustre09/project/6091102/juzgh/CosmOrford/configs/unet_condition.yaml \\
        --dataset_nbody /path/to/neurips-wl-challenge-flat \\
        --dataset_lognormal /path/to/lognormal \\
        --outdir /lustre09/project/6091102/juzgh/cosmoford_exp/ps_analysis
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datasets import load_from_disk, load_dataset
from tqdm import tqdm

from cosmoford.dataset import reshape_field_numpy, inverse_reshape_field_numpy
from cosmoford.emulator.torch_models import build_unet2d_condition_with_y
from cosmoford.emulator.neural_ode import solve_ode_forward
from cosmoford import SURVEY_MASK

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True,
                    help="Experiment label, e.g. p1_emd or p1_sink_1e-3")
parser.add_argument("--checkpoint_last", type=str, required=True,
                    help="Path to the last checkpoint (.pth)")
parser.add_argument("--checkpoint_best", type=str, default=None,
                    help="Path to the best checkpoint (.pth); skipped if not set")
parser.add_argument("--config_yaml", type=str, required=True,
                    help="UNet YAML config file")
parser.add_argument("--dataset_nbody", type=str, required=True,
                    help="Path to neurips-wl-challenge-flat DatasetDict (load_from_disk)")
parser.add_argument("--dataset_lognormal", type=str, required=True,
                    help="Path to lognormal dataset (load_from_disk)")
parser.add_argument("--outdir", type=str,
                    default="/lustre09/project/6091102/juzgh/cosmoford_exp/ps_analysis")
parser.add_argument("--n_cosmo", type=int, default=101,
                    help="Number of distinct cosmologies in the N-body train split")
parser.add_argument("--n_logn_per_cosmo", type=int, default=10,
                    help="Number of nearest lognormal sims to use per cosmology")
parser.add_argument("--ode_batch_size", type=int, default=20,
                    help="Batch size for the ODE forward pass")
args = parser.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"figure.dpi": 150, "font.size": 12})

# ── Survey masks ───────────────────────────────────────────────────────────────
# Full domain (1424, 176) — used for PDF, matching the notebook exactly
mask_full = (SURVEY_MASK > 0)                                        # (1424, 176) bool
# Reshaped domain (1834, 88) — used for PS computation
mask_2d = np.concatenate(
    [SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]], axis=0
).astype(np.float32)   # (1834, 88)


# ── Power spectrum (ported from InvestigateLogNormal.ipynb) ───────────────────
def power_spectrum_batch(x, pixsize=2.0 / 60 / 180 * np.pi,
                         kedge=np.logspace(2, 4, 16)):
    """
    Azimuthally-averaged 2-D power spectrum for batched maps.
    x : torch.Tensor (batch, ny, nx)
    Returns (power_k, power) each of shape (batch, nk).
    """
    assert x.ndim == 3
    batch_size, ny, nx = x.shape
    device_ps = x.device
    dtype = x.dtype

    if not isinstance(kedge, torch.Tensor):
        kedge = torch.tensor(kedge, device=device_ps, dtype=dtype)
    else:
        kedge = kedge.to(device=device_ps, dtype=dtype)

    xk = torch.fft.rfft2(x)
    xk2 = (xk * xk.conj()).real

    ky = torch.fft.fftfreq(ny, d=pixsize, device=device_ps, dtype=dtype)
    kx = torch.fft.rfftfreq(nx, d=pixsize, device=device_ps, dtype=dtype)
    k = torch.sqrt(ky.reshape(-1, 1) ** 2 + kx.reshape(1, -1) ** 2) * 2 * np.pi

    index = torch.searchsorted(kedge, k.flatten()).reshape(ny, nx // 2 + 1)

    n_bins = len(kedge)
    nk = n_bins - 1

    xk2_flat = xk2.reshape(batch_size, -1)
    k_flat = k.flatten().unsqueeze(0).expand(batch_size, -1)
    index_flat = index.flatten().unsqueeze(0).expand(batch_size, -1)

    power = torch.zeros(batch_size, n_bins, device=device_ps, dtype=dtype)
    power_k = torch.zeros(batch_size, n_bins, device=device_ps, dtype=dtype)
    Nmode = torch.zeros(batch_size, n_bins, device=device_ps, dtype=dtype)

    for b in range(batch_size):
        power[b].index_add_(0, index_flat[b], xk2_flat[b])
        power_k[b].index_add_(0, index_flat[b], k_flat[b])
        Nmode[b].index_add_(0, index_flat[b], torch.ones_like(xk2_flat[b]))

    mirror_slice = slice(1, -1) if nx % 2 == 0 else slice(1, None)
    xk2_mirror = xk2[:, :, mirror_slice].reshape(batch_size, -1)
    k_mirror = k[:, mirror_slice].flatten().unsqueeze(0).expand(batch_size, -1)
    index_mirror = index[:, mirror_slice].flatten().unsqueeze(0).expand(batch_size, -1)
    for b in range(batch_size):
        power[b].index_add_(0, index_mirror[b], xk2_mirror[b])
        power_k[b].index_add_(0, index_mirror[b], k_mirror[b])
        Nmode[b].index_add_(0, index_mirror[b], torch.ones_like(xk2_mirror[b]))

    sel = Nmode > 0
    power[sel] = power[sel] / Nmode[sel]
    power_k[sel] = power_k[sel] / Nmode[sel]
    power_k = power_k[:, 1:nk + 1]

    power *= pixsize ** 2 / ny / nx
    power = power[:, 1:nk + 1]

    return power_k, power


def compute_mean_ps(maps_2d):
    """maps_2d: np.ndarray (B, H, W) — returns (l, mean_ps) as numpy."""
    t = torch.from_numpy(maps_2d.astype(np.float32))
    l, ps = power_spectrum_batch(t)
    return l[0].numpy(), ps.numpy()   # (nk,), (B, nk)


# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading N-body dataset …")
dset_nbody_raw = load_from_disk(args.dataset_nbody)
dset_nbody = dset_nbody_raw.with_format("numpy")
train_nbody = dset_nbody["train"]

print("Loading lognormal dataset …")
dset_logn = load_from_disk(args.dataset_lognormal).with_format("numpy")

# ── Infer theta/kappa layout for the lognormal dataset ────────────────────────
# The lognormal dataset may have kappa shape (B, H, W) or (B, 10, H, W).
# For 4D kappa, preprocess_batch uses theta[:, 1:] (drops first param) → y_dim=3.
# For 3D kappa, we use theta[:, :3].
_sample_kappa = np.array(dset_logn[0]["kappa"])
_logn_kappa_is_4d = _sample_kappa.ndim == 3 and False  # will check batch below
# Fetch a small batch to check ndim of batch kappa
_batch2 = dset_logn.select([0, 1])[:]
_batch_kappa = np.array(_batch2["kappa"])
_logn_kappa_is_4d = _batch_kappa.ndim == 4
_logn_theta_full = np.array(dset_logn["theta"])   # (N_logn, D)  — column-only load
print(f"Lognormal: {len(dset_logn)} sims, kappa ndim (batched)={_batch_kappa.ndim}, "
      f"theta shape={_logn_theta_full.shape}")


def get_logn_kappa_and_theta(indices):
    """
    Return reshaped kappa (len(indices), 1834, 88) and
    conditioning theta (len(indices), y_dim) for lognormal sims at `indices`.
    Handles both (B,H,W) and (B,10,H,W) kappa formats.
    """
    batch = dset_logn.select(indices.tolist())[:]
    kappa = np.array(batch["kappa"])   # (n, H, W) or (n, 10, H, W)
    theta = np.array(batch["theta"])   # (n, D)

    if kappa.ndim == 4:
        # Pick the first map per sim (index 0); theta drops first column
        kappa = kappa[:, 0, :, :]
        theta_cond = theta[:, 1:]      # (n, D-1) → should be y_dim=3
    else:
        # 3D kappa; keep first 3 theta columns
        theta_cond = theta[:, :3]

    kappa_reshaped = reshape_field_numpy(kappa)  # (n, 1834, 88)
    return kappa_reshaped, theta_cond


# ── N-body power spectra (per cosmology) + pixel histogram accumulation ────────
print(f"Computing N-body power spectra for {args.n_cosmo} cosmologies …")
ps_nbody_list = []   # each entry: (100, nk)
theta_cosmo = []     # each entry: [omega_c, S8]

# Pre-define shared histogram bins for the PDF (set after first batch)
PDF_BINS = 200
_nbody_hist_counts = None
_nbody_hist_bins = None

for i in tqdm(range(args.n_cosmo)):
    kappa_i = train_nbody["kappa"][i::args.n_cosmo]              # (100, 1424, 176)
    kappa_r = reshape_field_numpy(kappa_i) * mask_2d[None]       # (100, 1834, 88)
    l_arr, ps_i = compute_mean_ps(kappa_r)                       # (nk,), (100, nk)
    ps_nbody_list.append(ps_i)
    theta_cosmo.append(train_nbody["theta"][i][:2])              # [omega_c, S8]

    # Accumulate pixel histogram on full (1424,176) maps — matches notebook exactly
    kappa_full = train_nbody["kappa"][i::args.n_cosmo].astype(np.float32)  # (100,1424,176)
    px = kappa_full[:, mask_full].ravel()
    if _nbody_hist_bins is None:
        lo, hi = np.percentile(px, 0.1), np.percentile(px, 99.9)
        _nbody_hist_bins = np.linspace(lo, hi, PDF_BINS + 1)
        _nbody_hist_counts = np.zeros(PDF_BINS, dtype=np.float64)
    c, _ = np.histogram(px, bins=_nbody_hist_bins)
    _nbody_hist_counts += c

ps_nbody = np.stack(ps_nbody_list)    # (n_cosmo, 100, nk)
theta_arr = np.stack(theta_cosmo)     # (n_cosmo, 2)
l_arr = l_arr                          # (nk,)  wavenumber bin centres

# ── Lognormal power spectra (per cosmology) ────────────────────────────────────
print(f"Finding {args.n_logn_per_cosmo} nearest lognormal sims per cosmology …")

# Pre-compute distances in (omega_c, S8) space
# lognormal theta[:, 0] = omega_c,  theta[:, 1] = S8  (matches notebook convention)
logn_cosmo = _logn_theta_full[:, :2]   # (N_logn, 2)

all_logn_kappa = []     # will become (n_cosmo * n_logn_per_cosmo, 1834, 88)
all_logn_theta_cond = []
cosmo_of_sample = []    # which cosmology index each entry belongs to
nearest_indices_per_cosmo = []

for i in range(args.n_cosmo):
    d = np.sqrt(((logn_cosmo - theta_arr[i]) ** 2).sum(axis=1))
    indx = np.argsort(d)[:args.n_logn_per_cosmo]
    nearest_indices_per_cosmo.append(indx)
    kappa_r, theta_cond = get_logn_kappa_and_theta(indx)
    all_logn_kappa.append(kappa_r)
    all_logn_theta_cond.append(theta_cond)
    cosmo_of_sample.extend([i] * args.n_logn_per_cosmo)

all_logn_kappa = np.concatenate(all_logn_kappa, axis=0)          # (N_total, 1834, 88)
all_logn_theta_cond = np.concatenate(all_logn_theta_cond, axis=0) # (N_total, y_dim)
cosmo_of_sample = np.array(cosmo_of_sample)

# Lognormal PS per cosmology (before ODE)
print("Computing lognormal power spectra …")
ps_logn_list = []
logn_masked = all_logn_kappa * mask_2d[None]
_, ps_logn_all = compute_mean_ps(logn_masked)   # (N_total, nk)
for i in range(args.n_cosmo):
    sel = cosmo_of_sample == i
    ps_logn_list.append(ps_logn_all[sel])        # (n_logn, nk)
ps_logn = np.stack(ps_logn_list)                 # (n_cosmo, n_logn, nk)


# ── ODE inference for a checkpoint ────────────────────────────────────────────
def run_ode_on_lognormal(checkpoint_path, config):
    """Load UNet from checkpoint and run ODE on all lognormal maps."""
    unet = build_unet2d_condition_with_y(config).to(device)
    unet.load_state_dict(torch.load(checkpoint_path, map_location=device))
    unet.eval()
    print(f"  Loaded checkpoint: {checkpoint_path}")

    x0_all = torch.from_numpy(all_logn_kappa[:, None, :, :]).float()  # (N, 1, H, W)
    th_all = torch.from_numpy(all_logn_theta_cond.astype(np.float32))  # (N, y_dim)

    chunks = []
    n_total = len(x0_all)
    for start in tqdm(range(0, n_total, args.ode_batch_size), desc="  ODE batches"):
        x0_c = x0_all[start:start + args.ode_batch_size].to(device)
        th_c = th_all[start:start + args.ode_batch_size].to(device)
        with torch.no_grad():
            traj = solve_ode_forward(x0_c, unet, th_c, device)  # (T, bs, H, W)
        chunks.append(traj[-1])   # take t=1 end-point: (bs, H, W)
    return np.concatenate(chunks, axis=0)   # (N_total, H, W)


# ── PDF helper ────────────────────────────────────────────────────────────────
def plot_pdf_comparison(nbody_hist_counts, nbody_hist_bins,
                        logn_maps_reshaped, unet_maps_reshaped,
                        run_name, ckpt_label, outdir):
    """
    Pixel-value PDF matching the notebook exactly:
      - N-body: pre-accumulated on full (1424,176) maps with SURVEY_MASK
      - LogNormal / UNet: inverse-reshaped to (1424,176), then masked
      - hist(..., 100 bins, alpha=0.3)
    """
    bins = nbody_hist_bins  # 100 bins in full-map domain

    # Inverse-reshape lognormal and UNet maps to full domain, then mask
    logn_full = inverse_reshape_field_numpy(logn_maps_reshaped)   # (N,1424,176)
    unet_full = inverse_reshape_field_numpy(unet_maps_reshaped)   # (N,1424,176)
    px_ln = logn_full[:, mask_full].ravel().astype(np.float32)
    px_un = unet_full[:, mask_full].ravel().astype(np.float32)

    fig, ax = plt.subplots(figsize=(9, 5))
    # N-body: plot from pre-accumulated counts (same style as notebook hist)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = np.diff(bins)
    nb_density = nbody_hist_counts / (nbody_hist_counts.sum() * bin_widths)
    ax.bar(bin_centres, nb_density, width=bin_widths, alpha=0.3,
           color="C0", label="N-body")
    ax.hist(px_ln, bins=bins, density=True, alpha=0.3,
            color="C1", label="LogNormal (no ODE)")
    ax.hist(px_un, bins=bins, density=True, alpha=0.3,
            color="C2", label=f"UNet ODE ({ckpt_label})")
    ax.set_xlabel(r"$\kappa$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"{run_name} [{ckpt_label}]  —  pixel PDF", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fname = outdir / f"pdf_{run_name}_{ckpt_label}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Plotting helper ────────────────────────────────────────────────────────────
def plot_ps_comparison(l, ps_nbody, ps_logn, ps_unet,
                       theta_arr, run_name, ckpt_label, outdir):
    """
    Two-panel figure:
      left  — (P / P_nbody - 1) coloured by Omega_m
      right — (P / P_nbody - 1) coloured by S8

    Both lognormal (dashed) and UNet (solid) are overlaid.
    """
    p_nb = ps_nbody.mean(axis=1)   # (n_cosmo, nk)  mean over N-body realisations
    p_ln = ps_logn.mean(axis=1)    # (n_cosmo, nk)
    p_un = ps_unet                  # (n_cosmo, nk)

    ratio_ln = p_ln / p_nb - 1
    ratio_un = p_un / p_nb - 1

    for param_idx, param_name in [(0, "Omega_m"), (1, "S8")]:
        vals = theta_arr[:, param_idx]
        norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
        cmap = cm.viridis

        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(len(theta_arr)):
            color = cmap(norm(vals[i]))
            ax.semilogx(l, ratio_ln[i], color=color, alpha=0.25,
                        lw=1, linestyle="--")
            ax.semilogx(l, ratio_un[i], color=color, alpha=0.6,
                        lw=1.2, linestyle="-")

        # Legend proxies
        ax.plot([], [], "k--", lw=1.5, alpha=0.5, label="LogNormal (no ODE)")
        ax.plot([], [], "k-",  lw=1.5, label=f"UNet ODE ({ckpt_label})")

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(param_name, fontsize=12)

        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("$\\ell$", fontsize=13)
        ax.set_ylabel("$P / P_{\\rm N-body} - 1$", fontsize=13)
        ax.set_title(f"{run_name} [{ckpt_label}]  —  coloured by {param_name}",
                     fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout()

        fname = outdir / f"ps_{run_name}_{ckpt_label}_{param_name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")

    # Summary: mean ± 1σ comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(l, ratio_ln.mean(0), "b--", lw=2, label="LogNormal mean")
    ax.fill_between(l,
                    ratio_ln.mean(0) - ratio_ln.std(0),
                    ratio_ln.mean(0) + ratio_ln.std(0),
                    color="b", alpha=0.15, label="LogNormal ±1σ")
    ax.semilogx(l, ratio_un.mean(0), "r-", lw=2, label=f"UNet ODE mean ({ckpt_label})")
    ax.fill_between(l,
                    ratio_un.mean(0) - ratio_un.std(0),
                    ratio_un.mean(0) + ratio_un.std(0),
                    color="r", alpha=0.15, label="UNet ODE ±1σ")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("$\\ell$", fontsize=13)
    ax.set_ylabel("$P / P_{\\rm N-body} - 1$", fontsize=13)
    ax.set_title(f"{run_name} [{ckpt_label}]  —  mean ± 1σ over cosmologies", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fname = outdir / f"ps_{run_name}_{ckpt_label}_summary.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Load UNet config ───────────────────────────────────────────────────────────
with open(args.config_yaml) as f:
    unet_config = yaml.safe_load(f)
# Match sample_size to the reshaped field dimensions
unet_config["sample_size"] = [all_logn_kappa.shape[1], all_logn_kappa.shape[2]]

# ── Run for each checkpoint ────────────────────────────────────────────────────
checkpoints = [("last", args.checkpoint_last)]
if args.checkpoint_best is not None:
    checkpoints.append(("best", args.checkpoint_best))

for ckpt_label, ckpt_path in checkpoints:
    print(f"\n=== {args.run_name}  [{ckpt_label}]  checkpoint: {ckpt_path} ===")
    unet_maps = run_ode_on_lognormal(ckpt_path, unet_config)   # (N_total, H, W)
    unet_maps_masked = unet_maps * mask_2d[None]

    # PS per cosmology
    _, ps_unet_all = compute_mean_ps(unet_maps_masked)         # (N_total, nk)
    ps_unet_per_cosmo = np.stack([
        ps_unet_all[cosmo_of_sample == i].mean(axis=0)
        for i in range(args.n_cosmo)
    ])                                                          # (n_cosmo, nk)

    plot_ps_comparison(
        l_arr, ps_nbody, ps_logn, ps_unet_per_cosmo,
        theta_arr, args.run_name, ckpt_label, outdir
    )

    # Pixel PDF — N-body from accumulated histogram; logn/unet inverse-reshaped
    plot_pdf_comparison(
        nbody_hist_counts=_nbody_hist_counts,
        nbody_hist_bins=_nbody_hist_bins,
        logn_maps_reshaped=all_logn_kappa,   # unmasked reshaped; mask applied inside
        unet_maps_reshaped=unet_maps,        # unmasked reshaped; mask applied inside
        run_name=args.run_name,
        ckpt_label=ckpt_label,
        outdir=outdir,
    )

print("\nDone.")
