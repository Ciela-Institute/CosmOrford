"""Evaluate emulated dataset quality against real N-body maps at one cosmology.

Picks one random cosmology from the N-body validation set, loads all maps at
that cosmology, finds k_nearest emulated maps in (Omega_m, S8) space, and
compares power spectrum, pixel PDF, and HOS (wavelet peaks + L1 norms).

Usage:
    python scripts/eval_emulated_dataset.py \\
        --emulated_path /lustre09/.../emulated_logn_test1 \\
        --nbody_path    /home/.../neurips-wl-challenge-flat \\
        --outdir        /lustre09/.../emulated_logn_test1_eval \\
        --k_nearest     20 \\
        --seed          42
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk

from cosmoford import NOISE_STD, SURVEY_MASK
from cosmoford.summaries import (
    power_spectrum_batch,
    compute_wavelet_peaks_batch,
    compute_wavelet_l1_norms_batch,
)

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

PIXSIZE   = 2.0 / 60 / 180 * np.pi
KEDGE     = np.logspace(2, 4, 16)
K_CENTER  = np.sqrt(KEDGE[:-1] * KEDGE[1:])
MASK_FLAT = (SURVEY_MASK > 0).ravel()


def find_nearest_indices(theta_query, thetas_pool, k):
    """Indices of the k pool entries closest to theta_query[:2] in normalised L2."""
    q   = theta_query[:2].astype(np.float64)
    p   = thetas_pool[:, :2].astype(np.float64)
    std = p.std(axis=0).clip(min=1e-8)
    dists = np.sum(((p - q) / std) ** 2, axis=1)
    return np.argsort(dists)[:k]


def plot_ps(ps_nb, ps_em, theta, outdir):
    """Mean ± std power spectrum for N-body vs emulated."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    mean_nb = ps_nb.mean(0);  std_nb = ps_nb.std(0)
    mean_em = ps_em.mean(0);  std_em = ps_em.std(0)

    for ax, yscale in zip(axes, ["linear", "log"]):
        ax.fill_between(K_CENTER, mean_nb - std_nb, mean_nb + std_nb,
                        alpha=0.3, color="C0")
        ax.plot(K_CENTER, mean_nb, color="C0", label=f"N-body ({len(ps_nb)} maps)")
        ax.fill_between(K_CENTER, mean_em - std_em, mean_em + std_em,
                        alpha=0.3, color="C1")
        ax.plot(K_CENTER, mean_em, color="C1", label=f"Emulated ({len(ps_em)} maps)")
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$P(\ell)$")
        ax.set_title(f"Power spectrum ({yscale})")
        ax.legend()

    fig.suptitle(rf"$\Omega_m={theta[0]:.3f},\ S_8={theta[1]:.3f}$", y=1.01)
    fig.tight_layout()
    out = outdir / "ps_comparison.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_pdf(maps_nb, maps_em, theta, outdir):
    """Pixel PDF for N-body vs emulated, linear and log scale."""
    px_nb = maps_nb.reshape(len(maps_nb), -1)[:, MASK_FLAT].ravel()
    px_em = maps_em.reshape(len(maps_em), -1)[:, MASK_FLAT].ravel()

    bins = np.linspace(-0.15, 0.4, 200)
    bw   = bins[1] - bins[0]
    bc   = 0.5 * (bins[:-1] + bins[1:])
    h_nb, _ = np.histogram(px_nb, bins=bins)
    h_em, _ = np.histogram(px_em, bins=bins)
    pdf_nb = h_nb / (len(px_nb) * bw)
    pdf_em = h_em / (len(px_em) * bw)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, yscale in zip(axes, ["linear", "log"]):
        ax.stairs(pdf_nb, bins, fill=True, alpha=0.4, color="C0",
                  label=f"N-body ({len(maps_nb)} maps)")
        ax.stairs(pdf_em, bins, fill=True, alpha=0.4, color="C1",
                  label=f"Emulated ({len(maps_em)} maps)")
        ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("Density")
        ax.set_yscale(yscale)
        ax.set_title(f"Pixel PDF ({yscale})"); ax.legend()

    fig.suptitle(rf"$\Omega_m={theta[0]:.3f},\ S_8={theta[1]:.3f}$", y=1.01)
    fig.tight_layout()
    out = outdir / "pdf_comparison.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


def plot_hos(maps_nb, maps_em, theta, outdir):
    """Wavelet peak counts and L1 norms, mean ± std across maps."""
    t_nb = torch.from_numpy(maps_nb)
    t_em = torch.from_numpy(maps_em)

    peaks_nb = compute_wavelet_peaks_batch(t_nb, noise_std=NOISE_STD, normalize=False).cpu().numpy()
    peaks_em = compute_wavelet_peaks_batch(t_em, noise_std=NOISE_STD, normalize=False).cpu().numpy()
    l1_nb    = compute_wavelet_l1_norms_batch(t_nb, noise_std=NOISE_STD, normalize=False).cpu().numpy()
    l1_em    = compute_wavelet_l1_norms_batch(t_em, noise_std=NOISE_STD, normalize=False).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, nb, em, title in zip(
        axes,
        [peaks_nb, l1_nb],
        [peaks_em, l1_em],
        ["Wavelet peak counts", "Wavelet L1 norms"],
    ):
        x = np.arange(nb.shape[1])
        mn_nb = nb.mean(0);  sd_nb = nb.std(0)
        mn_em = em.mean(0);  sd_em = em.std(0)

        ax.plot(x, mn_nb, color="C0", label=f"N-body ({len(nb)} maps)")
        ax.fill_between(x, mn_nb - sd_nb, mn_nb + sd_nb, alpha=0.3, color="C0")
        ax.plot(x, mn_em, color="C1", label=f"Emulated ({len(em)} maps)")
        ax.fill_between(x, mn_em - sd_em, mn_em + sd_em, alpha=0.3, color="C1")
        ax.set_xlabel("Coefficient index"); ax.set_ylabel("Mean value")
        ax.set_title(title); ax.legend()

    fig.suptitle(rf"$\Omega_m={theta[0]:.3f},\ S_8={theta[1]:.3f}$", y=1.01)
    fig.tight_layout()
    out = outdir / "hos_comparison.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emulated_path", type=str,
                        default="/lustre09/project/6091102/juzgh/cosmoford_exp/emulated_datasets/emulated_logn_test1")
    parser.add_argument("--nbody_path", type=str,
                        default="/home/juzgh/links/projects/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-flat")
    parser.add_argument("--outdir", type=str,
                        default="/lustre09/project/6091102/juzgh/cosmoford_exp/emulated_datasets/emulated_logn_test1_eval")
    parser.add_argument("--k_nearest", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load thetas ──────────────────────────────────────────────────────────────
    print("Loading N-body thetas ...")
    ds_nb      = load_from_disk(str(args.nbody_path))["validation"]
    thetas_nb  = np.array(ds_nb["theta"], dtype=np.float32)
    unique_cosmo = np.unique(thetas_nb[:, :2].round(5), axis=0)
    print(f"  → {len(thetas_nb)} maps, {len(unique_cosmo)} unique cosmologies")

    print("Loading emulated thetas ...")
    ds_em     = load_from_disk(str(args.emulated_path))
    thetas_em = np.array(ds_em["theta"], dtype=np.float32)
    print(f"  → {len(thetas_em)} emulated maps")

    # ── Pick one cosmology at random ─────────────────────────────────────────────
    idx_cosmo = rng.integers(len(unique_cosmo))
    theta     = unique_cosmo[idx_cosmo]
    print(f"\nChosen cosmology: Ω_m={theta[0]:.4f}, S8={theta[1]:.4f}")

    # ── Load N-body maps at that cosmology ───────────────────────────────────────
    mask_nb   = np.all(thetas_nb[:, :2].round(5) == theta, axis=1)
    idx_nb    = np.where(mask_nb)[0].tolist()
    maps_nb   = np.array(ds_nb.select(idx_nb)["kappa"], dtype=np.float32)
    print(f"  N-body maps loaded: {maps_nb.shape}")

    # ── Find k nearest emulated maps ─────────────────────────────────────────────
    idx_em    = find_nearest_indices(theta, thetas_em, k=args.k_nearest).tolist()
    maps_em   = np.array(ds_em.select(idx_em)["maps"], dtype=np.float32)
    print(f"  Emulated maps loaded: {maps_em.shape}")

    # ── Compute and plot each summary ────────────────────────────────────────────
    print("\nComputing power spectra ...")
    _, ps_nb = power_spectrum_batch(torch.from_numpy(maps_nb), pixsize=PIXSIZE,
                                    kedge=KEDGE, normalize=False)
    _, ps_em = power_spectrum_batch(torch.from_numpy(maps_em), pixsize=PIXSIZE,
                                    kedge=KEDGE, normalize=False)
    plot_ps(ps_nb.cpu().numpy(), ps_em.cpu().numpy(), theta, outdir)

    print("Computing pixel PDFs ...")
    plot_pdf(maps_nb, maps_em, theta, outdir)

    print("Computing HOS ...")
    plot_hos(maps_nb, maps_em, theta, outdir)

    print(f"\nAll outputs saved to {outdir}")
