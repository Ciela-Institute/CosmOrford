"""
Test GRF generation with three different cosmologies to verify
the power spectrum of generated maps matches theory for each.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_lognormal_dataset"))
from get_hsc_redshift_distribution import get_redshift_distribution
from projector import projector_func
import astropy.units as u

from create_grf_dataset import compute_convergence_cl, draw_grf, correct_kappa

NSIDE = 2048
CAMB_LMAX = 4000
RESO = 2
N_REAL = 5  # realisations per cosmology for averaging

nz_path = os.path.join(os.path.dirname(__file__), "..", "..", "nz.fits")
z, dndz = get_redshift_distribution(nz_path)

h = 0.7
Omega_bh2 = 0.0224

cosmologies = {
    "fiducial": {"little_h": h, "Omega_m": 0.3, "S_8": 0.8, "Omega_b": Omega_bh2 / h**2},
    "high Om, high S8": {"little_h": h, "Omega_m": 0.55, "S_8": 0.95, "Omega_b": Omega_bh2 / h**2},
    "low Om, low S8": {"little_h": h, "Omega_m": 0.1, "S_8": 0.65, "Omega_b": Omega_bh2 / h**2},
}
colors = {"fiducial": "k", "high Om, high S8": "r", "low Om, low S8": "b"}

# --- Compute C_l and measure from realisations ---
cls_theory = {}
cls_measured_avg = {}
for name, params in cosmologies.items():
    print(f"Computing C_l for '{name}': Om={params['Omega_m']}, S8={params['S_8']}")
    cl = compute_convergence_cl(params, z, dndz, lmax=CAMB_LMAX)
    cls_theory[name] = cl

    print(f"  Drawing {N_REAL} realisations and measuring power spectra...")
    cl_avg = np.zeros(CAMB_LMAX + 1)
    for i in range(N_REAL):
        kappa_i = draw_grf(cl, nside=NSIDE, seed=1000 * hash(name) % 10000 + i)
        cl_avg += hp.anafast(kappa_i, lmax=CAMB_LMAX)
    cls_measured_avg[name] = cl_avg / N_REAL

# --- Plot: theory vs measured for each cosmology ---
ell = np.arange(CAMB_LMAX + 1)
fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
for name in cosmologies:
    c = colors[name]
    ax.loglog(ell[2:], cls_theory[name][2:], color=c, lw=2, label=f"{name} (theory)")
    ax.loglog(ell[2:], cls_measured_avg[name][2:], color=c, lw=1, ls="--", alpha=0.7,
              label=f"{name} (mean {N_REAL} reals)")
ax.set_xlabel("$\\ell$")
ax.set_ylabel("$C_\\ell^{\\kappa\\kappa}$")
ax.set_title("Theory vs measured power spectra for 3 cosmologies")
ax.legend(fontsize=8, ncol=2)

# Ratio plot
ax2 = axes[1]
for name in cosmologies:
    c = colors[name]
    cl_th = cls_theory[name][2:]
    cl_me = cls_measured_avg[name][2:]
    valid = cl_th > 0
    ratio = np.ones_like(cl_me)
    ratio[valid] = cl_me[valid] / cl_th[valid]
    ax2.semilogx(ell[2:], ratio, color=c, lw=1, label=name)
ax2.axhline(1, color="gray", ls="--")
ax2.set_ylim(0.85, 1.15)
ax2.set_xlabel("$\\ell$")
ax2.set_ylabel("Measured / Theory")
ax2.legend(fontsize=8)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), "grf_two_cosmo_cls.png")
plt.savefig(outpath, dpi=150)
print(f"Power spectrum comparison saved to {outpath}")

# --- Draw maps and extract one patch each ---
fig, axes_map = plt.subplots(len(cosmologies), 1, figsize=(20, 3 * len(cosmologies)),
                              gridspec_kw={"hspace": 0.5})
rng = np.random.default_rng(42)
lon, lat = rng.uniform(0, 360), rng.uniform(-90, 90)

for j, (name, cl) in enumerate(cls_theory.items()):
    kappa = draw_grf(cl, nside=NSIDE, seed=42)
    kappa_corr = correct_kappa(kappa)
    patch = projector_func(
        kappa_corr,
        patch_center=(lon, lat),
        patch_shape=(1424, 176),
        reso=RESO * u.arcmin.to(u.degree),
        proj_mode="gnomonic",
    ).T

    params = cosmologies[name]
    vmax = 0.04  # fixed scale for comparison
    im = axes_map[j].imshow(patch.T, vmin=-vmax, vmax=vmax, cmap="RdBu_r", origin="lower")
    axes_map[j].set_title(f"{name}: $\\Omega_m$={params['Omega_m']}, $S_8$={params['S_8']}, "
                          f"std={patch.std():.5f}")
    axes_map[j].set_xlabel("x [pixels]")
    axes_map[j].set_ylabel("y [pixels]")
    plt.colorbar(im, ax=axes_map[j], fraction=0.01, pad=0.005, label="$\\kappa$")

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), "grf_two_cosmo_patches.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Patch comparison saved to {outpath}")

print("\nDone!")
