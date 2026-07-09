#!/usr/bin/env python
"""Contour figures for the artifact: HOS champion vs biggest-budget CNN, on each of the 10 fiducial
observations + the average (~noiseless) observation. Custom KDE credible contours (68/95%)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

D = np.load("/home/tersenov/experiments/hos_night/contour_data.npz")
hos_per, hos_avg = D["hos_per"], D["hos_avg"]
cnn_per, cnn_avg = D["cnn_per"], D["cnn_avg"]
truth = D["truth"]
nobs = len(hos_per)
OUTDIR = "/home/tersenov/experiments/hos_night"

HOS_C, CNN_C = "#d1495b", "#2e77b5"   # crimson / steel-blue
plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})


def fom(samp):
    c = np.cov(samp.T); d = np.linalg.det(c)
    return 1.0 / np.sqrt(d) if d > 0 else np.nan


def kde_grid(samp, xi, yi):
    k = gaussian_kde(samp.T, bw_method=0.28)
    XX, YY = np.meshgrid(xi, yi)
    Z = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    dens = k(samp.T)
    l95, l68 = np.percentile(dens, 5), np.percentile(dens, 32)
    return XX, YY, Z, (l95, l68)


def draw(ax, samp, color, fill=True):
    lo = np.percentile(np.vstack([hos_per.reshape(-1, 2), cnn_per.reshape(-1, 2)]), 0.5, axis=0)
    hi = np.percentile(np.vstack([hos_per.reshape(-1, 2), cnn_per.reshape(-1, 2)]), 99.5, axis=0)
    xi = np.linspace(lo[0], hi[0], 200); yi = np.linspace(lo[1], hi[1], 200)
    XX, YY, Z, (l95, l68) = kde_grid(samp, xi, yi)
    if fill:
        ax.contourf(XX, YY, Z, levels=[l95, l68, Z.max()], colors=[color, color], alpha=0.22)
        ax.contourf(XX, YY, Z, levels=[l68, Z.max()], colors=[color], alpha=0.30)
    ax.contour(XX, YY, Z, levels=[l95, l68], colors=color, linewidths=1.3)


def truth_marker(ax):
    ax.axvline(truth[0], color="0.35", lw=0.7, ls="--", zorder=0)
    ax.axhline(truth[1], color="0.35", lw=0.7, ls="--", zorder=0)
    ax.plot(truth[0], truth[1], "*", color="black", ms=9, mec="white", mew=0.6, zorder=5)


legend = [Line2D([], [], color=HOS_C, lw=3, label="HOS (analytical) — FoM %.0f" % np.nanmean([fom(p) for p in hos_per])),
          Line2D([], [], color=CNN_C, lw=3, label="CNN (neural, 20200 sims) — FoM %.0f" % np.nanmean([fom(p) for p in cnn_per])),
          Line2D([], [], color="black", marker="*", ls="", ms=9, label="truth")]

# ---- per-observation grid ----
fig, axes = plt.subplots(2, 5, figsize=(16, 6.6), sharex=True, sharey=True)
for i, ax in enumerate(axes.ravel()):
    draw(ax, cnn_per[i], CNN_C); draw(ax, hos_per[i], HOS_C); truth_marker(ax)
    ax.set_title(f"observation {i}", fontsize=10)
    if i % 5 == 0:
        ax.set_ylabel(r"$S_8$")
    if i >= 5:
        ax.set_xlabel(r"$\Omega_m$")
fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.045), fontsize=11)
fig.suptitle("Posterior contours per fiducial observation — analytical HOS vs neural CNN (68% / 95%)",
             y=1.10, fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/contours_per_obs.png", dpi=155, bbox_inches="tight", facecolor="white")
print("saved contours_per_obs.png")

# ---- average (noiseless) observation ----
fig2, ax = plt.subplots(figsize=(6.2, 6.0))
draw(ax, cnn_avg, CNN_C); draw(ax, hos_avg, HOS_C); truth_marker(ax)
ax.set_xlabel(r"$\Omega_m$"); ax.set_ylabel(r"$S_8$")
ax.set_title("Average (noiseless) observation\nmean data vector over the 10 realizations",
             fontsize=12, weight="bold")
leg2 = [Line2D([], [], color=HOS_C, lw=3, label="HOS (analytical) — FoM %.0f" % fom(hos_avg)),
        Line2D([], [], color=CNN_C, lw=3, label="CNN (neural) — FoM %.0f" % fom(cnn_avg)),
        Line2D([], [], color="black", marker="*", ls="", ms=9, label="truth (%.3f, %.3f)" % (truth[0], truth[1]))]
ax.legend(handles=leg2, frameon=False, fontsize=10, loc="best")
fig2.tight_layout()
fig2.savefig(f"{OUTDIR}/contours_avg.png", dpi=160, bbox_inches="tight", facecolor="white")
print("saved contours_avg.png")

print("HOS per-obs FoM:", [round(fom(p)) for p in hos_per])
print("CNN per-obs FoM:", [round(fom(p)) for p in cnn_per])
print("avg-obs FoM  HOS %.0f  CNN %.0f" % (fom(hos_avg), fom(cnn_avg)))
