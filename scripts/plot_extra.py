#!/usr/bin/env python
"""Calibration figure (TARP + SBC, HOS vs CNN) and the averaging diagnostic (FoM vs realizations
averaged) that explains the average-observation width. Matches the artifact palette."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = np.load("/home/tersenov/experiments/hos_night/extra_data.npz")
HOS_C, CNN_C = "#d1495b", "#2e77b5"
OUT = "/home/tersenov/experiments/hos_night"
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "axes.linewidth": 0.9})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def tarp_dev(e, a):
    return float(np.max(np.abs(np.asarray(e) - np.asarray(a))))


# ---------- calibration ----------
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.7))

ax = axs[0]
ax.plot([0, 1], [0, 1], color="0.6", lw=1.1, ls="--", zorder=0)
ax.plot(D["hos_alpha"], D["hos_ecp"], color=HOS_C, lw=2.6,
        label=f"HOS  (max dev {tarp_dev(D['hos_ecp'], D['hos_alpha']):.3f})")
ax.plot(D["cnn_alpha"], D["cnn_ecp"], color=CNN_C, lw=2.6,
        label=f"CNN  (max dev {tarp_dev(D['cnn_ecp'], D['cnn_alpha']):.3f})")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("credibility level"); ax.set_ylabel("expected coverage")
ax.set_title("TARP coverage", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=10.5, loc="upper left"); clean(ax)

for j, (name, ax) in enumerate([(r"$\Omega_m$", axs[1]), (r"$S_8$", axs[2])]):
    for ranks, c, lab in [(D["hos_ranks"], HOS_C, "HOS"), (D["cnn_ranks"], CNN_C, "CNN")]:
        n = len(ranks); nb = 20
        h, edges = np.histogram(ranks[:, j], bins=nb)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        ax.step(ctr, h / (n / nb), where="mid", color=c, lw=2.2, label=lab)
    ax.axhline(1.0, color="0.55", ls="--", lw=1.1)
    ax.fill_between([edges[0], edges[-1]], 1 - 3 * np.sqrt(nb) / np.sqrt(n) / (1),
                    1 + 3 * np.sqrt(nb) / np.sqrt(n), color="0.6", alpha=0.10)
    ax.set_ylim(0, 2); ax.set_xlabel("posterior rank of truth")
    ax.set_ylabel("density / uniform")
    ax.set_title(f"SBC rank uniformity — {name}", fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=10.5); clean(ax)

fig.suptitle("Both estimators are calibrated — joint coverage (TARP) and per-parameter rank uniformity (SBC)",
             fontsize=13.5, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/calibration.png", dpi=155, bbox_inches="tight", facecolor="white")
print("saved calibration.png")

# ---------- averaging diagnostic ----------
fig2, ax = plt.subplots(figsize=(7.6, 5.2))
hk, hf = D["hos_avg_k"], D["hos_avg_f"]
ck, cf = D["cnn_avg_k"], D["cnn_avg_f"]
ax.plot(ck, cf, "-o", color=CNN_C, lw=2.6, ms=8, mec="white", mew=1, label="CNN — 8-D summary")
ax.plot(hk, hf, "-o", color=HOS_C, lw=2.6, ms=8, mec="white", mew=1, label="HOS — 941-D + embedding")
ax.annotate("averaging → cleaner input than\nany single-realization training vector\n→ out-of-distribution for the HOS flow",
            xy=(10, hf[-1]), xytext=(4.4, 1180), color=HOS_C, fontsize=10.5, fontweight="bold",
            ha="left", arrowprops=dict(arrowstyle="->", color=HOS_C, lw=1.2))
ax.set_xlabel("noise realizations averaged into the observation")
ax.set_ylabel(r"Figure of merit")
ax.set_xticks([1, 2, 3, 5, 10]); ax.set_xlim(0.6, 10.6); ax.set_ylim(950, 1850)
ax.grid(True, axis="y", color="0.87", lw=0.6); ax.set_axisbelow(True); clean(ax)
ax.legend(frameon=False, fontsize=11.5, loc="center right")
ax.set_title("Why the average observation differs — an estimator effect, not the statistics",
             fontsize=12.8, fontweight="bold", loc="left", pad=12)
fig2.tight_layout()
fig2.savefig(f"{OUT}/avg_diag.png", dpi=160, bbox_inches="tight", facecolor="white")
print("saved avg_diag.png")
