#!/usr/bin/env python
"""Figure of merit vs. compressor simulation budget: HOS (analytic) is flat, the CNN (learned)
must climb to reach it. Matches the artifact palette."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOS_C, CNN_C = "#d1495b", "#2e77b5"
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "axes.linewidth": 0.9})

b = np.array([100, 1000, 5000, 10000, 20200])
cnn = np.array([493, 996, 1460, 1668, 1707]); cnn_s = np.array([17, 98, 59, 206, 122])
hos, hos_s = 1672, 106

fig, ax = plt.subplots(figsize=(8.4, 5.3))
ax.axhspan(hos - hos_s, hos + hos_s, color=HOS_C, alpha=0.10, zorder=0)
ax.axhline(hos, color=HOS_C, lw=3.0, zorder=4, label="HOS (analytic) — no compression sims")
ax.fill_between(b, cnn - cnn_s, cnn + cnn_s, color=CNN_C, alpha=0.15, zorder=1)
ax.plot(b, cnn, "-o", color=CNN_C, lw=2.8, ms=8.5, zorder=5, mec="white", mew=1.1,
        label="CNN (learned)")

ax.set_xscale("log")
ax.set_xticks(b); ax.set_xticklabels([f"{x:,}" for x in b])
ax.set_xlabel("simulations used to train the compressor")
ax.set_ylabel(r"Figure of merit  (1/$\sqrt{\det\,\mathrm{Cov}}$)")
ax.set_ylim(380, 1900); ax.set_xlim(80, 26000)
ax.grid(True, which="major", axis="y", color="0.86", lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=13, loc="lower right")
ax.set_title("Learned vs. analytic summaries", fontsize=16, fontweight="bold", loc="left", pad=12)
fig.tight_layout()
fig.savefig("/home/tersenov/experiments/hos_night/budget_curve.png", dpi=170,
            bbox_inches="tight", facecolor="white")
print("saved budget_curve.png")
