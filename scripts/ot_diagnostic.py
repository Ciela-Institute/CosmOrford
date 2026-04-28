"""
Phase -1: Pre-training OT pairing diagnostic.

Evaluates OT coupling quality between lognormal and N-body datasets for
different (eps, ot_method, ot_reg) WITHOUT any UNet training.

Metrics for each configuration:
  r_omega_c, r_s8   — Pearson correlation of soft expected paired cosmological
                       parameters (T_norm @ theta_x1). Inflated for diffuse plans.
  delta_Omega_c,
  delta_S8           — E_{(i,j)~T}[|theta_x0[i] - theta_x1[j]|]: expected cosmo mismatch
                       under the plan. Uniform T → same as random pairing baseline.
                       Permutation T → true per-pair mismatch. Compare to random baseline.
  mean_displacement  — mean ||x0 - x1||^2 for paired maps
  plan_entropy       — entropy H(T); log(n)=6.2 = permutation, 2*log(n)=12.4 = uniform

Also reports the "natural eps" = E[C_y] / E[C_x], the scale at which
spatial and cosmological costs contribute equally to the OT objective.

Usage:
    python ot_diagnostic.py \
        --dataset_dir_nbody /path/to/nbody \
        --dataset_dir_logn  /path/to/lognormal \
        --output_dir        /path/to/output \
        --batch_size        500
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import ot
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from datasets import load_from_disk

from cosmoford.dataset import reshape_field_numpy
from cosmoford.emulator.utils import preprocess_batch

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir_nbody", type=str, required=True)
parser.add_argument("--dataset_dir_logn",  type=str, required=True)
parser.add_argument("--output_dir",        type=str, default="ot_diagnostic_output")
parser.add_argument("--batch_size",        type=int, default=500)
parser.add_argument("--seed",              type=int, default=42)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
print("Loading datasets...")
dataset_nbody = load_from_disk(args.dataset_dir_nbody).with_format("numpy")
dataset_logn  = load_from_disk(args.dataset_dir_logn).with_format("numpy")

val_nbody = dataset_nbody["validation"]

n_logn = len(dataset_logn)
perm    = np.random.default_rng(2).permutation(n_logn).tolist()
n_test  = int(0.2 * n_logn)
val_logn = dataset_logn.select(perm[:n_test])

bs = min(args.batch_size, len(val_nbody), len(val_logn))
idx_nbody = rng.choice(len(val_nbody), size=bs, replace=False).tolist()
idx_logn  = rng.choice(len(val_logn),  size=bs, replace=False).tolist()

batch_nbody_raw = val_nbody.select(idx_nbody)[:]
batch_logn_raw  = val_logn.select(idx_logn)[:]

batch_logn_proc, batch_nbody_proc = preprocess_batch(
    [batch_logn_raw, batch_nbody_raw], rng
)

x0       = batch_logn_proc["maps"]   # (B, H, W)
x1       = batch_nbody_proc["maps"]  # (B, H, W)
theta_x0 = np.array(batch_logn_proc["theta"])[:, :2]   # (B, 2) Omega_c, S8
theta_x1 = np.array(batch_nbody_proc["theta"])[:, :2]  # (B, 2)

print(f"Batch size: {bs}  |  map shape: {x0.shape[1:]}")

# ---------------------------------------------------------------------------
# Cost matrices (GPU)
# ---------------------------------------------------------------------------
print("Computing cost matrices on GPU...")
x0_t  = torch.from_numpy(x0.reshape(bs, -1)).float().to(device)
x1_t  = torch.from_numpy(x1.reshape(bs, -1)).float().to(device)
yx0_t = torch.from_numpy(theta_x0).float().to(device)
yx1_t = torch.from_numpy(theta_x1).float().to(device)

C_x = ot.dist(x0_t, x1_t).cpu().numpy()   # (B, B)
C_y = ot.dist(yx0_t, yx1_t).cpu().numpy() # (B, B)

natural_eps = float(C_y.mean() / C_x.mean())
print(f"\nNatural eps = E[C_y] / E[C_x] = {natural_eps:.4f}")
print(f"  E[C_x] = {C_x.mean():.6f}  (spatial cost)")
print(f"  E[C_y] = {C_y.mean():.6f}  (cosmo cost)")
print(f"  At eps = natural_eps, both terms contribute equally to the OT objective.\n")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(T: np.ndarray) -> dict:
    """Compute pairing quality metrics from a transport plan T (n x m)."""
    T = np.asarray(T, dtype=np.float64)

    # Row-normalise: p(x1 | x0_i)
    T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-15)

    # Soft: expected paired cosmo params (inflated for diffuse T)
    theta_paired = T_norm @ theta_x1         # (B, 2)
    r_oc, _ = pearsonr(theta_x0[:, 0], theta_paired[:, 0])
    r_s8, _ = pearsonr(theta_x0[:, 1], theta_paired[:, 1])

    # Expected delta: E_{(i,j)~T}[|theta_x0[i] - theta_x1[j]|]
    # Same structure as mean_disp but with |Δθ| instead of ||x0-x1||²
    # For uniform T → same as random pairing. For permutation T → true per-pair mismatch.
    abs_diff_oc = np.abs(theta_x0[:, 0:1] - theta_x1[:, 0])  # (B, B)
    abs_diff_s8 = np.abs(theta_x0[:, 1:2] - theta_x1[:, 1])  # (B, B)
    delta_oc = float(np.sum(T * abs_diff_oc))
    delta_s8 = float(np.sum(T * abs_diff_s8))

    # Weighted spatial displacement
    mean_disp = float(np.sum(T * C_x))

    # Plan entropy (log(n)=permutation, 2*log(n)=uniform for n samples)
    flat = T.ravel()
    mask = flat > 1e-15
    entropy = float(-np.sum(flat[mask] * np.log(flat[mask])))

    return dict(r_omega_c=r_oc, r_s8=r_s8,
                delta_omega_c=delta_oc, delta_s8=delta_s8,
                mean_displacement=mean_disp, entropy=entropy)

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
eps_values    = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.8]
ot_reg_values = [0.001, 0.005, 0.01, 0.1, 0.3]

a = np.ones(bs) / bs
b = np.ones(bs) / bs
a_t = torch.from_numpy(a).float()
b_t = torch.from_numpy(b).float()

results = []

# --- Random pairing baseline ---
# Uniform T = 1/n² for all (i,j) → E[|Δθ|] = mean over all pairs
T_uniform = np.ones((bs, bs)) / bs**2  # joint uniform, total mass=1, consistent with T
abs_diff_oc = np.abs(theta_x0[:, 0:1] - theta_x1[:, 0])  # (B, B)
abs_diff_s8 = np.abs(theta_x0[:, 1:2] - theta_x1[:, 1])  # (B, B)
delta_oc_rand = float(np.sum(T_uniform * abs_diff_oc))
delta_s8_rand = float(np.sum(T_uniform * abs_diff_s8))
print(f"\nRandom pairing baseline: delta_Ωc = {delta_oc_rand:.4f}  delta_S8 = {delta_s8_rand:.4f}")
print(f"  (any good OT plan should have smaller deltas than this)\n")

HDR = (f"{'method':>10} {'eps':>8} {'ot_reg':>10} | "
       f"{'r_Ωc':>7} {'r_S8':>7} {'δΩc':>7} {'δS8':>7} {'disp':>12} {'H(T)':>8}")
print(HDR)
print("-" * len(HDR))

for eps in eps_values:
    M     = eps * C_x + C_y
    M     = M / M.max()
    M_t   = torch.from_numpy(M).float()

    # --- Sinkhorn (log-domain for numerical stability at small reg) ---
    for reg in ot_reg_values:
        try:
            T = ot.sinkhorn(a_t, b_t, M_t, reg=reg,
                            method='sinkhorn_log', numItermax=10000).numpy()
            if np.isnan(T).any() or T.sum() < 0.1:
                print(f"{'sinkhorn':>10} {eps:>8.4f} {reg:>10.2e} | DEGENERATE (NaN or zero T)")
                continue
            m = compute_metrics(T)
            results.append(dict(method="sinkhorn", eps=eps, ot_reg=reg, **m))
            print(f"{'sinkhorn':>10} {eps:>8.4f} {reg:>10.2e} | "
                  f"{m['r_omega_c']:>7.3f} {m['r_s8']:>7.3f} "
                  f"{m['delta_omega_c']:>7.4f} {m['delta_s8']:>7.4f} "
                  f"{m['mean_displacement']:>12.4f} {m['entropy']:>8.3f}")
        except Exception as e:
            print(f"{'sinkhorn':>10} {eps:>8.4f} {reg:>10.2e} | FAILED: {e}")

    # --- EMD ---
    try:
        T = ot.emd(a, b, M)
        m = compute_metrics(T)
        results.append(dict(method="emd", eps=eps, ot_reg=None, **m))
        print(f"{'emd':>10} {eps:>8.4f} {'—':>10} | "
              f"{m['r_omega_c']:>7.3f} {m['r_s8']:>7.3f} "
              f"{m['delta_omega_c']:>7.4f} {m['delta_s8']:>7.4f} "
              f"{m['mean_displacement']:>12.4f} {m['entropy']:>8.3f}")
    except Exception as e:
        print(f"{'emd':>10} {eps:>8.4f} {'—':>8} | FAILED: {e}")

# ---------------------------------------------------------------------------
# Save table
# ---------------------------------------------------------------------------
df = pd.DataFrame(results)
csv_path = output_dir / "ot_diagnostic.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.ravel()

metrics_info = [
    ("r_omega_c",        "Pearson r(Ω_c)  [soft]",            "higher is better — but inflated for diffuse T"),
    ("r_s8",             "Pearson r(S8)  [soft]",             "higher is better — but inflated for diffuse T"),
    ("delta_omega_c",    "Mean |ΔΩ_c|  [hard argmax]",        f"lower is better; random={delta_oc_rand:.4f}"),
    ("delta_s8",         "Mean |ΔS8|  [hard argmax]",         f"lower is better; random={delta_s8_rand:.4f}"),
    ("mean_displacement","Mean spatial displacement ‖x0−x1‖²","lower is better"),
    ("entropy",          "Plan entropy H(T)",                  f"log(n)={np.log(bs):.2f}=permutation  2·log(n)={2*np.log(bs):.2f}=uniform"),
]

palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(ot_reg_values)))
colors_reg = {reg: palette[i] for i, reg in enumerate(ot_reg_values)}

for ax, (metric, title, hint) in zip(axes, metrics_info):
    for reg in ot_reg_values:
        sub = df[(df["method"] == "sinkhorn") & (df["ot_reg"] == reg)]
        if sub.empty:
            continue
        ax.plot(sub["eps"], sub[metric], marker="o",
                color=colors_reg[reg], label=f"sinkhorn reg={reg}")

    sub_emd = df[df["method"] == "emd"]
    ax.plot(sub_emd["eps"], sub_emd[metric],
            marker="s", linestyle="--", color="black", label="emd")

    ax.axvline(natural_eps, color="red", linestyle=":",
               label=f"natural eps={natural_eps:.3f}")

    if metric == "delta_omega_c":
        ax.axhline(delta_oc_rand, color="gray", linestyle="--", label=f"random={delta_oc_rand:.4f}")
    elif metric == "delta_s8":
        ax.axhline(delta_s8_rand, color="gray", linestyle="--", label=f"random={delta_s8_rand:.4f}")

    ax.set_xlabel("eps (log scale)")
    ax.set_xscale("log")
    ax.set_title(f"{title}\n({hint})", fontsize=10)
    ax.legend(fontsize=7)

plt.suptitle("OT Pairing Diagnostic — no UNet training", fontsize=13)
plt.tight_layout()
fig_path = output_dir / "ot_diagnostic.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Figure saved to {fig_path}")
