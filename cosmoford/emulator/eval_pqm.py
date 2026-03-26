"""Standalone PQMass evaluation script.

Compares two sets of maps using PQMass. Works with any dataset that has a
`kappa` field. The candidate can optionally be passed through the UNet emulator.

Usage — N-body vs LogNormal/Gower Street etc (raw, no emulation):
    python eval_pqm.py \
        --dataset_ref  <path/to/neurips-wl-challenge-flat> \
        --dataset_cand <path/to/lognormal> \
        --label_ref "N-body" --label_cand "LogNormal" \
        --outdir pqm_eval

Usage — N-body vs UNet emulated maps:
    python eval_pqm.py \
        --dataset_ref  <path/to/neurips-wl-challenge-flat> \
        --dataset_cand <path/to/lognormal> \
        --checkpoint   <path/to/unet_best.pth> \
        --config_yaml  <path/to/unet_condition_large.yaml> \
        --label_ref "N-body" --label_cand "UNet" \
        --outdir pqm_eval

"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from datasets import load_from_disk

from cosmoford.dataset import reshape_field_numpy
from cosmoford.emulator.torch_models import build_unet2d_condition_with_y
from cosmoford.emulator.neural_ode import solve_ode_forward
from cosmoford.emulator.utils import pqm_evaluate, sample_one_per_cosmology

plt.style.use("seaborn-v0_8")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_ref",    type=str, required=True,  help="Path to reference dataset (e.g. N-body)")
parser.add_argument("--dataset_cand",   type=str, required=True,  help="Path to candidate dataset")
parser.add_argument("--split_ref",      type=str, default="validation", help="Dataset split to use for reference")
parser.add_argument("--split_cand",     type=str, default=None,   help="Dataset split to use for candidate (default: use full dataset shuffled)")
parser.add_argument("--label_ref",      type=str, default="N-body", help="Label for the reference dataset")
parser.add_argument("--label_cand",     type=str, default="Candidate", help="Label for the candidate dataset")
parser.add_argument("--checkpoint",     type=str, default=None,   help="UNet checkpoint (.pth). If set, candidate maps are passed through the ODE emulator.")
parser.add_argument("--config_yaml",    type=str, default=None,   help="UNet YAML config (required if --checkpoint is set)")
parser.add_argument("--n_samples",      type=int, default=500,    help="Number of maps per distribution")
parser.add_argument("--ode_batch_size", type=int, default=20,     help="Batch size for ODE forward pass (to avoid GPU OOM)")
parser.add_argument("--num_refs",       type=int, default=100,    help="PQMass reference cells")
parser.add_argument("--re_tessellation", type=int, default=30,    help="Number of PQMass re-tessellations")
parser.add_argument("--outdir",         type=str, default="pqm_eval")
parser.add_argument("--seed",           type=int, default=42)
parser.add_argument("--one_per_cosmo_ref", action="store_true",
                    help="Sample exactly one realization per unique cosmology from the "
                         "reference dataset (required for the NeurIPS WL Challenge N-body "
                         "data which has 101 cosmologies × 256 realizations).")
args = parser.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

rng = np.random.default_rng(args.seed)
N = args.n_samples


def load_maps(dataset_path, split, n, rng, second_split=False):
    """Load N random maps from any dataset with a `kappa` field.

    Handles both formats:
      - (B, H, W)       — e.g. N-body
      - (B, 10, H, W)   — e.g. lognormal (one map per sim picked at random)

    If the dataset has named splits (e.g. train/validation) and `split` is set,
    that split is used. Otherwise the full dataset is treated as one pool.

    `second_split=True` draws a different random subset (different seed) so
    two independent batches can be drawn from the same dataset (self-test).
    """
    dset = load_from_disk(dataset_path)

    if hasattr(dset, 'keys') and split in dset:
        pool = dset[split].with_format("numpy")
    else:
        # No named splits — use the whole dataset, hold out 20% as "validation"
        flat = dset.with_format("numpy")
        n_total = len(flat)
        seed = 3 if second_split else 2
        perm = np.random.default_rng(seed).permutation(n_total).tolist()
        n_val = int(0.2 * n_total)
        pool = flat.select(perm[:n_val])

    seed = rng.integers(0, 2**32)
    idx = np.random.default_rng(seed).choice(len(pool), size=min(n, len(pool)), replace=False).tolist()
    batch = pool.select(idx)[:]

    kappa = np.array(batch["kappa"])
    # (B, 10, H, W) → pick one map per sim
    if kappa.ndim == 4:
        i = rng.integers(0, kappa.shape[1])
        kappa = kappa[:, i, :, :]
    # kappa is now (B, H, W) — reshape to reduced field
    maps = reshape_field_numpy(kappa)  # (B, H_red, W_red)

    theta = np.array(batch["theta"]) if "theta" in batch else None
    return maps, theta


# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading reference dataset...")
if args.one_per_cosmo_ref:
    # NeurIPS WL Challenge N-body: 101 cosmologies × 256 realizations. Only one realization per cosmology may be used to satisfy the i.i.d.
    dset_ref = load_from_disk(args.dataset_ref)
    split_pool = dset_ref[args.split_ref] if hasattr(dset_ref, 'keys') and args.split_ref in dset_ref else dset_ref
    maps_ref, _ = sample_one_per_cosmology(split_pool.with_format("numpy"), rng)
    print(f"  one_per_cosmo_ref: {len(maps_ref)} cosmologies found — using {len(maps_ref)} i.i.d. samples")
    if args.num_refs >= len(maps_ref):
        import warnings
        warnings.warn(
            f"num_refs={args.num_refs} >= n_cosmo={len(maps_ref)}. "
            f"Reducing num_refs to {len(maps_ref) // 2} to leave enough samples per cell.",
            stacklevel=1,
        )
        args.num_refs = len(maps_ref) // 2
else:
    maps_ref, _ = load_maps(args.dataset_ref, args.split_ref, N, rng)

print("Loading candidate dataset...")
split_cand = args.split_cand or args.split_ref
# Use a different random draw when both datasets are the same path (self-test)
same_dataset = (args.dataset_ref == args.dataset_cand) and (split_cand == args.split_ref)
maps_cand, theta_cand = load_maps(args.dataset_cand, split_cand, N, rng, second_split=same_dataset)

# ── Optionally run UNet ODE ────────────────────────────────────────────────────
if args.checkpoint is not None:
    assert theta_cand is not None, "Candidate dataset must have a `theta` field to condition the UNet."
    print("Loading UNet and running ODE forward pass...")
    with open(args.config_yaml) as f:
        config = yaml.safe_load(f)
    config["sample_size"] = [maps_ref.shape[-1], maps_ref.shape[-2]]
    unet = build_unet2d_condition_with_y(config).to(device)
    unet.load_state_dict(torch.load(args.checkpoint, map_location=device))
    unet.eval()

    # Slice theta to 3 params to match training convention
    theta_use = theta_cand[:, [0, 1, -1][:theta_cand.shape[1]]]

    x0_all = torch.from_numpy(maps_cand[:, None, :, :]).float()   # (N, 1, H, W)
    theta_all = torch.from_numpy(theta_use).float()
    chunks = []
    for start in range(0, len(x0_all), args.ode_batch_size):
        x0_chunk = x0_all[start:start + args.ode_batch_size].to(device)
        theta_chunk = theta_all[start:start + args.ode_batch_size].to(device)
        with torch.no_grad():
            x_pred = solve_ode_forward(x0_chunk, unet, theta_chunk, device)
        chunks.append(x_pred[-1])  # (chunk, H, W)
    maps_cand = np.concatenate(chunks, axis=0)  # (N, H, W)

# ── PQMass ─────────────────────────────────────────────────────────────────────
print(f"Running PQMass: {args.label_ref} vs {args.label_cand}...")
# Move maps_ref to Torch and put it on cuda
torch_map_refs = torch.tensor(maps_ref, device=device)
torch_maps_cand = torch.tensor(maps_cand, device=device)
chi2_vals, fig = pqm_evaluate(torch_map_refs, torch_maps_cand, num_refs=args.num_refs, re_tessellation=args.re_tessellation)
fig.suptitle(f"PQMass — {args.label_ref} vs {args.label_cand}", fontsize=13)

out_path = outdir / f"pqm_{args.label_ref.replace(' ', '_')}_vs_{args.label_cand.replace(' ', '_')}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Summary ────────────────────────────────────────────────────────────────────
dof = args.num_refs - 1
print(f"\n{'─'*50}")
print(f"Expected χ²(dof={dof}) mean = {dof:.1f}")
print(f"{args.label_ref} vs {args.label_cand}  mean χ² = {np.mean(chi2_vals):.2f}  (std={np.std(chi2_vals):.2f})")
print(f"{'─'*50}")
print(f"Figure saved to {out_path}")
