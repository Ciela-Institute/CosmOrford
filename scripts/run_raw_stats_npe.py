#!/usr/bin/env python3
"""Standalone RAW-analytical-summary NPE runner (new-NDE baseline).

Feeds the FULL analytical statistics vector (``compressor._compute_stats(x)`` —
e.g. ~320-D for L1, higher for HOS+scattering) straight to a conditional MAF, i.e.
with NO learned 8-D compression, then computes FoM + posterior samples at the
fiducial cosmology. Output format matches scripts/run_npe_budget_scan.py so that
scripts/analyze_hos_conditioning_sweep.py can plot raw and compressed side by side.

This is deliberately a SEPARATE script (chosen over editing the pipeline) so that
Justine's canonical run_npe_budget_scan.py stays untouched — it dropped
``use_raw_stats_context`` and hardcodes ``context_dim=8``. We reuse
``cosmoford.models_nopatch.build_flow`` (which accepts an arbitrary context_dim) and
mirror the pipeline's holdout/fiducial prep exactly: reshape_field, NO extra noise
(holdout maps are already noisy), and the same 90/10 fixed split + best-seed
selection. Inference is single-best-seed ("best" mode), matching our recommendation.

Usage:
  python scripts/run_raw_stats_npe.py \
      --checkpoints_path $HOME/experiments/budget_scan/checkpoints/raw_l1_p5 \
      --budget 20200 \
      --holdout_path /project/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout \
      --out $HOME/experiments/hos_cond_sweep/results/raw_l1_best_ctx0/budget-20200
"""
import argparse
import json
import re
from glob import glob
from pathlib import Path

import numpy as np
import torch

from datasets import load_from_disk
from cosmoford import THETA_MEAN, THETA_STD
from cosmoford.dataset import reshape_field_numpy
from cosmoford.models_nopatch import StatsCompressorNoPatch, build_flow


def find_best_checkpoint(checkpoints_path: str, budget: int) -> str:
    """Lowest val_(log_prob|nll) checkpoint, step tie-break; else last.ckpt.

    Mirrors run_npe_budget_scan.find_best_checkpoint so we pick the same file.
    """
    cdir = Path(checkpoints_path) / f"budget-{budget}"
    ckpts = glob(str(cdir) + "/**/*.ckpt", recursive=True)
    cands = []  # (loss, step, path)
    for c in ckpts:
        if Path(c).name == "last.ckpt":
            continue
        m = re.search(r"val_(?:log_prob|nll)=([-+]?\d*\.?\d+)", c)
        sm = re.search(r"(?:step|epoch)=(\d+)", c)
        if m:
            cands.append((float(m.group(1)), int(sm.group(1)) if sm else 0, c))
    if cands:
        best_loss = min(c[0] for c in cands)
        tied = [c for c in cands if c[0] == best_loss]
        return max(tied, key=lambda c: c[1])[2]
    last = cdir / "last.ckpt"
    if last.exists() and last.stat().st_size > 0:
        return str(last)
    raise FileNotFoundError(f"No checkpoint under {cdir}")


def raw_context(compressor, kappa_reshaped, device):
    """Full analytical stats vector for one (1834, 88) map -> (1, raw_dim)."""
    x = torch.from_numpy(kappa_reshaped.astype(np.float32)).unsqueeze(0).to(device)
    return compressor._compute_stats(x)


def precompute_contexts(compressor, kappa_all, device):
    ctxs = []
    with torch.no_grad():
        for i in range(len(kappa_all)):
            kr = reshape_field_numpy(kappa_all[i][np.newaxis])[0]  # (1834, 88)
            ctxs.append(raw_context(compressor, kr, device).cpu())
            if (i + 1) % 1000 == 0:
                print(f"  context {i+1}/{len(kappa_all)}")
    return torch.cat(ctxs, dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None, help="direct .ckpt path (overrides checkpoints_path)")
    ap.add_argument("--checkpoints_path", default=None, help="root containing budget-N/")
    ap.add_argument("--budget", type=int, default=20200)
    ap.add_argument("--holdout_path", required=True)
    ap.add_argument("--out", required=True, help="results dir (gets results.json + *.npy)")
    ap.add_argument("--npe_seeds", type=int, default=10)
    ap.add_argument("--npe_epochs", type=int, default=100)
    ap.add_argument("--npe_lr", type=float, default=1e-4)
    ap.add_argument("--npe_batch_size", type=int, default=128)
    ap.add_argument("--npe_patience", type=int, default=50)
    ap.add_argument("--n_transforms", type=int, default=5)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--n_posterior_samples", type=int, default=10000)
    ap.add_argument("--context_normalization", action="store_true",
                    help="z-score the context on the train split — recommended for the "
                         "high-dim raw stats vector, whose dims have heterogeneous scales")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.checkpoint or find_best_checkpoint(args.checkpoints_path, args.budget)
    print(f"Loading compressor (for _compute_stats + normalization): {ckpt}")
    compressor = StatsCompressorNoPatch.load_from_checkpoint(ckpt, map_location=device)
    compressor.eval().to(device)
    for p in compressor.parameters():
        p.requires_grad = False

    dd = load_from_disk(args.holdout_path)

    # ── holdout train -> raw context / theta ──
    train = dd["train"].with_format("numpy")
    kappa_all = np.array(train["kappa"])
    theta_all = np.array(train["theta"])
    theta_norm = ((theta_all[:, :2] - THETA_MEAN[:2]) / THETA_STD[:2]).astype(np.float32)
    print(f"Holdout train: {len(kappa_all)} maps; computing raw contexts...")
    S = precompute_contexts(compressor, kappa_all, device)
    T = torch.from_numpy(theta_norm)
    raw_dim = S.shape[1]
    print(f"raw context dim = {raw_dim}, N = {S.shape[0]}")

    # ── fixed 90/10 split (seed 0), same convention as the pipeline ──
    n = S.shape[0]
    n_val = n // 10
    n_train = n - n_val
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, va = perm[:n_train], perm[n_train:]
    s_tr, t_tr, s_va, t_va = S[tr], T[tr], S[va], T[va]

    # Context normalization stats from the TRAIN split only (applied to train/val/fiducial).
    if args.context_normalization:
        ctx_mean = s_tr.mean(dim=0, keepdim=True)
        ctx_std = s_tr.std(dim=0, keepdim=True) + 1e-8
        print(f"context normalization ON (train mean/std over {raw_dim} dims)")
    else:
        ctx_mean = ctx_std = None

    def norm_ctx(s):
        if ctx_mean is None:
            return s
        return (s - ctx_mean.to(s.device)) / ctx_std.to(s.device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(s_tr, t_tr), batch_size=args.npe_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(s_va, t_va), batch_size=args.npe_batch_size)

    def make_flow():
        return build_flow(param_dim=2, context_dim=raw_dim,
                          n_transforms=args.n_transforms, hidden_dim=args.hidden_dim).to(device)

    # ── train npe_seeds flows, keep best-per-seed, track overall best (best mode) ──
    seed_states = []
    overall_best_nll = float("inf")
    best_seed_idx = 0
    for seed in range(args.npe_seeds):
        torch.manual_seed(seed + 92)
        flow = make_flow()
        opt = torch.optim.Adam(flow.parameters(), lr=args.npe_lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.npe_epochs)
        best_val, patience, best_state = float("inf"), 0, None
        for epoch in range(args.npe_epochs):
            flow.train()
            for sb, tb in train_loader:
                sb, tb = sb.to(device), tb.to(device)
                loss = -flow.log_prob(tb, context=norm_ctx(sb)).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
                opt.step()
            sch.step()
            flow.eval()
            with torch.no_grad():
                vls = [(-flow.log_prob(tb.to(device), context=norm_ctx(sb.to(device))).mean()).item()
                       for sb, tb in val_loader]
            mv = float(np.mean(vls))
            if mv < best_val:
                best_val = mv
                best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= args.npe_patience:
                    break
        seed_states.append(best_state)
        print(f"  seed {seed}: best val nll = {best_val:.4f}")
        if best_val < overall_best_nll:
            overall_best_nll, best_seed_idx = best_val, seed
    print(f"Best seed = {best_seed_idx} (val nll {overall_best_nll:.4f})")

    seed_flows = []
    for st in seed_states:
        f = make_flow()
        f.load_state_dict(st)
        f.eval()
        seed_flows.append(f)
    best_flow = seed_flows[best_seed_idx]

    # ── fiducial FoM + posterior samples (best mode) ──
    fid = dd["fiducial"].with_format("numpy")
    kappa_fom = np.array(fid["kappa"])
    print(f"Fiducial: {len(kappa_fom)} maps; sampling posteriors...")
    n_plot = min(5, len(kappa_fom))
    all_ps, per_seed, fom_values = [], [], []
    with torch.no_grad():
        for i, ki in enumerate(kappa_fom):
            kr = reshape_field_numpy(ki[np.newaxis])[0]
            s = norm_ctx(raw_context(compressor, kr, device))
            samp = best_flow.sample(args.n_posterior_samples, context=s).squeeze(0).cpu().numpy()
            sp = samp * THETA_STD[:2] + THETA_MEAN[:2]
            all_ps.append(sp)
            cov = np.cov(sp.T)
            det = np.linalg.det(cov)
            fom_values.append(1.0 / np.sqrt(det) if det > 0 else 0.0)
            if i < n_plot:
                per_seed.append([
                    (f.sample(args.n_posterior_samples, context=s).squeeze(0).cpu().numpy()
                     * THETA_STD[:2] + THETA_MEAN[:2])
                    for f in seed_flows
                ])
    fom_mean, fom_std = float(np.mean(fom_values)), float(np.std(fom_values))
    print(f"FoM = {fom_mean:.2f} ± {fom_std:.2f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "posterior_samples.npy", np.stack(all_ps, axis=0))
    np.save(out / "per_seed_posterior_samples.npy",
            np.stack([np.stack(o, axis=0) for o in per_seed], axis=0))
    (out / "results.json").write_text(json.dumps({
        "fom_mean": fom_mean, "fom_std": fom_std, "fom_values": fom_values,
        "inference_mode": "best", "best_val_nll": overall_best_nll,
        "best_seed_idx": best_seed_idx, "raw_context_dim": raw_dim,
        "n_fiducial_maps": len(kappa_fom), "n_posterior_samples": args.n_posterior_samples,
        "compressor_checkpoint": ckpt, "context": "raw_stats",
        "context_normalization": bool(args.context_normalization),
        "npe_epochs": args.npe_epochs,
    }, indent=2))
    print(f"Wrote {out}/results.json")


if __name__ == "__main__":
    main()
