#!/usr/bin/env python3
"""Calibration harness: TARP + SBC + MIRA on the held-out validation set.

Answers "is a tighter posterior (higher FoM) actually correct, or overconfident?"
FoM only measures posterior *width*; calibration measures whether that width is
honest. We evaluate the same NPE on a held-out set with VARYING theta (the fiducial
maps are one cosmology, so they cannot test calibration).

Works for both context types so raw-analytical and compressed summaries are compared
on one code path:
  --context raw         -> compressor._compute_stats(x)   (full ~320-941 D vector)
  --context compressed  -> compressor.compress(x)         (learned 8 D summary)

Metrics
-------
* TARP   : tarp.get_tarp_coverage (Lemos, Coogan et al. 2023). We report
           max_alpha |ECP(alpha) - alpha|  (0 = perfectly calibrated).
* MIRA   : mira_score.mira (Sammy S., github.com/SammyS15/mira-score).
* SBC    : hand-rolled, canonical Talts et al. 2018 rank statistic, matching
           sbi.diagnostics.run_sbc: rank = #{posterior_samples < truth} per dim,
           in {0,...,L}; uniform ranks <=> calibrated. We report a per-dim KS
           p-value vs uniform and the rank histogram (U-shape => too narrow /
           overconfident; inverted-U => too wide / conservative; slope => biased).

Training is reproduced from the config (same fixed 90/10 split + seed offsets as
scripts/run_raw_stats_npe.py / run_npe_budget_scan.py), so no saved flow is needed.
"""
import argparse
import json
import math
import re
from glob import glob
from pathlib import Path

import numpy as np
import torch

from datasets import load_from_disk
from cosmoford import THETA_MEAN, THETA_STD
from cosmoford.dataset import reshape_field_numpy
from cosmoford.models_nopatch import StatsCompressorNoPatch, build_flow  # noqa: F401 (default class)


def _import_class(path):
    """Import a class from a dotted path so any compressor exposing compress()/_compute_stats()
    can be loaded — e.g. the neural RegressionModelNoPatch (EfficientNet map-level summaries)."""
    import importlib
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


# ───────────────────────── compressor / context ─────────────────────────
def find_best_checkpoint(checkpoints_path, budget):
    cdir = Path(checkpoints_path) / f"budget-{budget}"
    cands = []
    for c in glob(str(cdir) + "/**/*.ckpt", recursive=True):
        if Path(c).name == "last.ckpt":
            continue
        m = re.search(r"val_(?:log_prob|nll)=([-+]?\d*\.?\d+)", c)
        sm = re.search(r"(?:step|epoch)=(\d+)", c)
        if m:
            cands.append((float(m.group(1)), int(sm.group(1)) if sm else 0, c))
    if cands:
        best = min(c[0] for c in cands)
        return max([c for c in cands if c[0] == best], key=lambda c: c[1])[2]
    last = cdir / "last.ckpt"
    if last.exists() and last.stat().st_size > 0:
        return str(last)
    raise FileNotFoundError(f"no checkpoint under {cdir}")


def context_of(compressor, kappa_reshaped, mode, device):
    x = torch.from_numpy(kappa_reshaped.astype(np.float32)).unsqueeze(0).to(device)
    return compressor._compute_stats(x) if mode == "raw" else compressor.compress(x)


def build_context(compressor, kappa_all, mode, device):
    out = []
    with torch.no_grad():
        for i in range(len(kappa_all)):
            kr = reshape_field_numpy(kappa_all[i][np.newaxis])[0]
            out.append(context_of(compressor, kr, mode, device).cpu())
            if (i + 1) % 1000 == 0:
                print(f"  context {i+1}/{len(kappa_all)}", flush=True)
    return torch.cat(out, dim=0)


# ───────────────────────── SBC (Talts et al. 2018) ─────────────────────────
def sbc_ranks(posterior_samples, true_theta):
    """Rank statistic per dimension. Matches sbi.diagnostics.run_sbc.

    posterior_samples: (N, L, D)   true_theta: (N, D)
    returns ranks (N, D), each in {0, ..., L} = #{ samples strictly < truth }.
    """
    return (posterior_samples < true_theta[:, None, :]).sum(axis=1)


def sbc_uniformity_pvalue(ranks_col, L):
    """KS p-value that a rank column {0..L} is uniform (continuity-corrected)."""
    from scipy.stats import kstest
    u = (ranks_col.astype(float) + 0.5) / (L + 1)  # -> (0,1), uniform if calibrated
    return float(kstest(u, "uniform").pvalue)


def sbc_shape(ranks_col, L):
    """Coarse diagnosis from where rank mass concentrates."""
    edge = np.mean((ranks_col <= 0.05 * L) | (ranks_col >= 0.95 * L))
    mid = np.mean((ranks_col >= 0.30 * L) & (ranks_col <= 0.70 * L))
    lo, hi = np.mean(ranks_col < 0.5 * L), np.mean(ranks_col > 0.5 * L)
    if abs(lo - hi) > 0.15:
        return "sloped/biased"
    if edge > 1.6 * (0.10):        # >1.6x the uniform expectation in the 10% edge mass
        return "U-shaped -> too NARROW (overconfident)"
    if mid > 1.4 * (0.40):         # >1.4x uniform in the central 40%
        return "inverted-U -> too WIDE (conservative)"
    return "~uniform (calibrated)"


# ───────────────────────── split / norm / load / FoM ─────────────────────────
def split_and_norm(S, T, cfg):
    """Fixed 90/10 split (seed 0 — identical to run_npe_budget_scan / run_raw_stats_npe),
    with context z-scoring computed on the TRAIN split only. Reproduces exactly the
    context the deck's saved flow was trained to expect."""
    n = S.shape[0]
    n_val = n // 10
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, va = perm[: n - n_val], perm[n - n_val:]
    s_tr, t_tr, s_va, t_va = S[tr], T[tr], S[va], T[va]
    cm = s_tr.mean(0, keepdim=True) if cfg.context_normalization else None
    # clamp_min (NOT + eps): matches the old pipeline that trained the deck flows. With + eps,
    # near-constant raw-stat dims (std~0, e.g. empty L1 SNR bins) divide by ~1e-8 and explode,
    # which corrupts the context a *loaded* flow was trained on (raw_l1 OLD FoM 226 bug).
    cs = s_tr.std(0, keepdim=True).clamp_min(1e-6) if cfg.context_normalization else None
    def norm(s):
        return s if cm is None else (s - cm.to(s.device)) / cs.to(s.device)
    return s_tr, t_tr, s_va, t_va, norm


def load_saved_flow(path, ctx_dim, n_transforms, hidden_dim, device):
    """Reconstruct the exact architecture and load a saved flow state_dict (strict)."""
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, (list, tuple)):
        sd = sd[0]
    if isinstance(sd, dict) and "state_dict" in sd and all(isinstance(k, str) for k in sd):
        # guard: only unwrap if it's a wrapper, not a raw tensor-valued state_dict
        if not any(torch.is_tensor(v) for v in list(sd.values())[:1]):
            sd = sd["state_dict"]
    flow = build_flow(param_dim=2, context_dim=ctx_dim,
                      n_transforms=n_transforms, hidden_dim=hidden_dim).to(device)
    flow.load_state_dict(sd, strict=True)
    flow.eval()
    return flow


def compute_fom(flow, norm, compressor, dd, mode, device, n_samples=10000, save_path=None):
    """FoM = mean over fiducial maps of 1/sqrt(det Cov(Omega_m, S_8)).
    If save_path is set, also save the per-fiducial-map posterior samples (physical units,
    shape (n_fid, n_samples, 2)) for contour plots."""
    fid = dd["fiducial"].with_format("numpy")
    kappa = np.array(fid["kappa"])
    foms, saved = [], []
    with torch.no_grad():
        for ki in kappa:
            kr = reshape_field_numpy(ki[np.newaxis])[0]
            s = norm(context_of(compressor, kr, mode, device))
            samp = flow.sample(n_samples, context=s).squeeze(0).cpu().numpy()
            sp = samp * THETA_STD[:2] + THETA_MEAN[:2]
            if save_path is not None:
                saved.append(sp)
            cov = np.cov(sp.T); det = np.linalg.det(cov)
            foms.append(1.0 / np.sqrt(det) if det > 0 else np.nan)
    if save_path is not None:
        np.save(save_path, np.stack(saved))    # (n_fid, n_samples, 2) physical θ
    return float(np.nanmean(foms))


def recover_fixed_norm(comp, flat_path, n_maps, device):
    """Recompute + inject the compressor's fixed-norm buffers (feature_mean/std) over the flat
    train split, replicating StatsCompressorNoPatch.on_train_start. For raw_l1_p5 the buffers
    were not saved in the checkpoint, so _compute_stats silently skips the z-scoring; this
    restores it so the loaded deck flow sees the context it was trained on. Must run BEFORE any
    context is built (buffers absent => the pass below returns the RAW, un-normalized stats)."""
    flat = load_from_disk(flat_path)["train"].with_format("numpy")
    n = min(n_maps, len(flat))
    kappa = np.array(flat[:n]["kappa"])
    feats = build_context(comp, kappa, "raw", device)      # buffers absent -> raw stats
    mean = feats.mean(0)
    std = feats.std(0, unbiased=False) + 1e-8               # matches on_train_start
    comp.register_buffer("feature_mean", mean.to(device))
    comp.register_buffer("feature_std", std.to(device))
    print(f"recovered fixed-norm buffers over {n} flat maps (dim={mean.numel()})", flush=True)


# ───────────────────────── train NPE (reproducible) ─────────────────────────
def train_npe(S, T, cfg, device, comp=None, dd=None):
    s_tr, t_tr, s_va, t_va, norm = split_and_norm(S, T, cfg)
    ctx_dim = S.shape[1]

    tl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(s_tr, t_tr),
                                     batch_size=cfg.npe_batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(s_va, t_va),
                                     batch_size=cfg.npe_batch_size)
    mk = lambda: build_flow(param_dim=2, context_dim=ctx_dim,
                            n_transforms=cfg.n_transforms, hidden_dim=cfg.hidden_dim,
                            embedding_dim=(cfg.embedding_dim or None),
                            embedding_hidden=cfg.embedding_hidden,
                            embedding_dropout=cfg.embedding_dropout,
                            made_dropout=cfg.made_dropout).to(device)
    warm = max(1, int(cfg.npe_warmup_frac * cfg.npe_epochs))
    def _lr(ep):   # linear warm-up then cosine decay
        if ep < warm:
            return (ep + 1) / warm
        prog = (ep - warm) / max(1, cfg.npe_epochs - warm)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    def train_one(seed_val):
        torch.manual_seed(seed_val)
        flow = mk()
        opt = torch.optim.Adam(flow.parameters(), lr=cfg.npe_lr, weight_decay=cfg.weight_decay)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
        bv, pat, bs = float("inf"), 0, None
        for _ in range(cfg.npe_epochs):
            flow.train()
            for sb, tb in tl:
                sb, tb = sb.to(device), tb.to(device)
                loss = -flow.log_prob(tb, context=norm(sb)).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
            sch.step()
            flow.eval()
            with torch.no_grad():
                mv = float(np.mean([(-flow.log_prob(tb.to(device), context=norm(sb.to(device))).mean()).item()
                                    for sb, tb in vl]))
            if mv < bv:
                bv, pat, bs = mv, 0, {k: v.cpu().clone() for k, v in flow.state_dict().items()}
            else:
                pat += 1
                if pat >= cfg.npe_patience:
                    break
        return bs, bv

    # retry-on-collapse: a bad init lands the flow in the context-ignoring near-prior basin
    # (FoM ~ tens). Warm-up doesn't cure it, so detect via a cheap fiducial FoM and re-init.
    can_check = comp is not None and dd is not None and cfg.max_retries > 0
    seed_states, seed_nlls = [], []
    for seed in range(cfg.npe_seeds):
        qf = None
        for attempt in range(cfg.max_retries + 1):
            bs, bv = train_one(seed + 92 + attempt * 1000)
            if not can_check:
                break
            ft = mk(); ft.load_state_dict(bs); ft.eval()
            qf = compute_fom(ft, norm, comp, dd, cfg.context, device, n_samples=1500)
            if qf >= cfg.collapse_floor or attempt == cfg.max_retries:
                break
            print(f"  seed {seed} attempt {attempt}: collapsed (FoM {qf:.0f} < {cfg.collapse_floor:g}), "
                  f"re-init", flush=True)
        print(f"  seed {seed}: best val nll {bv:.4f}" + (f"  FoM~{qf:.0f}" if qf is not None else ""),
              flush=True)
        seed_states.append(bs); seed_nlls.append(bv)
    seed_flows = []
    for st in seed_states:
        f = mk(); f.load_state_dict(st); f.eval(); seed_flows.append(f)
    best_idx = int(np.argmin(seed_nlls))
    return seed_flows, best_idx, norm, (s_va, t_va), seed_nlls


def sample_norm(flow, contexts_norm, L, device, batch=64):
    """Draw L posterior samples (normalized theta) per context. -> (N, L, 2)"""
    out = []
    with torch.no_grad():
        for i in range(len(contexts_norm)):
            s = contexts_norm[i:i+1].to(device)
            out.append(flow.sample(L, context=s).squeeze(0).cpu().numpy())
    return np.stack(out, axis=0)


def evaluate_flow(flow, norm, comp, dd, s_va, t_va, cfg, device):
    """FoM (fiducial) + TARP dev + SBC KS p-values for a single flow. MIRA is dropped
    (truth-only-box normalization bug in mira-score, github.com/SammyS15/mira-score)."""
    from tarp import get_tarp_coverage
    fom = compute_fom(flow, norm, comp, dd, cfg.context, device)
    ncal = min(cfg.n_cal_maps, len(s_va)); L = cfg.n_cal_samples
    truth = t_va[:ncal].numpy()
    samp = sample_norm(flow, norm(s_va[:ncal]), L, device)      # (ncal, L, 2) normalized
    ecp, alpha = get_tarp_coverage(samp.transpose(1, 0, 2), truth,
                                   references="random", metric="euclidean", norm=True)
    ranks = sbc_ranks(samp, truth)
    return {"fom": fom, "tarp_dev": float(np.max(np.abs(ecp - alpha))),
            "sbc_om_p": sbc_uniformity_pvalue(ranks[:, 0], L),
            "sbc_s8_p": sbc_uniformity_pvalue(ranks[:, 1], L),
            "ecp": ecp, "alpha": alpha, "ranks": ranks, "L": L, "ncal": ncal}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints_path", default=None,
                    help="root holding budget-<N>/ checkpoint dirs (best is auto-picked by "
                         "val_log_prob). Omit if you pass --compressor_ckpt directly.")
    ap.add_argument("--compressor_ckpt", default=None,
                    help="exact compressor .ckpt path (bypasses --checkpoints_path/--budget auto-pick)")
    ap.add_argument("--compressor_class", default="cosmoford.models_nopatch.StatsCompressorNoPatch",
                    help="dotted class path of the compressor to load; use "
                         "cosmoford.models_nopatch.RegressionModelNoPatch for the neural map-level summaries")
    ap.add_argument("--budget", type=int, default=20200)
    ap.add_argument("--context", choices=["raw", "compressed"], required=True)
    ap.add_argument("--holdout_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--n_transforms", type=int, default=5)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--npe_epochs", type=int, default=500)
    ap.add_argument("--npe_seeds", type=int, default=10)
    ap.add_argument("--npe_lr", type=float, default=1e-4)
    ap.add_argument("--npe_batch_size", type=int, default=128)
    ap.add_argument("--npe_patience", type=int, default=50)
    ap.add_argument("--npe_warmup_frac", type=float, default=0.05,
                    help="fraction of epochs for linear LR warm-up (anti-collapse); 0 disables")
    ap.add_argument("--embedding_dim", type=int, default=0,
                    help="if >0, add a jointly-trained MLP embedding net (context -> embedding_dim) "
                         "before the flow — the high-D raw-context fix. 0 = original no-embedding flow.")
    ap.add_argument("--embedding_hidden", type=int, default=256)
    ap.add_argument("--embedding_dropout", type=float, default=0.1)
    ap.add_argument("--made_dropout", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--collapse_frac", type=float, default=0.4,
                    help="safety net: drop seeds whose FoM < this fraction of the median FoM. 0 = keep all.")
    ap.add_argument("--max_retries", type=int, default=3,
                    help="re-init a seed up to this many times if its fiducial FoM < --collapse_floor "
                         "(source fix for the near-prior collapse). 0 = no retry.")
    ap.add_argument("--collapse_floor", type=float, default=300.0,
                    help="absolute FoM below which a freshly-trained seed is deemed collapsed and re-init'd")
    ap.add_argument("--context_normalization", action="store_true")
    ap.add_argument("--fixed_norm_from", default=None,
                    help="flat dataset path (…/neurips-wl-challenge-flat). Recompute + inject the "
                         "compressor's fixed-normalization buffers over its flat train split, for "
                         "checkpoints where they weren't saved (raw_l1_p5). Needed to reproduce the "
                         "deck's raw_l1 context.")
    ap.add_argument("--fixed_norm_maps", type=int, default=8000,
                    help="how many flat-train maps to average for the recovered buffers")
    ap.add_argument("--load_flow", default=None,
                    help="path to a saved flow state_dict (e.g. the deck's npe_flow.pt). "
                         "If set, skip training and calibrate THAT exact flow; pass the "
                         "matching --n_transforms/--hidden_dim (deck: 4 and 128/256).")
    ap.add_argument("--max_train_maps", type=int, default=0,
                    help="cap holdout-train maps used to build context + train (0=all; use a "
                         "small value like 500 for fast smoke tests)")
    ap.add_argument("--n_cal_maps", type=int, default=2000, help="cap on validation maps for calibration")
    ap.add_argument("--n_cal_samples", type=int, default=1000, help="L: posterior samples per map")
    cfg = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = cfg.compressor_ckpt or find_best_checkpoint(cfg.checkpoints_path, cfg.budget)
    CompCls = _import_class(cfg.compressor_class)
    print(f"compressor: {CompCls.__name__} <- {ckpt}", flush=True)
    comp = CompCls.load_from_checkpoint(ckpt, map_location=device)
    comp.eval().to(device)
    for p in comp.parameters():
        p.requires_grad = False

    if cfg.fixed_norm_from:
        recover_fixed_norm(comp, cfg.fixed_norm_from, cfg.fixed_norm_maps, device)

    dd = load_from_disk(cfg.holdout_path)
    tr = dd["train"].with_format("numpy")
    kappa = np.array(tr["kappa"]); theta = np.array(tr["theta"])
    if cfg.max_train_maps > 0:
        kappa, theta = kappa[:cfg.max_train_maps], theta[:cfg.max_train_maps]
        print(f"capped holdout-train to {len(kappa)} maps (smoke)", flush=True)
    theta_n = ((theta[:, :2] - THETA_MEAN[:2]) / THETA_STD[:2]).astype(np.float32)
    print(f"holdout train: {len(kappa)} maps; building {cfg.context} context...", flush=True)
    S = build_context(comp, kappa, cfg.context, device)
    T = torch.from_numpy(theta_n)
    print(f"context dim = {S.shape[1]}", flush=True)

    if cfg.load_flow:
        _, _, s_va, t_va, norm = split_and_norm(S, T, cfg)
        seed_flows = [load_saved_flow(cfg.load_flow, S.shape[1], cfg.n_transforms, cfg.hidden_dim, device)]
        best_idx, seed_nlls = 0, [None]
        print(f"loaded saved flow: {cfg.load_flow}", flush=True)
    else:
        seed_flows, best_idx, norm, (s_va, t_va), seed_nlls = train_npe(S, T, cfg, device, comp, dd)

    print(f"evaluating {len(seed_flows)} flow(s): FoM + TARP + SBC per seed ...", flush=True)
    per = [evaluate_flow(f, norm, comp, dd, s_va, t_va, cfg, device) for f in seed_flows]
    # ── drop collapsed seeds: a fraction of NPE flows diverge to a near-prior (FoM ~ tens) from a
    #    bad init; they poison the mean. Keep seeds with FoM >= collapse_frac x median FoM. ──
    foms = np.array([p["fom"] for p in per], dtype=float)
    med = float(np.median(foms))
    kept = [i for i in range(len(per)) if foms[i] >= cfg.collapse_frac * med] if len(per) > 1 else [0]
    dropped = [i for i in range(len(per)) if i not in kept]
    if dropped:
        print(f"  dropped {len(dropped)} collapsed seed(s) FoM={[round(foms[i]) for i in dropped]} "
              f"(< {cfg.collapse_frac:g}x median {med:.0f})", flush=True)
    if seed_nlls[best_idx] is not None and best_idx not in kept and kept:
        best_idx = min(kept, key=lambda i: seed_nlls[i])   # plot a survivor, not a collapsed seed
    prim = per[best_idx]                     # best-val-nll surviving flow — used for the saved plots
    ecp, alpha, ranks, L, ncal = prim["ecp"], prim["alpha"], prim["ranks"], prim["L"], prim["ncal"]
    tarp_dev = prim["tarp_dev"]
    sbc = {name: {"ks_pvalue": prim[f"sbc_{k}_p"], "shape": sbc_shape(ranks[:, d], L)}
           for d, (name, k) in enumerate([("Omega_m", "om"), ("S_8", "s8")])}

    def agg(key):
        v = [per[i][key] for i in kept]
        return {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "median": float(np.median(v)), "values": [float(x) for x in v]}
    seed_summary = {k: agg(k) for k in ["fom", "tarp_dev", "sbc_om_p", "sbc_s8_p"]}
    seed_summary["n_kept"] = len(kept)
    seed_summary["n_dropped"] = len(dropped)
    seed_summary["dropped_fom"] = [round(float(foms[i]), 1) for i in dropped]
    print(f"  seed-avg over {len(kept)} kept: FoM = {seed_summary['fom']['mean']:.0f} +/- "
          f"{seed_summary['fom']['std']:.0f} (median {seed_summary['fom']['median']:.0f})  "
          f"TARPdev = {seed_summary['tarp_dev']['mean']:.3f}", flush=True)

    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "sbc_ranks.npy", ranks)
    np.save(out / "tarp_ecp_alpha.npy", np.stack([ecp, alpha]))
    compute_fom(seed_flows[best_idx], norm, comp, dd, cfg.context, device,
                save_path=str(out / "fiducial_posterior.npy"))   # for contour plots
    results = {
        "label": cfg.label, "context": cfg.context, "context_dim": int(S.shape[1]),
        "fom_mean": prim["fom"], "tarp_max_deviation": tarp_dev, "sbc": sbc,
        "loaded_flow": cfg.load_flow, "best_val_nll": seed_nlls[best_idx],
        "seed_summary": seed_summary,        # <-- seeds-averaged FoM / TARP / SBC (mean +/- std)
        "n_cal_maps": ncal, "n_cal_samples": L,
        "flow": {"n_transforms": cfg.n_transforms, "hidden_dim": cfg.hidden_dim,
                 "npe_epochs": cfg.npe_epochs, "npe_seeds": cfg.npe_seeds,
                 "context_normalization": cfg.context_normalization,
                 "embedding_dim": cfg.embedding_dim, "embedding_hidden": cfg.embedding_hidden,
                 "embedding_dropout": cfg.embedding_dropout, "made_dropout": cfg.made_dropout,
                 "weight_decay": cfg.weight_decay},
        "compressor_checkpoint": ckpt,
    }
    (out / "calibration.json").write_text(json.dumps(results, indent=2))

    # ── plots: TARP coverage + SBC rank histograms ──
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(alpha, ecp, lw=2); ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("credibility level"); ax[0].set_ylabel("expected coverage")
    ax[0].set_title(f"TARP  (max dev {tarp_dev:.3f})")
    B = 20
    for d, name in enumerate(["Omega_m", "S_8"]):
        a = ax[d + 1]
        a.hist(ranks[:, d], bins=B, range=(0, L), color="C0", alpha=.8)
        exp = ncal / B
        band = 2.576 * np.sqrt(exp * (1 - 1 / B))
        a.axhline(exp, color="k", ls="--", lw=1)
        a.axhspan(exp - band, exp + band, color="grey", alpha=.25)
        a.set_title(f"SBC rank: {name}\nKS p={sbc[name]['ks_pvalue']:.3f}  {sbc[name]['shape']}",
                    fontsize=9)
        a.set_xlabel("rank")
    fig.suptitle(f"{cfg.label}  |  TARP dev {tarp_dev:.3f}  (seed-avg FoM "
                 f"{seed_summary['fom']['mean']:.0f}±{seed_summary['fom']['std']:.0f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "calibration.png", dpi=130)
    print(f"\n{cfg.label}: TARP_dev={tarp_dev:.3f}  "
          f"SBC KS p=({sbc['Omega_m']['ks_pvalue']:.3f},{sbc['S_8']['ks_pvalue']:.3f})", flush=True)
    print(f"wrote {out}/calibration.json + calibration.png", flush=True)


if __name__ == "__main__":
    main()
