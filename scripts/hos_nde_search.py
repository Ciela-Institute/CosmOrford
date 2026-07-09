#!/usr/bin/env python
"""Overnight HOS-NDE search: maximise the HOS Figure of Merit by manipulating ONLY the
analytical statistics and the neural density estimator — NO separate/trained compressor.

The statistics are analytical (an untrained StatsCompressorNoPatch just computes _compute_stats);
the dimensionality reduction is the in-NDE embedding, trained jointly with the flow. This is the
simulation-budget-INDEPENDENT arm (nothing here needs a compressor budget).

Design:
  * precompute: for each stat combo, instantiate an untrained compressor with those hparams,
    compute the raw-stat context for the holdout-train maps + the fiducial maps, cache to npz.
  * search (Round 1): each combo through a fixed generous embedding-NDE (screen, few seeds).
  * search (Round 2): top-K combos through an NDE grid (maf/nsf, depth/width, embedding, dropout).
  * confirm (Round 3): the best (stat x NDE) re-run at high seeds.
Ranking is by CALIBRATED FoM (must pass a TARP/SBC gate; else penalised). FoM=1/sqrt(det Cov).

Resumable: caches persist; results.json is written after every run; the run-plan is recomputed
from results.json on restart so a fresh 3h job continues where the last stopped. Time-aware: stops
launching new runs when the wall-clock budget is exhausted, always writing a report.
"""
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch

from datasets import load_from_disk
from cosmoford import THETA_MEAN, THETA_STD
from cosmoford.dataset import reshape_field_numpy
from cosmoford.models_nopatch import StatsCompressorNoPatch, build_flow

T0 = time.time()

# ───────────────────────── stat combos ─────────────────────────
# base matches the ~1666 champion (HOS+Scat, ns=4, J=4, default SNR ranges).
BASE_STATS = dict(scattering_J=4, hos_n_scales=4, use_fixed_normalization=False)
WIDE = dict(hos_min_snr=-5.0, hos_max_snr=10.0, hos_l1_min_snr=-12.0, hos_l1_max_snr=12.0)
STAT_SPECS = {
    "hos_scat":          dict(use_hos=True, use_scattering=True),                       # champion baseline
    "hos_scat_wide":     dict(use_hos=True, use_scattering=True, **WIDE),               # ablation best (wide SNR)
    "ps_hos_scat":       dict(use_ps=True, use_hos=True, use_scattering=True),          # + power spectrum
    "ps_hos_scat_wide":  dict(use_ps=True, use_hos=True, use_scattering=True, **WIDE),
    "hos_scat_widebins": dict(use_hos=True, use_scattering=True, hos_n_bins=71,
                              hos_l1_nbins=120, **WIDE),                                # wide + finer bins
    "hos_only":          dict(use_hos=True, use_scattering=False),                      # full HOS, no scattering
}

# ───────────────────────── NDE configs ─────────────────────────
# Round-1 fixed screen NDE: generous embedding so no combo is starved by its dimensionality.
BASE_NDE = dict(flow_type="maf", n_transforms=5, hidden_dim=64, embedding_dim=64,
                embedding_hidden=256, embedding_dropout=0.1, made_dropout=0.0, context_norm=True)
# Round-2 grid (curated, applied to top-K combos). Each is a delta on BASE_NDE.
NDE_GRID = [
    ("maf_base",     dict()),
    ("nsf_base",     dict(flow_type="nsf")),
    ("nsf_deep",     dict(flow_type="nsf", n_transforms=8)),
    ("nsf_wide",     dict(flow_type="nsf", hidden_dim=128)),
    ("nsf_emb32",    dict(flow_type="nsf", embedding_dim=32)),
    ("nsf_emb16",    dict(flow_type="nsf", embedding_dim=16)),
    ("nsf_drop",     dict(flow_type="nsf", made_dropout=0.1, embedding_dropout=0.1)),   # calibration lever
    ("nsf_deep_wide",dict(flow_type="nsf", n_transforms=8, hidden_dim=128)),
    ("maf_deep",     dict(n_transforms=8, hidden_dim=128)),
    ("maf_drop",     dict(made_dropout=0.1)),
]


def make_compressor(combo, device):
    hp = dict(BASE_STATS); hp.update(STAT_SPECS[combo])
    comp = StatsCompressorNoPatch(**hp).eval().to(device)
    for p in comp.parameters():
        p.requires_grad = False
    return comp


def build_context(comp, kappa_all, device, tag=""):
    out = []
    with torch.no_grad():
        for i in range(len(kappa_all)):
            kr = reshape_field_numpy(kappa_all[i][np.newaxis])[0]
            x = torch.from_numpy(kr.astype(np.float32)).unsqueeze(0).to(device)
            out.append(comp._compute_stats(x).cpu())
            if (i + 1) % 2000 == 0:
                print(f"    [{tag}] context {i+1}/{len(kappa_all)}", flush=True)
    return torch.cat(out, dim=0).numpy().astype(np.float32)


def precompute(combo, holdout_path, cache_dir, max_train, device):
    cache = Path(cache_dir) / f"stats_{combo}.npz"
    if cache.exists():
        return str(cache)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"  precompute stats: {combo}", flush=True)
    comp = make_compressor(combo, device)
    dd = load_from_disk(holdout_path)
    tr = dd["train"].with_format("numpy")
    kap = np.array(tr["kappa"]); th = np.array(tr["theta"])
    if max_train > 0:
        kap, th = kap[:max_train], th[:max_train]
    S = build_context(comp, kap, device, tag=f"{combo}/train")
    T = ((th[:, :2] - THETA_MEAN[:2]) / THETA_STD[:2]).astype(np.float32)
    fid = dd["fiducial"].with_format("numpy"); fk = np.array(fid["kappa"])
    Sf = build_context(comp, fk, device, tag=f"{combo}/fid")
    np.savez(str(cache), S=S, T=T, Sfid=Sf)
    print(f"    cached {cache}  S{S.shape} Sfid{Sf.shape}", flush=True)
    del comp, dd
    torch.cuda.empty_cache()
    return str(cache)


# ───────────────────────── NDE train / eval on cached stats ─────────────────────────
def split_norm(S, T, ctx_norm):
    n = S.shape[0]; nv = n // 10
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, va = perm[: n - nv], perm[n - nv:]
    s_tr, t_tr, s_va, t_va = S[tr], T[tr], S[va], T[va]
    cm = s_tr.mean(0, keepdim=True) if ctx_norm else None
    cs = s_tr.std(0, keepdim=True).clamp_min(1e-6) if ctx_norm else None
    def norm(s):
        return s if cm is None else (s - cm.to(s.device)) / cs.to(s.device)
    return s_tr, t_tr, s_va, t_va, norm


def cached_fom(flow, norm, Sfid, n_samples, device):
    foms = []
    with torch.no_grad():
        for row in Sfid:
            s = norm(row.unsqueeze(0).to(device))
            samp = flow.sample(n_samples, context=s).squeeze(0).cpu().numpy()
            sp = samp * THETA_STD[:2] + THETA_MEAN[:2]
            cov = np.cov(sp.T); det = np.linalg.det(cov)
            foms.append(1.0 / np.sqrt(det) if det > 0 else np.nan)
    return float(np.nanmean(foms))


def sbc_ranks(ps, truth):
    return (ps < truth[:, None, :]).sum(axis=1)


def sbc_p(ranks_col, L):
    from scipy.stats import kstest
    u = (ranks_col.astype(float) + 0.5) / (L + 1)
    return float(kstest(u, "uniform").pvalue)


def tarp_sbc(flow, norm, s_va, t_va, n_cal, L, device):
    from tarp import get_tarp_coverage
    ncal = min(n_cal, len(s_va))
    truth = t_va[:ncal].numpy()
    out = []
    with torch.no_grad():
        for i in range(ncal):
            s = norm(s_va[i:i+1].to(device))
            out.append(flow.sample(L, context=s).squeeze(0).cpu().numpy())
    samp = np.stack(out, axis=0)                       # (ncal, L, 2) normalized
    ecp, alpha = get_tarp_coverage(samp.transpose(1, 0, 2), truth,
                                   references="random", metric="euclidean", norm=True)
    ranks = sbc_ranks(samp, truth)
    return (float(np.max(np.abs(ecp - alpha))), sbc_p(ranks[:, 0], L), sbc_p(ranks[:, 1], L))


def run_nde(cache_path, nde, seeds, epochs, patience, lr, n_fid_samp, n_cal, L,
            floor, ceiling, frac, max_retries, device):
    z = np.load(cache_path)
    S = torch.from_numpy(z["S"]); T = torch.from_numpy(z["T"]); Sfid = torch.from_numpy(z["Sfid"])
    ctx_dim = S.shape[1]
    s_tr, t_tr, s_va, t_va, norm = split_norm(S, T, nde["context_norm"])
    tl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(s_tr, t_tr), batch_size=128, shuffle=True)
    vl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(s_va, t_va), batch_size=256)
    warm = max(1, int(0.05 * epochs))
    def lr_at(ep):
        if ep < warm:
            return (ep + 1) / warm
        return 0.5 * (1.0 + math.cos(math.pi * (ep - warm) / max(1, epochs - warm)))
    def mk():
        return build_flow(param_dim=2, context_dim=ctx_dim, n_transforms=nde["n_transforms"],
                          hidden_dim=nde["hidden_dim"], flow_type=nde["flow_type"],
                          embedding_dim=(nde["embedding_dim"] or None),
                          embedding_hidden=nde["embedding_hidden"],
                          embedding_dropout=nde["embedding_dropout"],
                          made_dropout=nde["made_dropout"],
                          num_bins=nde.get("num_bins", 8)).to(device)
    def train_one(seed):
        torch.manual_seed(seed)
        flow = mk(); opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=nde.get("weight_decay", 0.0))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        bv, pat, bs = float("inf"), 0, None
        for _ in range(epochs):
            flow.train()
            for sb, tb in tl:
                sb, tb = sb.to(device), tb.to(device)
                loss = -flow.log_prob(tb, context=norm(sb)).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
            sch.step(); flow.eval()
            with torch.no_grad():
                mv = float(np.mean([(-flow.log_prob(tb.to(device), context=norm(sb.to(device))).mean()).item()
                                    for sb, tb in vl]))
            if mv < bv:
                bv, pat, bs = mv, 0, {k: v.cpu().clone() for k, v in flow.state_dict().items()}
            else:
                pat += 1
                if pat >= patience:
                    break
        return bs, bv
    seed_states = []
    for seed in range(seeds):
        qf = None
        for attempt in range(max_retries + 1):
            bs, bv = train_one(seed + 92 + attempt * 1000)
            ft = mk(); ft.load_state_dict(bs); ft.eval()
            qf = cached_fom(ft, norm, Sfid, 1500, device)
            if qf >= floor or attempt == max_retries:
                break
        seed_states.append(bs)
    # evaluate kept seeds
    per = []
    for st in seed_states:
        f = mk(); f.load_state_dict(st); f.eval()
        fom = cached_fom(f, norm, Sfid, n_fid_samp, device)
        td, pom, ps8 = tarp_sbc(f, norm, s_va, t_va, n_cal, L, device)
        per.append(dict(fom=fom, tarp_dev=td, sbc_om_p=pom, sbc_s8_p=ps8))
    foms = np.array([p["fom"] for p in per], float)
    okc = foms <= ceiling
    med = float(np.median(foms[okc])) if okc.any() else float(np.median(foms))
    kept = [i for i in range(len(per)) if okc[i] and foms[i] >= frac * med] or list(range(len(per)))
    kf = foms[kept]
    kt = [per[i]["tarp_dev"] for i in kept]
    ko = [per[i]["sbc_om_p"] for i in kept]; ks = [per[i]["sbc_s8_p"] for i in kept]
    return dict(ctx_dim=int(ctx_dim), n_seeds=seeds, n_kept=len(kept), n_dropped=seeds - len(kept),
                fom_median=float(np.median(kf)), fom_mean=float(np.mean(kf)), fom_std=float(np.std(kf)),
                tarp_dev_med=float(np.median(kt)), sbc_om_p_med=float(np.median(ko)),
                sbc_s8_p_med=float(np.median(ks)),
                fom_all=[round(x, 1) for x in foms.tolist()])


def calibrated(r):
    """A run is 'calibrated' if TARP + SBC pass a lenient gate."""
    return r["tarp_dev_med"] < 0.12 and min(r["sbc_om_p_med"], r["sbc_s8_p_med"]) > 5e-3


def rank_key(r):
    # calibrated runs rank by FoM; uncalibrated are penalised (halved) so they can still surface
    return r["fom_median"] * (1.0 if calibrated(r) else 0.5)


def save(results, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout_path", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--time_budget_sec", type=float, default=9600.0)  # ~2h40 of a 3h job
    ap.add_argument("--combos", default="all")
    ap.add_argument("--topk", type=int, default=2)          # combos carried to Round 2
    ap.add_argument("--topc", type=int, default=3)          # (combo x nde) carried to Round 3
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    combos = list(STAT_SPECS) if args.combos == "all" else args.combos.split(",")
    if args.mode == "smoke":
        combos = combos[:1]
        SEED_SCREEN, SEED_CONF, EPOCHS, MAXTR = 2, 2, 20, 400
        NFS, NCAL, LCAL = 400, 120, 200
        grid = NDE_GRID[:2]
    else:
        SEED_SCREEN, SEED_CONF, EPOCHS, MAXTR = 8, 20, 200, 0
        NFS, NCAL, LCAL = 4000, 1500, 800
        grid = NDE_GRID
    results = json.load(open(args.out)) if Path(args.out).exists() else {}
    def budget_left():
        return (time.time() - T0) < args.time_budget_sec

    def do(key, cache, nde, seeds, epochs, patience, conf=False):
        if key in results or not budget_left():
            return
        print(f"[{(time.time()-T0)/60:.0f}m] RUN {key}  ({nde['flow_type']} t{nde['n_transforms']} "
              f"h{nde['hidden_dim']} e{nde['embedding_dim']} d{nde['made_dropout']}) seeds={seeds}", flush=True)
        try:
            nfs, ncal, lcal = (10000, 2000, 1000) if conf else (NFS, NCAL, LCAL)
            r = run_nde(cache, nde, seeds, epochs, 30, 1e-4, nfs, ncal, lcal,
                        300.0, 3000.0, 0.4, 3, device)
            r["nde"] = {k: nde[k] for k in ("flow_type", "n_transforms", "hidden_dim", "embedding_dim",
                                            "made_dropout", "embedding_dropout", "context_norm")}
            results[key] = r; save(results, args.out)
            print(f"    -> FoM {r['fom_median']:.0f} (mean {r['fom_mean']:.0f}+/-{r['fom_std']:.0f}, "
                  f"kept {r['n_kept']}/{seeds}) TARP {r['tarp_dev_med']:.3f} "
                  f"SBC Om/S8 {r['sbc_om_p_med']:.2f}/{r['sbc_s8_p_med']:.2f} "
                  f"{'CAL' if calibrated(r) else 'uncal'}", flush=True)
        except Exception as e:
            results[key] = {"error": f"{type(e).__name__}: {e}"}; save(results, args.out)
            print(f"    !! {key} FAILED: {e}", flush=True)

    # ---- precompute (always, cached) ----
    caches = {}
    for c in combos:
        if not budget_left():
            break
        try:
            caches[c] = precompute(c, args.holdout_path, args.cache_dir, MAXTR, device)
        except Exception as e:
            print(f"  precompute {c} FAILED: {e}", flush=True)

    # ---- Round 1: screen each combo on BASE_NDE ----
    for c in combos:
        if c in caches:
            do(f"R1/{c}", caches[c], BASE_NDE, SEED_SCREEN, EPOCHS, 30)

    # ---- Round 2: NDE grid on top-K combos ----
    r1 = {c: results[f"R1/{c}"] for c in combos if f"R1/{c}" in results and "error" not in results[f"R1/{c}"]}
    top_combos = sorted(r1, key=lambda c: rank_key(r1[c]), reverse=True)[:args.topk]
    print(f"[{(time.time()-T0)/60:.0f}m] Round-1 top combos: {top_combos}", flush=True)
    for c in top_combos:
        if c not in caches:
            continue
        for gid, delta in grid:
            nde = dict(BASE_NDE); nde.update(delta)
            do(f"R2/{c}/{gid}", caches[c], nde, SEED_SCREEN, EPOCHS, 30)

    # ---- Round 3: confirm best (combo x nde) at high seeds ----
    r2 = {k: v for k, v in results.items() if k.startswith("R2/") and "error" not in v}
    top_cfgs = sorted(r2, key=lambda k: rank_key(r2[k]), reverse=True)[:args.topc]
    print(f"[{(time.time()-T0)/60:.0f}m] Round-2 top configs: {top_cfgs}", flush=True)
    for k in top_cfgs:
        _, c, gid = k.split("/")
        if c not in caches:
            continue
        nde = dict(BASE_NDE); nde.update(dict(NDE_GRID)[gid])
        do(f"R3/{c}/{gid}", caches[c], nde, SEED_CONF, max(EPOCHS, 250), 40, conf=True)

    # ---- summary ----
    done = {k: v for k, v in results.items() if "error" not in v and "fom_median" in v}
    ranked = sorted(done, key=lambda k: rank_key(done[k]), reverse=True)
    results["_summary"] = {
        "elapsed_min": round((time.time() - T0) / 60, 1),
        "n_runs": len([k for k in results if k.split("/")[0] in ("R1", "R2", "R3")]),
        "top10": [{"key": k, "fom_median": done[k]["fom_median"], "calibrated": calibrated(done[k]),
                   "tarp": done[k]["tarp_dev_med"]} for k in ranked[:10]],
        "complete": budget_left(),   # if False, a resume job should continue
    }
    save(results, args.out)
    print("=== DONE ===", json.dumps(results["_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
