#!/usr/bin/env python
"""Follow-up to the overnight HOS-NDE search:
  (1) Regularize NSF into calibration — NSF had big FoM headroom (up to 2233) but was systematically
      OVERCONFIDENT (SBC~0, TARP~0.06). Sweep dropout / weight_decay / fewer spline bins / smaller
      nets to keep the FoM while restoring calibration. This is the real route to beating ~1666.
  (2) Full-epoch (500) confirm of the best CALIBRATED MAF configs (the overnight R3 ran 250 ep and
      dipped slightly — check it isn't epoch-limited).
Reuses hos_nde_search.run_nde on the already-cached analytical stats. Resumable + time-aware.
"""
import argparse, json, time
from pathlib import Path
import torch
import hos_nde_search as H

T0 = time.time()


def nde(**kw):
    d = dict(H.BASE_NDE); d.update(kw); return d


# NSF regularization candidates (all flow_type="nsf") — from the overconfident deep/wide winners
# down to smaller/heavily-regularized variants.
NSF_REG = [
    ("nsf_dw_reg",  nde(flow_type="nsf", n_transforms=8, hidden_dim=128, made_dropout=0.2,
                        embedding_dropout=0.2, weight_decay=1e-4)),
    ("nsf_dw_reg2", nde(flow_type="nsf", n_transforms=8, hidden_dim=128, made_dropout=0.3,
                        embedding_dropout=0.2, weight_decay=1e-4, num_bins=5)),
    ("nsf_md_reg",  nde(flow_type="nsf", n_transforms=6, hidden_dim=96, made_dropout=0.15,
                        embedding_dropout=0.15, weight_decay=1e-4, num_bins=6)),
    ("nsf_sm_reg",  nde(flow_type="nsf", n_transforms=5, hidden_dim=64, made_dropout=0.2,
                        embedding_dropout=0.2, weight_decay=1e-4, num_bins=5)),
]
COMBOS_REG = ["ps_hos_scat", "hos_scat_wide"]      # highest NSF headroom in the night run

# Full-epoch confirms of the best CALIBRATED MAF configs from the night.
MAF_CONF = [
    ("maf_deep_fullep", "hos_scat_wide", nde(flow_type="maf", n_transforms=8, hidden_dim=128)),
    ("maf_drop_fullep", "ps_hos_scat",   nde(flow_type="maf", n_transforms=5, hidden_dim=64, made_dropout=0.1)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout_path", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time_budget_sec", type=float, default=9000.0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = json.load(open(a.out)) if Path(a.out).exists() else {}

    def left():
        return (time.time() - T0) < a.time_budget_sec

    caches = {}
    for c in set(COMBOS_REG + [m[1] for m in MAF_CONF]):
        caches[c] = H.precompute(c, a.holdout_path, a.cache_dir, 0, device)  # cached => instant

    def do(key, cache, ndecfg, seeds, epochs, conf):
        if key in results or not left():
            return
        print(f"[{(time.time()-T0)/60:.0f}m] {key}  ({ndecfg['flow_type']} t{ndecfg['n_transforms']} "
              f"h{ndecfg['hidden_dim']} md{ndecfg.get('made_dropout')} wd{ndecfg.get('weight_decay',0)} "
              f"nb{ndecfg.get('num_bins',8)}) seeds={seeds} ep={epochs}", flush=True)
        try:
            nfs, ncal, lcal = (10000, 2000, 1000) if conf else (4000, 1500, 800)
            r = H.run_nde(cache, ndecfg, seeds, epochs, 40 if conf else 30, 1e-4,
                          nfs, ncal, lcal, 300.0, 3000.0, 0.4, 3, device)
            r["nde"] = {k: ndecfg.get(k) for k in ("flow_type", "n_transforms", "hidden_dim",
                        "embedding_dim", "made_dropout", "embedding_dropout", "weight_decay",
                        "num_bins", "context_norm")}
            results[key] = r; H.save(results, a.out)
            print(f"   -> FoM {r['fom_median']:.0f} (mean {r['fom_mean']:.0f}+/-{r['fom_std']:.0f}, "
                  f"kept {r['n_kept']}/{seeds}) TARP {r['tarp_dev_med']:.3f} "
                  f"SBC {r['sbc_om_p_med']:.2f}/{r['sbc_s8_p_med']:.2f} "
                  f"{'CAL' if H.calibrated(r) else 'uncal'}", flush=True)
        except Exception as e:
            results[key] = {"error": f"{type(e).__name__}: {e}"}; H.save(results, a.out)
            print(f"   !! {key} FAILED: {e}", flush=True)

    # (1) NSF regularization screen (8 seeds, 250 ep)
    for c in COMBOS_REG:
        for gid, ndecfg in NSF_REG:
            do(f"NSFREG/{c}/{gid}", caches[c], ndecfg, 8, 250, False)
    # (2) full-epoch MAF confirm (20 seeds, 500 ep)
    for gid, c, ndecfg in MAF_CONF:
        do(f"MAFCONF/{gid}", caches[c], ndecfg, 20, 500, True)
    # (3) confirm the best CALIBRATED NSF-reg at 20 seeds / 500 ep
    reg = {k: v for k, v in results.items()
           if k.startswith("NSFREG/") and isinstance(v, dict) and "fom_median" in v and H.calibrated(v)}
    if reg:
        best = max(reg, key=lambda k: reg[k]["fom_median"])
        _, c, gid = best.split("/")
        do(f"NSFCONF/{c}/{gid}", caches[c], dict(NSF_REG)[gid], 20, 500, True)

    done = {k: v for k, v in results.items() if isinstance(v, dict) and "fom_median" in v}
    ranked = sorted(done, key=lambda k: done[k]["fom_median"] * (1 if H.calibrated(done[k]) else 0.5),
                    reverse=True)
    results["_summary"] = {"elapsed_min": round((time.time() - T0) / 60, 1), "complete": left(),
                           "top": [{"key": k, "fom": round(done[k]["fom_median"]),
                                    "cal": H.calibrated(done[k]), "tarp": round(done[k]["tarp_dev_med"], 3)}
                                   for k in ranked[:12]]}
    H.save(results, a.out)
    print("=== FOLLOWUP DONE ===", json.dumps(results["_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
