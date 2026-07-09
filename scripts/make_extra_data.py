#!/usr/bin/env python
"""Generate the remaining artifact data:
  (a) CALIBRATION — TARP coverage + SBC ranks for HOS and CNN on the holdout validation split.
  (b) AVERAGING DIAGNOSTIC — FoM as k single-realization summaries are averaged toward the
      noiseless mean (k=1..10), for HOS vs CNN. Explains why the HOS 'average observation' widens
      (out-of-distribution: the flow was trained on single noisy vectors).
Saves extra_data.npz.
"""
import math, numpy as np, torch
from datasets import load_from_disk
from cosmoford import THETA_MEAN, THETA_STD
from cosmoford.models_nopatch import build_flow, RegressionModelNoPatch
import run_calibration as rc
from tarp import get_tarp_coverage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOLD = "/project/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout"
ND = "/project/rrg-lplevass/shared/wl_chall_data/sammy_data_posteriors/budget_scan_nbody_final"
CACHE = "/home/tersenov/experiments/hos_night/cache/stats_hos_scat_wide.npz"
OUT = "/home/tersenov/experiments/hos_night/extra_data.npz"
MEAN, STD = THETA_MEAN[:2], THETA_STD[:2]


def fom(s):  # (N,2) physical
    c = np.cov(s.T); d = np.linalg.det(c)
    return 1.0 / np.sqrt(d) if d > 0 else np.nan


def calib(flow, norm, s_va, t_va, ncal=1500, L=800):
    n = min(ncal, len(s_va))
    out = []
    with torch.no_grad():
        for i in range(n):
            out.append(flow.sample(L, context=norm(s_va[i:i+1].to(device))).squeeze(0).cpu().numpy())
    samp = np.stack(out)                                  # (n, L, 2) normalized
    truth = (t_va[:n].numpy() if torch.is_tensor(t_va) else t_va[:n])
    ecp, alpha = get_tarp_coverage(samp.transpose(1, 0, 2), truth, references="random",
                                   metric="euclidean", norm=True)
    ranks = rc.sbc_ranks(samp, truth)
    return np.asarray(ecp), np.asarray(alpha), np.asarray(ranks), L


def avg_diag(flow, norm, Sfid, ks=(1, 2, 3, 5, 10)):
    res = {}
    for k in ks:
        fs = []
        for start in range(0, 10 - k + 1):                # sliding windows of size k
            m = Sfid[start:start+k].mean(0, keepdim=True)
            with torch.no_grad():
                s = flow.sample(8000, context=norm(m.to(device))).squeeze(0).cpu().numpy()
            fs.append(fom(s * STD + MEAN))
        res[k] = float(np.nanmean(fs))
    return res


class Cfg:
    context_normalization = True


ident = lambda s: s

# ---------------- HOS ----------------
z = np.load(CACHE); S = torch.from_numpy(z["S"]); T = torch.from_numpy(z["T"]); Sfid = torch.from_numpy(z["Sfid"])
s_tr, t_tr, s_va, t_va, norm = rc.split_and_norm(S, T, Cfg)
ctx_dim = S.shape[1]
tl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(s_tr, t_tr), batch_size=128, shuffle=True)
vl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(s_va, t_va), batch_size=256)


def mk():
    return build_flow(2, ctx_dim, n_transforms=8, hidden_dim=128, flow_type="maf",
                      embedding_dim=128, embedding_hidden=256, embedding_dropout=0.1).to(device)


def train(seed, epochs=400):
    torch.manual_seed(seed); flow = mk(); opt = torch.optim.Adam(flow.parameters(), 1e-4)
    warm = max(1, int(0.05 * epochs))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/warm if e < warm else 0.5*(1+math.cos(math.pi*(e-warm)/max(1, epochs-warm))))
    bv, pat, bs = 1e9, 0, None
    for _ in range(epochs):
        flow.train()
        for sb, tb in tl:
            loss = -flow.log_prob(tb.to(device), context=norm(sb.to(device))).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
        sch.step(); flow.eval()
        with torch.no_grad():
            mv = float(np.mean([(-flow.log_prob(tb.to(device), context=norm(sb.to(device))).mean()).item() for sb, tb in vl]))
        if mv < bv:
            bv, pat, bs = mv, 0, {k: v.cpu().clone() for k, v in flow.state_dict().items()}
        else:
            pat += 1
            if pat >= 40:
                break
    return bs, bv


best = None
for seed in range(6):
    bs, bv = train(92 + seed)
    print(f"HOS seed {seed} val {bv:.4f}", flush=True)
    if best is None or bv < best[1]:
        best = (bs, bv)
hos = mk(); hos.load_state_dict(best[0]); hos.eval()
hos_ecp, hos_alpha, hos_ranks, hos_L = calib(hos, norm, s_va, t_va)
hos_avg = avg_diag(hos, norm, Sfid)
print("HOS TARP dev %.3f ; avg-diag %s" % (float(np.max(np.abs(hos_ecp - hos_alpha))), hos_avg), flush=True)

# ---------------- CNN ----------------
cnn_ckpt = rc.find_best_checkpoint(ND, 20200)
cnn = RegressionModelNoPatch.load_from_checkpoint(cnn_ckpt, map_location=device).eval().to(device)
for p in cnn.parameters():
    p.requires_grad = False
cnn_flow = rc.load_saved_flow(f"{ND}/npe_results/budget-20200/npe_flow.pt", 8, 4, 64, device)

# CNN val summaries on the SAME val maps (same seed-0 split, same order as the cached HOS context)
dd = load_from_disk(HOLD); tr = dd["train"].with_format("numpy")
kap = np.array(tr["kappa"]); th = np.array(tr["theta"])
n = len(S); nv = n // 10
perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
val_idx = perm[n - nv:].numpy()
val_kap, val_th = kap[val_idx], th[val_idx]
Sval = rc.build_context(cnn, val_kap, "compressed", device)         # (nval, 8)
Tval = torch.from_numpy(((val_th[:, :2] - MEAN) / STD).astype(np.float32))
cnn_ecp, cnn_alpha, cnn_ranks, cnn_L = calib(cnn_flow, ident, Sval, Tval)
# CNN fiducial summaries for the averaging diagnostic
fid = dd["fiducial"].with_format("numpy"); fk = np.array(fid["kappa"])
Sfid_cnn = rc.build_context(cnn, fk, "compressed", device)          # (10, 8)
cnn_avg = avg_diag(cnn_flow, ident, Sfid_cnn)
print("CNN TARP dev %.3f ; avg-diag %s" % (float(np.max(np.abs(cnn_ecp - cnn_alpha))), cnn_avg), flush=True)

np.savez(OUT, hos_ecp=hos_ecp, hos_alpha=hos_alpha, hos_ranks=hos_ranks, hos_L=hos_L,
         cnn_ecp=cnn_ecp, cnn_alpha=cnn_alpha, cnn_ranks=cnn_ranks, cnn_L=cnn_L,
         hos_avg_k=np.array(list(hos_avg.keys())), hos_avg_f=np.array(list(hos_avg.values())),
         cnn_avg_k=np.array(list(cnn_avg.keys())), cnn_avg_f=np.array(list(cnn_avg.values())))
print("saved", OUT, flush=True)
