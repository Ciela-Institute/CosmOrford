#!/usr/bin/env python
"""Generate posterior samples for the artifact contour plots:
  HOS champion (raw HOS+Scat -> jointly-trained embedding NDE, no separate compressor) vs
  CNN (biggest-budget map-level neural summaries, budget-20200),
for each of the 10 fiducial observations + the AVERAGE (mean-summary, ~noiseless) observation.
Saves everything in physical (Om, S8) units. Validates the CNN regeneration against the team's
saved posterior_samples.npy.
"""
import math, numpy as np, torch
from datasets import load_from_disk
from cosmoford import THETA_MEAN, THETA_STD
from cosmoford.dataset import reshape_field_numpy
from cosmoford.models_nopatch import build_flow, RegressionModelNoPatch
import run_calibration as rc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOLD = "/project/rrg-lplevass/shared/wl_chall_data/neurips-wl-challenge-holdout"
ND = "/project/rrg-lplevass/shared/wl_chall_data/sammy_data_posteriors/budget_scan_nbody_final"
CNN_FLOW = f"{ND}/npe_results/budget-20200/npe_flow.pt"
CNN_SAVED = f"{ND}/npe_results/budget-20200/posterior_samples.npy"
CACHE = "/home/tersenov/experiments/hos_night/cache/stats_hos_scat_wide.npz"
OUT = "/home/tersenov/experiments/hos_night/contour_data.npz"
N = 10000


def fom(per):  # per (nobs, N, 2) physical
    fs = []
    for p in per:
        c = np.cov(p.T); d = np.linalg.det(c)
        fs.append(1.0 / np.sqrt(d) if d > 0 else np.nan)
    return float(np.nanmean(fs))


class Cfg:
    context_normalization = True


# ---------------- data ----------------
dd = load_from_disk(HOLD)
fid = dd["fiducial"].with_format("numpy")
fid_kappa = np.array(fid["kappa"]); truth = np.array(fid["theta"])[0, :2].astype(float)
nobs = len(fid_kappa)
print(f"{nobs} fiducial obs; truth (Om,S8)={truth}", flush=True)

# ---------------- HOS champion ----------------
z = np.load(CACHE)
S = torch.from_numpy(z["S"]); T = torch.from_numpy(z["T"]); Sfid = torch.from_numpy(z["Sfid"])
s_tr, t_tr, s_va, t_va, norm = rc.split_and_norm(S, T, Cfg)
ctx_dim = S.shape[1]
tl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(s_tr, t_tr), batch_size=128, shuffle=True)
vl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(s_va, t_va), batch_size=256)


def mk():
    return build_flow(2, ctx_dim, n_transforms=8, hidden_dim=128, flow_type="maf",
                      embedding_dim=128, embedding_hidden=256, embedding_dropout=0.1).to(device)


def train_one(seed, epochs=400):
    torch.manual_seed(seed); flow = mk(); opt = torch.optim.Adam(flow.parameters(), 1e-4)
    warm = max(1, int(0.05 * epochs))
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm))))
    bv, pat, bs = 1e9, 0, None
    for _ in range(epochs):
        flow.train()
        for sb, tb in tl:
            loss = -flow.log_prob(tb.to(device), context=norm(sb.to(device))).mean()
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
            if pat >= 40:
                break
    return bs, bv


best = None
for seed in range(6):
    bs, bv = train_one(92 + seed)
    print(f"  HOS seed {seed}: val nll {bv:.4f}", flush=True)
    if best is None or bv < best[1]:
        best = (bs, bv)
flow = mk(); flow.load_state_dict(best[0]); flow.eval()


def hsample(ctx_row):  # ctx_row (1,D) raw -> physical samples
    with torch.no_grad():
        s = flow.sample(N, context=norm(ctx_row.to(device))).squeeze(0).cpu().numpy()
    return s * THETA_STD[:2] + THETA_MEAN[:2]


hos_per = np.stack([hsample(Sfid[i:i+1]) for i in range(nobs)])
hos_avg = hsample(Sfid.mean(0, keepdim=True))
print(f"HOS FoM (per-obs) = {fom(hos_per):.0f}", flush=True)

# ---------------- CNN (biggest budget) ----------------
cnn_ckpt = rc.find_best_checkpoint(ND, 20200)
print(f"CNN compressor: {cnn_ckpt}", flush=True)
cnn = RegressionModelNoPatch.load_from_checkpoint(cnn_ckpt, map_location=device).eval().to(device)
for p in cnn.parameters():
    p.requires_grad = False
cnn_flow = rc.load_saved_flow(CNN_FLOW, 8, 4, 64, device)


def cnn_summary(ki):
    kr = reshape_field_numpy(ki[np.newaxis])[0]
    x = torch.from_numpy(kr.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        return cnn.compress(x)


sums = torch.cat([cnn_summary(fid_kappa[i]) for i in range(nobs)], 0)


def csample(row):
    with torch.no_grad():
        s = cnn_flow.sample(N, context=row.to(device)).squeeze(0).cpu().numpy()
    return s * THETA_STD[:2] + THETA_MEAN[:2]


cnn_per = np.stack([csample(sums[i:i+1]) for i in range(nobs)])
cnn_avg = csample(sums.mean(0, keepdim=True))
print(f"CNN FoM (per-obs) = {fom(cnn_per):.0f}", flush=True)

# validate against the team's saved posteriors
saved = np.load(CNN_SAVED)
print(f"CNN regen mean {cnn_per.reshape(-1,2).mean(0)} vs saved mean {saved.reshape(-1,2).mean(0)} "
      f"(saved FoM {fom(saved):.0f})", flush=True)

np.savez(OUT, hos_per=hos_per, hos_avg=hos_avg, cnn_per=cnn_per, cnn_avg=cnn_avg,
         cnn_saved=saved, truth=truth)
print(f"saved {OUT}", flush=True)
