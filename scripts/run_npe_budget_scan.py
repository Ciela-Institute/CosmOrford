"""NPE training + FoM vs budget scan.

For each budget checkpoint from the compressor budget scan:
1. Load frozen compressor
2. Pre-compute summaries with noise augmentation on holdout data
3. Train NPE (conditional MAF) on (summary, theta) pairs
4. Evaluate FoM at fiducial cosmology

Usage (Modal):
    .venv/bin/modal run scripts/run_npe_budget_scan.py

Usage (local):
    python scripts/run_npe_budget_scan.py \\
        --checkpoints_path /path/to/checkpoints \\
        --npe_results_path /path/to/npe_results \\
        --summaries_cache_path /path/to/summaries_cache \\
        --holdout_path /path/to/neurips-wl-challenge-holdout \\
        --config configs/experiments/npe_budget_scan.yaml \\
        [--budgets 100,200,500]

    All training hyperparameters are read from --config (YAML file).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from glob import glob


@dataclass
class NPEConfig:
    budgets: List[int] = field(default_factory=lambda: [100, 200, 500, 1000, 2000, 5000, 10000, 20200])
    n_noise_realizations: int = 1
    npe_epochs: int = 500
    npe_lr: float = 1e-3
    npe_batch_size: int = 512
    npe_patience: int = 50
    npe_seeds: int = 5
    n_posterior_samples: int = 10_000
    # Calibration validation (TARP + MIRA) on the flat dataset validation split
    n_val_maps: int = 500
    n_val_posterior_samples: int = 1000

    @classmethod
    def from_yaml(cls, path: str) -> "NPEConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def find_best_checkpoint(budget: int, checkpoints_path: Path, offline: bool = False) -> str:
    """Find the best compressor checkpoint for a given budget.

    Strategy:
    1. Parse val_mse from Lightning checkpoint filenames, pick lowest
    2. Fall back to last.ckpt
    3. Fall back to W&B artifact download (skipped when offline=True)
    """
    import re

    checkpoint_dir = checkpoints_path / f"budget-{budget}"
    checkpoints = str(checkpoint_dir)
    # print(checkpoints)
    checkpoints = glob(checkpoints + "/**/*.ckpt", recursive=True)

    # for ckpt in checkpoint_dir.glob("*.ckpt"): 
    #     print(ckpt)
    # Strategy 1: Parse val_mse from checkpoint filenames
    if checkpoint_dir.exists():
        best_path = None
        best_logp = float("inf")

        for ckpt in checkpoints:
            print(ckpt)
            if ckpt == "last.ckpt":
                continue
            match = re.search(r"val_(?:log_prob|nll)=([-+]?\d*\.?\d+)", ckpt)
            if match:
                logp = float(match.group(1))
                if logp < best_logp:
                    best_logp = logp
                    best_path = str(ckpt)

        if best_path is not None:
            print(f"Found best checkpoint for budget-{budget}: {best_path} (val_log_prob={best_logp:.6f})")
            return best_path

        # Strategy 2: Fall back to last.ckpt (skip if empty/corrupted)
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists() and last_ckpt.stat().st_size > 0:
            print(f"Using last.ckpt for budget-{budget}")
            return str(last_ckpt)

    if offline:
        raise FileNotFoundError(
            f"No local checkpoint found for budget-{budget} in {checkpoint_dir} "
            f"and --offline is set (W&B download disabled)."
        )

    # Strategy 3: W&B fallback
    print(f"No local checkpoint for budget-{budget}, trying W&B...")
    import wandb

    api = wandb.Api()
    runs = api.runs(
        "cosmostat/neurips-wl-challenge",
        filters={"tags": "budget-scan", "display_name": f"budget-{budget}"},
    )
    for run in runs:
        for art in run.logged_artifacts():
            if art.type == "model":
                art_dir = art.download(root=str(checkpoint_dir))
                for f in Path(art_dir).glob("**/*.ckpt"):
                    print(f"Downloaded W&B artifact for budget-{budget}: {f}")
                    return str(f)

    raise FileNotFoundError(f"No checkpoint found for budget-{budget}")


def _train_budget_core(budget: int, checkpoints_path, npe_results_path, summaries_cache_path, load_holdout, cfg: NPEConfig, offline: bool = False, vol=None, load_flat_val=None, context_normalization: bool = False):
    """Core NPE pipeline, independent of Modal or local execution.

    Args:
        budget: simulation budget
        checkpoints_path: Path to compressor checkpoints root
        npe_results_path: Path where NPE results will be written
        summaries_cache_path: Path for caching pre-computed summaries
        load_holdout: callable(split: str) -> HuggingFace Dataset
        cfg: NPEConfig with all training hyperparameters
        offline: if True, disable W&B checkpoint fallback and raise if no local checkpoint found
        vol: optional modal.Volume; if provided, .reload()/.commit() are called around I/O
        context_normalization: if True, enable context normalization
    """
    import json

    import numpy as np
    import torch
    import torch.nn.functional as F

    from cosmoford import NOISE_STD, THETA_MEAN, THETA_STD
    from cosmoford.dataset import reshape_field_numpy
    from cosmoford.models_nopatch import RegressionModelNoPatch, build_flow

    import os
    import wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Budget {budget}: starting NPE pipeline")
    print(f"{'='*60}")
    print(f"Config: {cfg}")

    _wandb_config = {
        "budget": budget,
        "npe_epochs": cfg.npe_epochs,
        "npe_lr": cfg.npe_lr,
        "npe_batch_size": cfg.npe_batch_size,
        "npe_patience": cfg.npe_patience,
        "npe_seeds": cfg.npe_seeds,
        "n_noise_realizations": cfg.n_noise_realizations,
        "n_posterior_samples": cfg.n_posterior_samples,
    }
    _wandb_kwargs = dict(
        entity="cosmostat",
        project="neurips-wl-challenge",
        mode=os.environ.get("WANDB_MODE", "offline"),
        config=_wandb_config,
        dir=str(npe_results_path),
    )

    if vol is not None:
        vol.reload()

    # ── 1-3. Load or compute summaries ──
    cache_dir = summaries_cache_path / f"budget-{budget}"
    cache_file = cache_dir / f"summaries_n{cfg.n_noise_realizations}.pt"
    compressor = None  # loaded lazily for FoM eval

    if cache_file.exists():
        print(f"Loading cached summaries from {cache_file}")
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        summaries_tensor = cached["summaries"]
        thetas_tensor = cached["thetas"]
        theta_all = cached["theta_all_raw"]  # unnormalized, for FoM eval
        ckpt_path = cached["compressor_checkpoint"]
        print(f"Loaded {summaries_tensor.shape[0]} cached pairs (compressor: {ckpt_path})")
    else:
        print("No cached summaries found, computing from scratch...")

        # ── 1. Load frozen compressor ──
        ckpt_path = find_best_checkpoint(budget, checkpoints_path, offline=offline)
        compressor = RegressionModelNoPatch.load_from_checkpoint(ckpt_path, map_location=device)
        compressor.eval()
        compressor.to(device)
        for p in compressor.parameters():
            p.requires_grad = False
        print(f"Loaded compressor from {ckpt_path}")

        # ── 2. Load holdout dataset ──
        print("Loading holdout dataset...")
        holdout = load_holdout("train")
        holdout = holdout.with_format("numpy")

        kappa_all = np.array(holdout["kappa"])  # (N, 1424, 176)
        theta_all = np.array(holdout["theta"])  # (N, 5)

        # Normalize theta to match training (only Omega_m, S_8)
        theta_norm = (theta_all[:, :2] - THETA_MEAN[:2]) / THETA_STD[:2]
        theta_norm = theta_norm.astype(np.float32)

        n_maps = len(kappa_all)
        print(f"Holdout: {n_maps} maps")

        # Build mask (same as model uses)
        from cosmoford import SURVEY_MASK
        mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

        # ── 3. Pre-compute summaries with noise augmentation ──
        print(f"Pre-computing summaries ({cfg.n_noise_realizations} noise realizations per map)...")
        all_summaries = []
        all_thetas = []

        with torch.no_grad():
            for i in range(n_maps):
                kappa_i = kappa_all[i]  # (1424, 176)
                kappa_reshaped = reshape_field_numpy(kappa_i[np.newaxis])[0]  # (1834, 88)

                for _ in range(cfg.n_noise_realizations):
                    noisy = kappa_reshaped # kappa maps from the holdout dataset are already noisy and masked
                    x = torch.from_numpy(noisy).unsqueeze(0).to(device)  # (1, 1834, 88)
                    s = compressor.compress(x)  # (1, 8)
                    all_summaries.append(s.cpu())
                    all_thetas.append(theta_norm[i])

                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i+1}/{n_maps} maps")

        summaries_tensor = torch.cat(all_summaries, dim=0)  # (N*n_noise, 8)
        thetas_tensor = torch.from_numpy(np.array(all_thetas))  # (N*n_noise, 2)

        # Cache to disk
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "summaries": summaries_tensor,
            "thetas": thetas_tensor,
            "theta_all_raw": theta_all,
            "compressor_checkpoint": ckpt_path,
            "n_noise_realizations": cfg.n_noise_realizations,
        }, cache_file)
        if vol is not None:
            vol.commit()
        print(f"Cached summaries to {cache_file}")

    print(f"Summary dataset: {summaries_tensor.shape[0]} pairs")

    # ── 4. Train/val split (90/10) ──
    n_total = summaries_tensor.shape[0]
    n_val = n_total // 10
    n_train = n_total - n_val

    # Fixed split (seed=0) so all NPE seeds share the same train/val data
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(0))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    s_train, t_train = summaries_tensor[train_idx], thetas_tensor[train_idx]
    s_val, t_val = summaries_tensor[val_idx], thetas_tensor[val_idx]
    print(f"Train: {n_train}, Val: {n_val}")

    # Context normalization stats, computed once from the training split. Must be applied
    # consistently everywhere a context summary is fed to the flow (train, val, FoM eval,
    # TARP/MIRA calibration) — not just during training — since the flow is trained to expect
    # normalized context when context_normalization=True.
    ctx_mean = s_train.mean(dim=0, keepdim=True) if context_normalization else None
    ctx_std = s_train.std(dim=0, keepdim=True) if context_normalization else None

    def _normalize_context(s):
        if not context_normalization:
            return s
        return (s - ctx_mean.to(s.device)) / ctx_std.to(s.device)

    train_dataset = torch.utils.data.TensorDataset(s_train, t_train)
    val_dataset = torch.utils.data.TensorDataset(s_val, t_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.npe_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.npe_batch_size)

    # ── 5. Train NPE (multiple seeds, keep best) ──
    results_dir = npe_results_path / f"budget-{budget}"
    results_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(name=f"npe_budget_{budget}", **_wandb_kwargs)
    overall_best_nll = float("inf")
    overall_best_state = None

    for seed in range(cfg.npe_seeds):
        print(f"\n--- NPE seed {seed+1}/{cfg.npe_seeds} ---")

        torch.manual_seed(seed + 42)
        flow = build_flow(param_dim=2, context_dim=8).to(device)
        optimizer = torch.optim.Adam(flow.parameters(), lr=cfg.npe_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.npe_epochs)

        best_val_nll = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(cfg.npe_epochs):
            # Train
            flow.train()
            train_losses = []
            for s_batch, t_batch in train_loader:
                s_batch, t_batch = s_batch.to(device), t_batch.to(device)
                s_batch = _normalize_context(s_batch)
                loss = -flow.log_prob(t_batch, context=s_batch).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()

            # Validate
            flow.eval()
            val_losses = []
            with torch.no_grad():
                for s_batch, t_batch in val_loader:
                    s_batch, t_batch = s_batch.to(device), t_batch.to(device)
                    s_batch = _normalize_context(s_batch)
                    val_losses.append(-flow.log_prob(t_batch, context=s_batch).mean().item())

            mean_train = np.mean(train_losses)
            mean_val = np.mean(val_losses)

            wandb.log({
                f"seed{seed+1}/train_nll": mean_train,
                f"seed{seed+1}/val_nll": mean_val,
                f"seed{seed+1}/lr": optimizer.param_groups[0]['lr'],
                f"seed{seed+1}/epoch": epoch,
            })

            if mean_val < best_val_nll:
                best_val_nll = mean_val
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 50 == 0 or patience_counter == 0:
                print(f"  Epoch {epoch+1:3d}: train_nll={mean_train:.4f}, val_nll={mean_val:.4f}, "
                      f"best={best_val_nll:.4f}, patience={patience_counter}/{cfg.npe_patience}")

            if patience_counter >= cfg.npe_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        print(f"  Seed {seed+1} best val NLL: {best_val_nll:.4f}")
        wandb.log({f"seed{seed+1}/best_val_nll": best_val_nll})

        if best_val_nll < overall_best_nll:
            overall_best_nll = best_val_nll
            overall_best_state = best_state

    # Restore overall best model
    flow = build_flow(param_dim=2, context_dim=8).to(device)
    flow.load_state_dict(overall_best_state)
    flow.eval()
    best_val_nll = overall_best_nll
    print(f"\nOverall best val NLL across {cfg.npe_seeds} seeds: {best_val_nll:.4f}")

    # ── 6. Compute FoM at fiducial cosmology ──
    print("Computing FoM (all fiducial maps)...")

    # Load compressor for FoM eval if not already loaded (cache path)
    if compressor is None:
        compressor = RegressionModelNoPatch.load_from_checkpoint(ckpt_path, map_location=device)
        compressor.eval()
        compressor.to(device)
        for p in compressor.parameters():
            p.requires_grad = False

    # Load fiducial kappa maps directly (split='fiducial' contains only fiducial cosmology maps)
    print("Loading fiducial maps for FoM evaluation...")
    holdout = load_holdout("fiducial")
    holdout = holdout.with_format("numpy")
    kappa_all_fom = np.array(holdout["kappa"])
    theta_all_fom = np.array(holdout["theta"])
    print(f"  Using {len(kappa_all_fom)} fiducial maps")

    from cosmoford import SURVEY_MASK
    mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    theta_mean_t = torch.tensor(THETA_MEAN[:2], device=device, dtype=torch.float32)
    theta_std_t = torch.tensor(THETA_STD[:2], device=device, dtype=torch.float32)

    fom_values = []
    mse_values = []  # compressor regression-head MSE, same formula as RegressionModelNoPatch.validation_step
    score_values = []  # compressor regression-head score, same formula as RegressionModelNoPatch.validation_step
    all_posterior_samples = []  # (n_maps, n_posterior_samples, 2) in physical units
    with torch.no_grad():
        for i, kappa_i in enumerate(kappa_all_fom):
            kappa_reshaped = reshape_field_numpy(kappa_i[np.newaxis])[0]
            noisy = kappa_reshaped
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            mean_raw, std_raw, s = compressor(x)  # mean/std in normalized parameter space, s: (1, 8)
            s = _normalize_context(s)

            # Sample posterior
            samples = flow.sample(cfg.n_posterior_samples, context=s)  # (1, N, 2)
            samples = samples.squeeze(0).cpu().numpy()  # (N, 2)

            # Unnormalize to physical parameters
            samples_phys = samples * THETA_STD[:2] + THETA_MEAN[:2]
            all_posterior_samples.append(samples_phys)

            # Compute FoM = 1 / sqrt(det(Cov))
            cov = np.cov(samples_phys.T)  # (2, 2)
            det = np.linalg.det(cov)
            if det > 0:
                fom = 1.0 / np.sqrt(det)
            else:
                fom = 0.0
            fom_values.append(fom)

            # Compressor regression-head mse/score against the true fiducial theta
            y_phys = torch.tensor(theta_all_fom[i, :2], device=device, dtype=torch.float32).unsqueeze(0)
            mean_phys = mean_raw * theta_std_t + theta_mean_t
            std_phys = std_raw * theta_std_t
            sq_error = (y_phys - mean_phys) ** 2
            score_i = -torch.sum(sq_error / std_phys**2 + torch.log(std_phys**2) + 1000.0 * sq_error, dim=1)
            mse_i = F.mse_loss(mean_phys, y_phys)
            score_values.append(score_i.item())
            mse_values.append(mse_i.item())

    fom_mean = np.mean(fom_values)
    fom_std = np.std(fom_values)
    print(f"  FoM = {fom_mean:.2f} ± {fom_std:.2f}")

    mse_mean = np.mean(mse_values)
    mse_std = np.std(mse_values)
    score_mean = np.mean(score_values)
    score_std = np.std(score_values)
    print(f"  MSE = {mse_mean:.6f} ± {mse_std:.6f}")
    print(f"  Score = {score_mean:.2f} ± {score_std:.2f}")

    # ── 7. Save results ──
    torch.save(best_state, results_dir / "npe_flow.pt")

    # Save posterior samples: shape (n_fiducial_maps, n_posterior_samples, 2)
    # columns: [Omega_m, S_8] in physical units
    posterior_samples_arr = np.stack(all_posterior_samples, axis=0)
    np.save(results_dir / "posterior_samples.npy", posterior_samples_arr)
    print(f"Posterior samples saved to {results_dir / 'posterior_samples.npy'} "
          f"(shape {posterior_samples_arr.shape})")

    results = {
        "budget": budget,
        "fom_mean": float(fom_mean),
        "fom_std": float(fom_std),
        "fom_values": [float(v) for v in fom_values],
        "mse_mean": float(mse_mean),
        "mse_std": float(mse_std),
        "mse_values": [float(v) for v in mse_values],
        "score_mean": float(score_mean),
        "score_std": float(score_std),
        "score_values": [float(v) for v in score_values],
        "best_val_nll": float(best_val_nll),
        "n_noise_realizations": cfg.n_noise_realizations,
        "n_fiducial_maps": len(kappa_all_fom),
        "n_posterior_samples": cfg.n_posterior_samples,
        "compressor_checkpoint": ckpt_path,
    }
    (results_dir / "results.json").write_text(json.dumps(results, indent=2))

    # ── 8. GetDist posterior plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots as gdplots

    # Prior bounds from the challenge dataset
    PRIOR_BOUNDS = {
        "Omega_m": (0.0913, 0.6190),
        "S_8":     (0.6801, 0.9552),
    }
    fiducial_omega_m = THETA_MEAN[0]
    fiducial_s8 = THETA_MEAN[1]

    # Build a flat prior MCSamples so getdist renders it as a box
    n_prior = 50_000
    prior_samples = np.column_stack([
        np.random.uniform(*PRIOR_BOUNDS["Omega_m"], size=n_prior),
        np.random.uniform(*PRIOR_BOUNDS["S_8"],     size=n_prior),
    ])
    mc_prior = MCSamples(
        samples=prior_samples,
        names=["Omega_m", "S_8"],
        labels=[r"\Omega_m", r"S_8"],
        label="Prior",
    )

    n_plot = min(5, len(all_posterior_samples))
    mc_list = [mc_prior]
    for i in range(n_plot):
        mc = MCSamples(
            samples=all_posterior_samples[i],
            names=["Omega_m", "S_8"],
            labels=[r"\Omega_m", r"S_8"],
            label=f"obs {i+1}",
        )
        mc_list.append(mc)

    g = gdplots.get_subplot_plotter()
    g.triangle_plot(mc_list, filled=False, legend_loc="upper right")
    # Mark fiducial cosmology on all subplots
    for ax in g.subplots.flat:
        if ax is not None:
            ax.axvline(fiducial_omega_m, color="k", ls="--", lw=0.8, alpha=0.6)
            ax.axhline(fiducial_s8, color="k", ls="--", lw=0.8, alpha=0.6)

    plot_path = results_dir / "posterior_plot.png"
    g.export(str(plot_path))
    print(f"Posterior plot saved to {plot_path}")

    wandb.log({
        "fom_mean": fom_mean,
        "fom_std": fom_std,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "score_mean": score_mean,
        "score_std": score_std,
        "best_val_nll": best_val_nll,
        "best_seed/posterior_plot": wandb.Image(str(plot_path)),
    })

    # ── 9. TARP + MIRA calibration tests ──
    # 9a: on the NPE's own held-out validation data (same distribution as training — sanity check)
    # 9b: on the flat dataset validation split (held-out maps, needs noise+mask)
    calibration_results = {}
    try:
        from tarp import get_tarp_coverage
        from mira_score import mira as mira_score
        import torch as _torch

        def _run_calibration(summaries_t, theta_norm, tag, n_cap=None):
            """Sample posteriors then run TARP + MIRA. Returns (ecp, alpha, mira_mean, mira_std)."""
            n = len(summaries_t) if n_cap is None else min(n_cap, len(summaries_t))
            summaries_t = _normalize_context(summaries_t[:n])
            theta_norm = theta_norm[:n]

            posterior = []
            with _torch.no_grad():
                for i in range(n):
                    s_i = summaries_t[i:i+1].to(device)
                    samps = flow.sample(cfg.n_val_posterior_samples, context=s_i)  # (1, S, 2)
                    posterior.append(samps.squeeze(0).cpu().numpy())
            posterior_arr = np.stack(posterior, axis=0)  # (n, S, 2)

            ecp, alpha = get_tarp_coverage(
                samples=posterior_arr.transpose(1, 0, 2),  # (S, n, 2)
                theta=theta_norm,
                references="random",
                metric="euclidean",
                norm=True,
            )
            posterior_torch = _torch.from_numpy(posterior_arr).unsqueeze(0)  # (1, n, S, 2)
            truth_torch = _torch.from_numpy(theta_norm.astype(np.float32))
            m_mean, m_std = mira_score(truth_torch, posterior_torch, num_runs=100, device=device)

            # TARP plot
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(alpha, ecp, lw=2, label="NPE")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
            ax.fill_between(alpha, alpha, ecp, alpha=0.15)
            ax.set_xlabel("Credibility level α")
            ax.set_ylabel("Expected coverage probability")
            ax.set_title(f"TARP [{tag}] budget={budget}, n={n}")
            ax.legend()
            tarp_path = results_dir / f"tarp_coverage_{tag}.png"
            fig.savefig(str(tarp_path), dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(f"  [{tag}] TARP max|ecp-α| = {np.max(np.abs(ecp - alpha)):.4f}  "
                  f"MIRA = {float(m_mean[0]):.4f} ± {float(m_std[0]):.4f}")
            wandb.log({
                f"calibration/{tag}/tarp_plot": wandb.Image(str(tarp_path)),
                f"calibration/{tag}/mira_score": float(m_mean[0]),
                f"calibration/{tag}/mira_std": float(m_std[0]),
                f"calibration/{tag}/tarp_max_deviation": float(np.max(np.abs(ecp - alpha))),
                f"calibration/{tag}/n_maps": n,
            })
            np.save(results_dir / f"tarp_ecp_{tag}.npy", ecp)
            np.save(results_dir / f"tarp_alpha_{tag}.npy", alpha)
            np.save(results_dir / f"val_posterior_samples_{tag}.npy", posterior_arr)
            return ecp, alpha, m_mean, m_std

        # 9a — NPE validation split (already computed, no reprocessing needed)
        print("\nCalibration 9a: NPE held-out validation data...")
        ecp_a, alpha_a, mira_mean_a, mira_std_a = _run_calibration(
            s_val.cpu(),
            t_val.cpu().numpy(),
            tag="npe_val",
            n_cap=cfg.n_val_maps,
        )
        calibration_results["npe_val"] = {
            "mira_mean": float(mira_mean_a[0]),
            "mira_std": float(mira_std_a[0]),
            "tarp_max_deviation": float(np.max(np.abs(ecp_a - alpha_a))),
        }

        # 9b — flat dataset validation split (raw maps: add noise + mask first)
        if load_flat_val is not None:
            print("\nCalibration 9b: flat dataset validation split...")
            from cosmoford import NOISE_STD as _NOISE_STD
            _mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])
            rng_val = np.random.default_rng(99)

            flat_val = load_flat_val("validation").with_format("numpy")
            n_flat = min(cfg.n_val_maps, len(flat_val))
            kappa_flat = np.array(flat_val[:n_flat]["kappa"])
            theta_flat = np.array(flat_val[:n_flat]["theta"])
            theta_flat_norm = ((theta_flat[:, :2] - THETA_MEAN[:2]) / THETA_STD[:2]).astype(np.float32)

            flat_summaries = []
            with _torch.no_grad():
                for kappa_i in kappa_flat:
                    kappa_rs = reshape_field_numpy(kappa_i[np.newaxis])[0]
                    kappa_noisy = (kappa_rs + rng_val.standard_normal(kappa_rs.shape) * _NOISE_STD) * _mask
                    x = _torch.from_numpy(kappa_noisy.astype(np.float32)).unsqueeze(0).to(device)
                    flat_summaries.append(compressor.compress(x).cpu())
            flat_summaries_t = _torch.cat(flat_summaries, dim=0)

            ecp_b, alpha_b, mira_mean_b, mira_std_b = _run_calibration(
                flat_summaries_t, theta_flat_norm, tag="flat_val"
            )
            calibration_results["flat_val"] = {
                "mira_mean": float(mira_mean_b[0]),
                "mira_std": float(mira_std_b[0]),
                "tarp_max_deviation": float(np.max(np.abs(ecp_b - alpha_b))),
            }

    except Exception as e:
        print(f"WARNING: TARP/MIRA calibration failed: {e}")

    if calibration_results:
        results["calibration"] = calibration_results
        (results_dir / "results.json").write_text(json.dumps(results, indent=2))

    wandb.finish()

    if vol is not None:
        vol.commit()
    print(f"Results saved to {results_dir}")
    print(f"Budget {budget}: DONE (FoM = {fom_mean:.2f} ± {fom_std:.2f})")
    return results


# ── Modal entry point (only loaded when invoked via `modal run`) ──────────────
if __name__ != "__main__":
    import modal

    volume = modal.Volume.from_name("cosmoford-training", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "torch>=2.4",
            "torchvision>=0.19",
            "lightning>=2.4",
            "datasets",
            "numpy",
            "wandb",
            "omegaconf",
            "pyyaml",
            "jsonargparse[signatures,omegaconf]>=4.27.7",
            "peft",
            "nflows",
            "matplotlib",
            "scikit-learn",
            "tarp",
            "mira_score",
            "getdist",
        )
        .add_local_dir("cosmoford", "/root/cosmoford", copy=True)
        .add_local_dir("configs", "/root/configs", copy=True)
        .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
        .run_commands("cd /root && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -e . --no-deps")
    )

    app = modal.App("cosmoford-npe-budget-scan", image=image)

    VOLUME_PATH = Path("/experiments")
    CHECKPOINTS_PATH = VOLUME_PATH / "checkpoints"
    NPE_RESULTS_PATH = VOLUME_PATH / "npe_results"
    SUMMARIES_CACHE_PATH = VOLUME_PATH / "summaries_cache"

    MODAL_CONFIG_PATH = "/root/configs/experiments/npe_budget_scan.yaml"

    @app.function(
        volumes={VOLUME_PATH: volume},
        gpu="a10g",
        timeout=86400,
        retries=modal.Retries(initial_delay=0.0, max_retries=0),
        single_use_containers=True,
        secrets=[modal.Secret.from_name("wandb-secret")],
    )
    def train_npe_for_budget(budget: int):
        from datasets import load_dataset as _hf_load
        cfg = NPEConfig.from_yaml(MODAL_CONFIG_PATH)
        return _train_budget_core(
            budget,
            CHECKPOINTS_PATH,
            NPE_RESULTS_PATH,
            SUMMARIES_CACHE_PATH,
            lambda split: _hf_load("CosmoStat/neurips-wl-challenge-holdout", split=split),
            cfg,
            vol=volume,
            load_flat_val=lambda split: _hf_load("CosmoStat/neurips-wl-challenge-flat", split=split),
        )

    @app.function(
        volumes={VOLUME_PATH: volume},
        timeout=60,
    )
    def load_all_results() -> list[dict]:
        import json
        volume.reload()
        results = []
        if NPE_RESULTS_PATH.exists():
            for d in sorted(NPE_RESULTS_PATH.iterdir()):
                rfile = d / "results.json"
                if rfile.exists():
                    results.append(json.loads(rfile.read_text()))
        return results

    @app.local_entrypoint()
    def main():
        cfg = NPEConfig.from_yaml(MODAL_CONFIG_PATH)
        handles = []
        for n in cfg.budgets:
            print(f"Spawning NPE pipeline for budget-{n}")
            handles.append(train_npe_for_budget.spawn(n))

        print(f"Waiting for {len(handles)} NPE runs to complete...")
        for h in handles:
            result = h.get()
            print(f"  budget-{result['budget']}: FoM = {result['fom_mean']:.2f} ± {result['fom_std']:.2f}")
        print("All NPE budget scan runs completed.")



# ── Local entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    from datasets import load_from_disk

    parser = argparse.ArgumentParser(
        description="NPE budget scan — local cluster mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoints_path", required=True,
                        help="Path to compressor checkpoints root (contains budget-N/ subdirs)")
    parser.add_argument("--npe_results_path", required=True,
                        help="Path where NPE results will be written")
    parser.add_argument("--summaries_cache_path", required=True,
                        help="Path for caching pre-computed summaries")
    parser.add_argument("--holdout_path", required=True,
                        help="Path to holdout DatasetDict (save_to_disk format, 'train' and 'fiducial' splits)")
    parser.add_argument("--flat_dataset_path", default=None,
                        help="Path to neurips-wl-challenge-flat DatasetDict (save_to_disk format). "
                             "If provided, TARP and MIRA calibration tests are run on the 'validation' split.")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (configs/experiments/npe_budget_scan.yaml)")
    parser.add_argument("--budgets",
                        help="Comma-separated list of budgets to run, e.g. 100,500,20200 "
                             "(overrides the budgets list in the config file)")
    parser.add_argument("--offline", help="Disable W&B checkpoint fallback; raise an error if a checkpoint "
                             "is not found locally", type = bool)
    parser.add_argument("--context_normalization", help="Enable context normalization", default=False,
                        type = bool)
    args = parser.parse_args()

    cfg = NPEConfig.from_yaml(args.config)
    if args.budgets is not None:
        cfg.budgets = [int(b) for b in args.budgets.split(",")]

    holdout_ds = load_from_disk(args.holdout_path)
    flat_ds = load_from_disk(args.flat_dataset_path) if args.flat_dataset_path else None

    for budget in cfg.budgets:
        _train_budget_core(
            budget,
            Path(args.checkpoints_path),
            Path(args.npe_results_path),
            Path(args.summaries_cache_path),
            lambda split, ds=holdout_ds: ds[split],
            cfg,
            offline=args.offline,
            load_flat_val=(lambda split, ds=flat_ds: ds[split]) if flat_ds is not None else None,
            context_normalization=args.context_normalization
        )