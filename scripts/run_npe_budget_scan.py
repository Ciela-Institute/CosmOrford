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
    n_noise_realizations: int = 16
    npe_epochs: int = 500
    npe_lr: float = 1e-3
    npe_batch_size: int = 512
    npe_patience: int = 50
    npe_seeds: int = 5
    n_posterior_samples: int = 10_000

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
            # match = re.search(r"val_log_prob=([\d.]+)", ckpt)
            match = re.search(r"val_log_prob=([-+]?\d*\.?\d+)", ckpt)
            if match:
                logp = float(match.group(1))
                if logp < best_logp:
                    best_logp = logp
                    best_path = str(ckpt)

        if best_path is not None:
            print(f"Found best checkpoint for budget-{budget}: {best_path} (val_log_prob={best_logp:.6f})")
            return best_path

        # Strategy 2: Fall back to last.ckpt
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
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


def _train_budget_core(budget: int, checkpoints_path, npe_results_path, summaries_cache_path, load_holdout, cfg: NPEConfig, offline: bool = False, vol=None):
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
    """
    import json

    import numpy as np
    import torch

    from cosmoford import NOISE_STD, THETA_MEAN, THETA_STD
    from cosmoford.dataset import reshape_field_numpy
    from cosmoford.models_nopatch import RegressionModelNoPatch, build_flow

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Budget {budget}: starting NPE pipeline")
    print(f"{'='*60}")
    print(f"Config: {cfg}")

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
                    noise = np.random.randn(*kappa_reshaped.shape).astype(np.float32) * NOISE_STD
                    noisy = (kappa_reshaped + noise) * mask
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

    train_dataset = torch.utils.data.TensorDataset(s_train, t_train)
    val_dataset = torch.utils.data.TensorDataset(s_val, t_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.npe_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.npe_batch_size)

    # ── 5. Train NPE (multiple seeds, keep best) ──
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
                loss = -flow.log_prob(t_batch, context=s_batch).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()

            # Validate
            flow.eval()
            val_losses = []
            with torch.no_grad():
                for s_batch, t_batch in val_loader:
                    s_batch, t_batch = s_batch.to(device), t_batch.to(device)
                    val_loss = -flow.log_prob(t_batch, context=s_batch).mean()
                    val_losses.append(val_loss.item())

            mean_train = np.mean(train_losses)
            mean_val = np.mean(val_losses)

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
    print(f"  Using {len(kappa_all_fom)} fiducial maps")

    from cosmoford import SURVEY_MASK
    mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    fom_values = []
    with torch.no_grad():
        for kappa_i in kappa_all_fom:
            kappa_reshaped = reshape_field_numpy(kappa_i[np.newaxis])[0]

            # Single noisy observation
            noise = np.random.randn(*kappa_reshaped.shape).astype(np.float32) * NOISE_STD
            # noisy = (kappa_reshaped + noise) * mask
            noisy = kappa_reshaped * mask # the holdout dataset is already noisy. 
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            s = compressor.compress(x)  # (1, 8)

            # Sample posterior
            samples = flow.sample(cfg.n_posterior_samples, context=s)  # (1, N, 2)
            samples = samples.squeeze(0).cpu().numpy()  # (N, 2)

            # Unnormalize to physical parameters
            samples_phys = samples * THETA_STD[:2] + THETA_MEAN[:2]

            # Compute FoM = 1 / sqrt(det(Cov))
            cov = np.cov(samples_phys.T)  # (2, 2)
            det = np.linalg.det(cov)
            if det > 0:
                fom = 1.0 / np.sqrt(det)
            else:
                fom = 0.0
            fom_values.append(fom)

    fom_mean = np.mean(fom_values)
    fom_std = np.std(fom_values)
    print(f"  FoM = {fom_mean:.2f} ± {fom_std:.2f}")

    # ── 7. Save results ──
    results_dir = npe_results_path / f"budget-{budget}"
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, results_dir / "npe_flow.pt")

    results = {
        "budget": budget,
        "fom_mean": float(fom_mean),
        "fom_std": float(fom_std),
        "fom_values": [float(v) for v in fom_values],
        "best_val_nll": float(best_val_nll),
        "n_noise_realizations": cfg.n_noise_realizations,
        "n_fiducial_maps": len(kappa_all_fom),
        "n_posterior_samples": cfg.n_posterior_samples,
        "compressor_checkpoint": ckpt_path,
    }
    (results_dir / "results.json").write_text(json.dumps(results, indent=2))

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
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file (configs/experiments/npe_budget_scan.yaml)")
    parser.add_argument("--budgets",
                        help="Comma-separated list of budgets to run, e.g. 100,500,20200 "
                             "(overrides the budgets list in the config file)")
    parser.add_argument("--offline", help="Disable W&B checkpoint fallback; raise an error if a checkpoint "
                             "is not found locally", type = bool)
    args = parser.parse_args()

    cfg = NPEConfig.from_yaml(args.config)
    if args.budgets is not None:
        cfg.budgets = [int(b) for b in args.budgets.split(",")]

    holdout_ds = load_from_disk(args.holdout_path)

    for budget in cfg.budgets:
        _train_budget_core(
            budget,
            Path(args.checkpoints_path),
            Path(args.npe_results_path),
            Path(args.summaries_cache_path),
            lambda split, ds=holdout_ds: ds[split],
            cfg,
            offline=args.offline,
        )