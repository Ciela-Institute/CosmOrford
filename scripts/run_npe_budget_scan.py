"""NPE training + FoM vs budget scan on Modal.

For each budget checkpoint from the compressor budget scan:
1. Load frozen compressor
2. Pre-compute summaries with noise augmentation on holdout data
3. Train NPE (conditional MAF) on (summary, theta) pairs
4. Evaluate FoM at fiducial cosmology

Usage:
    .venv/bin/modal run scripts/run_npe_budget_scan.py
"""
from pathlib import Path

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

BUDGETS = [100, 200, 500, 1000, 2000, 5000, 10000, 20200]

# NPE training hyperparameters
N_NOISE_REALIZATIONS = 16
NPE_EPOCHS = 500
NPE_LR = 1e-3
NPE_BATCH_SIZE = 512
NPE_PATIENCE = 50
NPE_SEEDS = 5

# FoM evaluation parameters
N_POSTERIOR_SAMPLES = 10_000

# Fiducial cosmology (unnormalized)
FIDUCIAL_OMEGA_M = 0.29
FIDUCIAL_S8 = 0.81


def find_best_checkpoint(budget: int) -> str:
    """Find the best compressor checkpoint for a given budget.

    Strategy:
    1. Parse val_mse from Lightning checkpoint filenames, pick lowest
    2. Fall back to last.ckpt
    3. Fall back to W&B artifact download
    """
    import re

    checkpoint_dir = CHECKPOINTS_PATH / f"budget-{budget}"

    # Strategy 1: Parse val_mse from checkpoint filenames
    if checkpoint_dir.exists():
        best_path = None
        best_mse = float("inf")
        for ckpt in checkpoint_dir.glob("*.ckpt"):
            if ckpt.name == "last.ckpt":
                continue
            match = re.search(r"val_mse=([\d.]+)", ckpt.name)
            if match:
                mse = float(match.group(1))
                if mse < best_mse:
                    best_mse = mse
                    best_path = str(ckpt)

        if best_path is not None:
            print(f"Found best checkpoint for budget-{budget}: {best_path} (val_mse={best_mse:.6f})")
            return best_path

        # Strategy 2: Fall back to last.ckpt
        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            print(f"Using last.ckpt for budget-{budget}")
            return str(last_ckpt)

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
                # Find the .ckpt file in the downloaded artifact
                for f in Path(art_dir).glob("**/*.ckpt"):
                    print(f"Downloaded W&B artifact for budget-{budget}: {f}")
                    return str(f)

    raise FileNotFoundError(f"No checkpoint found for budget-{budget}")


@app.function(
    volumes={VOLUME_PATH: volume},
    gpu="a10g",
    timeout=86400,
    retries=modal.Retries(initial_delay=0.0, max_retries=0),
    single_use_containers=True,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_npe_for_budget(budget: int):
    """End-to-end: load compressor, compute summaries, train NPE, evaluate FoM."""
    import json

    import numpy as np
    import torch
    from datasets import load_dataset

    from cosmoford import NOISE_STD, THETA_MEAN, THETA_STD
    from cosmoford.dataset import reshape_field_numpy
    from cosmoford.models_nopatch import RegressionModelNoPatch, build_flow

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Budget {budget}: starting NPE pipeline")
    print(f"{'='*60}")

    volume.reload()

    # ── 1-3. Load or compute summaries ──
    cache_dir = SUMMARIES_CACHE_PATH / f"budget-{budget}"
    cache_file = cache_dir / f"summaries_n{N_NOISE_REALIZATIONS}.pt"
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
        ckpt_path = find_best_checkpoint(budget)
        compressor = RegressionModelNoPatch.load_from_checkpoint(ckpt_path, map_location=device)
        compressor.eval()
        compressor.to(device)
        for p in compressor.parameters():
            p.requires_grad = False
        print(f"Loaded compressor from {ckpt_path}")

        # ── 2. Load holdout dataset ──
        print("Loading holdout dataset...")
        holdout = load_dataset("CosmoStat/neurips-wl-challenge-holdout", split="train")
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
        print(f"Pre-computing summaries ({N_NOISE_REALIZATIONS} noise realizations per map)...")
        all_summaries = []
        all_thetas = []

        with torch.no_grad():
            for i in range(n_maps):
                kappa_i = kappa_all[i]  # (1424, 176)
                kappa_reshaped = reshape_field_numpy(kappa_i[np.newaxis])[0]  # (1834, 88)

                for _ in range(N_NOISE_REALIZATIONS):
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

        # Cache to volume
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "summaries": summaries_tensor,
            "thetas": thetas_tensor,
            "theta_all_raw": theta_all,
            "compressor_checkpoint": ckpt_path,
            "n_noise_realizations": N_NOISE_REALIZATIONS,
        }, cache_file)
        volume.commit()
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=NPE_BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=NPE_BATCH_SIZE)

    # ── 5. Train NPE (multiple seeds, keep best) ──
    overall_best_nll = float("inf")
    overall_best_state = None

    for seed in range(NPE_SEEDS):
        print(f"\n--- NPE seed {seed+1}/{NPE_SEEDS} ---")
        torch.manual_seed(seed + 42)
        flow = build_flow(param_dim=2, context_dim=8).to(device)
        optimizer = torch.optim.Adam(flow.parameters(), lr=NPE_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NPE_EPOCHS)

        best_val_nll = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(NPE_EPOCHS):
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
                      f"best={best_val_nll:.4f}, patience={patience_counter}/{NPE_PATIENCE}")

            if patience_counter >= NPE_PATIENCE:
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
    print(f"\nOverall best val NLL across {NPE_SEEDS} seeds: {best_val_nll:.4f}")

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
    holdout = load_dataset("CosmoStat/neurips-wl-challenge-holdout", split="fiducial")
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
            noisy = (kappa_reshaped + noise) * mask
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            s = compressor.compress(x)  # (1, 8)

            # Sample posterior
            samples = flow.sample(N_POSTERIOR_SAMPLES, context=s)  # (1, N, 2)
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
    results_dir = NPE_RESULTS_PATH / f"budget-{budget}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save NPE weights
    torch.save(best_state, results_dir / "npe_flow.pt")

    # Save FoM results
    results = {
        "budget": budget,
        "fom_mean": float(fom_mean),
        "fom_std": float(fom_std),
        "fom_values": [float(v) for v in fom_values],
        "best_val_nll": float(best_val_nll),
        "n_noise_realizations": N_NOISE_REALIZATIONS,
        "n_fiducial_maps": len(kappa_all_fom),
        "n_posterior_samples": N_POSTERIOR_SAMPLES,
        "compressor_checkpoint": ckpt_path,
    }
    (results_dir / "results.json").write_text(json.dumps(results, indent=2))

    volume.commit()
    print(f"Results saved to {results_dir}")
    print(f"Budget {budget}: DONE (FoM = {fom_mean:.2f} ± {fom_std:.2f})")
    return results


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60,
)
def load_all_results() -> list[dict]:
    """Load all saved NPE results from the volume (used by plotting script)."""
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
    handles = []
    for n in BUDGETS:
        print(f"Spawning NPE pipeline for budget-{n}")
        handles.append(train_npe_for_budget.spawn(n))

    print(f"Waiting for {len(handles)} NPE runs to complete...")
    for h in handles:
        result = h.get()
        print(f"  budget-{result['budget']}: FoM = {result['fom_mean']:.2f} ± {result['fom_std']:.2f}")
    print("All NPE budget scan runs completed.")
