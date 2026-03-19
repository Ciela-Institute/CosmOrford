import argparse
from pathlib import Path
import os
from datasets import load_from_disk, load_dataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
import ot
import wandb
from cosmoford.emulator.torch_models import build_unet2d_condition_with_y
from cosmoford.dataset import reshape_field_numpy
from cosmoford.emulator.utils import (
    preprocess_batch,
    split_rng,
    augmentation_data_numpy,
    apply_mask,
    iter_microbatches,
    pqm_evaluate,
)
from cosmoford.emulator.neural_ode import solve_ode_forward

plt.style.use("seaborn-v0_8")
# Improve default figure quality for logged images (crisper text and colorbars)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
    # Larger fonts for readability after downscaling in W&B UI
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

parser = argparse.ArgumentParser()
parser.add_argument("--exp_config", type=str, required=True, help="Path to experiment YAML")
parser.add_argument("--sim_budget", type=int, default=None, help="Number of N-body simulations to train on (null = full dataset)")
cli = parser.parse_args()

with open(cli.exp_config) as f:
    _cfg = yaml.safe_load(f)
_cfg.pop("exp_name", None)

_defaults = {"seed": 42}
_defaults.update(_cfg)
args = argparse.Namespace(**_defaults)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

print("--Set Config--")

print("--Load Dataset--")

if args.dataset_dir_nbody is not None:
    dataset_nbody = load_from_disk(args.dataset_dir_nbody)
else:
    dataset_nbody = load_dataset("cosmostat/neurips-wl-challenge-flat")
dset_nbody = dataset_nbody.with_format('numpy')
test_dataset_nbody = dset_nbody['validation']

# Subset N-body training set for simulation budget scan.
# Use first N samples (range) to match the compressor budget scan convention.
_train_nbody_full = dset_nbody['train']
sim_budget = cli.sim_budget
if sim_budget is not None:
    _n_full = len(_train_nbody_full)
    train_dataset_nbody = _train_nbody_full.select(range(sim_budget), keep_in_memory=True).with_format('numpy')
    print(f"N-body budget: {sim_budget} / {_n_full} sims")
else:
    train_dataset_nbody = _train_nbody_full

dataset_lognormal = load_from_disk(args.dataset_dir_logn_train)
n = len(dataset_lognormal)
perm = np.random.default_rng(2).permutation(n).tolist()
n_test = int(0.2 * n)
train_dataset_lognormal = dataset_lognormal.select(perm[n_test:], keep_in_memory=True).with_format('numpy')
test_dataset_lognormal = dataset_lognormal.select(perm[:n_test], keep_in_memory=True).with_format('numpy')

print("train_dataset_lognormal", train_dataset_lognormal)
print("test_dataset_lognormal", test_dataset_lognormal)
print("train_dataset_pm", train_dataset_nbody)
print("test_dataset_pm", test_dataset_nbody)

# Use NumPy reshape helper to infer reduced field sizes without torch conversions
_kappa_sample = test_dataset_nbody[:10]['kappa']  # (10, H, W) numpy
_kappa_rs = reshape_field_numpy(_kappa_sample)    # (10, H_red, W_red)
field_npix_x = _kappa_rs.shape[-1]
field_npix_y = _kappa_rs.shape[-2]
# Align y dimension at init with training-time y (lognormal theta[:,1:])
nb_of_params_to_infer = 3

# just because I cannot access Noe's dataset directly
def get_iterable_dataset(dataset, batch_size, seed):
    # Ensure shuffling writes indices to a writable directory under the W&B run folder
    split_cache_dir = os.path.join(str(run_dir), "hf_indices")
    os.makedirs(split_cache_dir, exist_ok=True)
    idx_cache = os.path.join(split_cache_dir, f"shuffle_{seed}_{len(dataset)}.idx.arrow")
    return dataset.shuffle(seed=seed, indices_cache_file_name=idx_cache).iter(
        batch_size=batch_size, drop_last_batch=True
    )

print("--Set OT--")

def compute_cost_matrix(
    x0: torch.Tensor, x1: torch.Tensor, yx0: torch.Tensor, yx1: torch.Tensor, eps: float
) -> torch.Tensor:
    C_x = ot.dist(x0, x1)
    C_y = ot.dist(yx0, yx1)
    M = eps * C_x + C_y
    return M / M.max()

def get_paired_data(tp: np.ndarray, n_x0: int, rng: np.random.Generator):
    p = tp.flatten()
    p = p / p.sum()
    choices = rng.choice(tp.size, size=n_x0, replace=False, p=p)
    x0_paired, x1_paired = np.divmod(choices, tp.shape[1])
    return x0_paired, x1_paired

def sample_ot_plan(
    x0: np.ndarray,
    x1: np.ndarray,
    yx0: np.ndarray,
    yx1: np.ndarray,
    rng: np.random.Generator,
    eps: float,
    device: torch.device,
    reg: float,
    ot_method: str = "sinkhorn",
):
    x0_t = torch.from_numpy(x0.reshape(x0.shape[0], -1)).float().to(device)
    x1_t = torch.from_numpy(x1.reshape(x1.shape[0], -1)).float().to(device)
    yx0_t = torch.from_numpy(np.array(yx0)).float().to(device)
    yx1_t = torch.from_numpy(np.array(yx1)).float().to(device)
    n, m = x0_t.shape[0], x1_t.shape[0]
    a = torch.full((n,), 1.0 / n, device=device, dtype=torch.float32)
    b = torch.full((m,), 1.0 / m, device=device, dtype=torch.float32)
    M = compute_cost_matrix(x0_t, x1_t, yx0_t, yx1_t, eps)
    if ot_method == "emd":
        # ot.emd requires CPU tensors (LP solver runs on CPU)
        tp = ot.emd(a.cpu(), b.cpu(), M.cpu()).to(device)
    else:
        # sinkhorn_log: log-domain arithmetic, numerically stable for small reg
        # numItermax=50000 ensures convergence at ot_reg=1e-3
        tp = ot.sinkhorn(a, b, M, reg=reg, method="sinkhorn_log", numItermax=50000)
    return get_paired_data(tp.cpu().numpy(), n, rng)


print("--Define get sample--")

def sample_time(n_samples: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n_samples, 1, device=device)

def get_ot_batch(batch, rng: np.random.Generator, eps: float, device: torch.device, ot_reg: float, ot_method: str = "sinkhorn"):
    # Split parent RNG for independent, reproducible streams
    rng_pre, rng_x0, rng_x1, rng_plan = split_rng(rng, 4)

    batch = preprocess_batch(batch, rng_pre)
    batch_logn, batch_nbody = batch
    x0 = batch_logn['maps']
    x1 = batch_nbody['maps']
    theta_x0 = batch_logn['theta']
    theta_x1 = batch_nbody['theta']
    batch_size = len(batch_logn['theta'])

    t = sample_time(batch_size, device)
    # Augment before OT pairing

    x0, vmask0, hmask0 = augmentation_data_numpy(x0, rng_x0)
    x1, vmask1, hmask1 = augmentation_data_numpy(x1, rng_x1)

    inds_x0, inds_x1 = sample_ot_plan(x0, x1, theta_x0, theta_x1, rng_plan, eps, device, ot_reg, ot_method=ot_method)
    x0_paired = x0[inds_x0]
    x1_paired = x1[inds_x1]
    theta_x0_paired = theta_x0[inds_x0]
    theta_x1_paired = theta_x1[inds_x1]
    vmask1 = vmask1[inds_x1]
    hmask1 = hmask1[inds_x1]
    x0_paired = apply_mask(x0_paired, vmask1, hmask1)

    # Convert to torch tensors with desired shapes
    # Ensure channel-last shape then to torch (B, 1, H, W)
    if x0_paired.ndim == 3:
        x0_paired = x0_paired[..., None]
    if x1_paired.ndim == 3:
        x1_paired = x1_paired[..., None]
    x0_t = torch.from_numpy(x0_paired.transpose(0, 3, 1, 2)).float().to(device)
    x1_t = torch.from_numpy(x1_paired.transpose(0, 3, 1, 2)).float().to(device)
    theta_x0_t = torch.from_numpy(theta_x0_paired).float().to(device)
    theta_x1_t = torch.from_numpy(theta_x1_paired).float().to(device)
    return {
        'x0': x0_t,
        'x1': x1_t,
        't': t,
        'theta_x0': theta_x0_t,
        'theta_x1': theta_x1_t,
    }

print("--Flow matching training fun--")

def compute_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return x1 - x0

def sample_conditional_pt(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma: float
) -> torch.Tensor:
    # Broadcast t: (B,1,1,1)
    t_b = t.view(-1, 1, 1, 1)
    x = t_b * x1 + (1.0 - t_b) * x0
    return x + sigma * torch.randn_like(x)

def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # Sum over spatial dims, mean over batch
    return ((y_pred - y_true) ** 2).sum(dim=(2, 3)).mean()

def flow_matching_loss(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    yx0: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    v = compute_velocity(x0=x0, x1=x1)
    x_t = sample_conditional_pt(x0=x0, x1=x1, t=t, sigma=sigma)
    # model expects (B, C, H, W)
    t_in = t.view(-1)  # (B,)
    y_in = yx0  # (B, y_dim)
    # Provide encoder_hidden_states only if the UNet expects cross-attention
    enc = None
    cross_dim = getattr(model.config, "cross_attention_dim", None)
    if cross_dim is not None:
        enc = torch.zeros(x_t.size(0), 1, cross_dim, device=x_t.device, dtype=x_t.dtype)
    out = model(x_t, t_in, encoder_hidden_states=enc, y=y_in)
    v_pred = out.sample if hasattr(out, 'sample') else out
    return mse_loss(v_pred, v)

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x0: torch.Tensor,
    x1: torch.Tensor,
    yx0: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = flow_matching_loss(model, x0, x1, yx0, t, sigma)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


print("--Define the unet model--")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg_path = Path(args.config_yaml)
with open(cfg_path, 'r') as f:
    config = yaml.safe_load(f)
config_source = str(cfg_path)

# Ensure sample_size matches the reshaped sizes (H, W)
config["sample_size"] = [field_npix_x, field_npix_y]
unet = build_unet2d_condition_with_y(config).to(device)

print('--Set optimizer--')
optimizer = Adam(unet.parameters(), lr=args.base_lr)
scheduler = ExponentialLR(optimizer, gamma=args.gamma)

print('--Init WandB--')

# Initialize Weights & Biases (always enabled)
run_suffix = f"{np.random.randint(100, 1000)}"
auto_run_name = f"emulator_training_{run_suffix}"
_base_run_name = args.wandb_run_name or auto_run_name
_run_name = f"{_base_run_name}/budget_{sim_budget}" if sim_budget is not None else _base_run_name
wandb_kwargs = dict(project=args.wandb_project, name=_run_name, config={
    "eps": args.eps,
    "max_steps": args.max_steps,
    "batch_size": args.batch_size,
    "sigma": args.sigma,
    "micro_batch_size": args.micro_batch_size,
    "base_lr": args.base_lr,
    "gamma": args.gamma,
    "ot_reg": args.ot_reg,
    "ot_method": args.ot_method,
    "sim_budget": sim_budget,
    "model_config": config,
    "config_source": config_source,
})
if args.wandb_entity:
    wandb_kwargs["entity"] = args.wandb_entity
# mode can be "online" or "offline"
wandb_kwargs["mode"] = args.wandb_mode
run = wandb.init(**wandb_kwargs)
# Parameter counts
try:
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    wandb.summary["model/total_params"] = total_params
    wandb.summary["model/trainable_params"] = trainable_params
except Exception:
    pass
# Save code snapshot for reproducibility
try:
    # Log only the current module directory (emulator), not the whole cosmoford package
    wandb.run.log_code(root=str(Path(__file__).resolve().parent))
except Exception:
    pass

# Keep all run outputs inside the W&B run directory
run_dir = Path(wandb.run.dir).resolve()
fig_dir = run_dir / 'fig'
ckpt_dir = run_dir / 'checkpoints'

# Ensure the chosen directories exist
fig_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Persist resolved config to the W&B run directory and log it as an artifact
resolved_cfg_path = run_dir / 'config_used.yaml'
try:
    with open(resolved_cfg_path, 'w') as f:
        yaml.safe_dump(config, f)
    # Update wandb config with the saved file path for reference
    try:
        wandb.config.update({"config_file": str(resolved_cfg_path)}, allow_val_change=True)
    except Exception:
        pass
    # Log config artifact
    cfg_art = wandb.Artifact('unet-config', type='config')
    cfg_art.add_file(str(resolved_cfg_path))
    wandb.log_artifact(cfg_art)
except Exception:
    pass

print('--PQMass evaluation--')

def _run_pqm(model, label, seed_data, seed_rng, fig_path, chi2_key, plot_key, extra_log=None):
    try:
        pqm_bs = 500
        ds_logn = get_iterable_dataset(test_dataset_lognormal, pqm_bs, seed_data)
        ds_nbody = get_iterable_dataset(test_dataset_nbody, pqm_bs, seed_data)
        batch_logn = next(ds_logn)
        batch_nbody = next(ds_nbody)
        batch = get_ot_batch([batch_logn, batch_nbody], np.random.default_rng(seed_rng), eps, device, args.ot_reg, ot_method=args.ot_method)

        chunks = []
        for start in range(0, batch['x0'].shape[0], micro_bs):
            x0_c = batch['x0'][start:start + micro_bs]
            theta_c = batch['theta_x0'][start:start + micro_bs]
            with torch.no_grad():
                pred_c = solve_ode_forward(x0_c, model, theta_c, device)
            chunks.append(pred_c[-1])
        maps_gen = np.concatenate(chunks, axis=0)
        maps_ref = batch['x1'].detach().cpu().squeeze(1).numpy()

        chi2_vals, fig = pqm_evaluate(maps_ref, maps_gen)
        fig.suptitle(f"PQMass: N-body vs UNet ({label})", fontsize=13)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log = {
            chi2_key: float(np.mean(chi2_vals)),
            plot_key: wandb.Image(str(fig_path), caption=f"PQMass N-body vs UNet ({label})"),
        }
        if extra_log:
            log.update(extra_log)
        wandb.log(log)
        print(f"PQMass [{label}]: mean χ² = {np.mean(chi2_vals):.1f}")
        return float(np.mean(chi2_vals))
    except Exception as e:
        print(f"PQMass [{label}] failed: {e}")
        return None



print('--Training--')

eps = args.eps
max_steps = args.max_steps
batch_size = min(args.batch_size, len(train_dataset_nbody))
sigma = args.sigma
micro_bs = int(args.micro_batch_size)

num_training_steps_total = max_steps
nb_checkpoints = num_training_steps_total // 5
pqm_step_interval = max(1, num_training_steps_total // max(1, args.n_pqm_evals))
step = 0
best_val_loss = float('inf')
best_pqm_chi2 = float('inf')

def _save_best_ckpt():
    best_ckpt = str(ckpt_dir / "unet_best.pth")
    torch.save(unet.state_dict(), best_ckpt)
    try:
        art = wandb.Artifact("unet-best", type="model")
        art.add_file(best_ckpt)
        wandb.log_artifact(art)
    except Exception:
        pass

epoch = 0
pbar = tqdm(total=max_steps, desc="training")
while step < max_steps:
    ds_train_logn = get_iterable_dataset(train_dataset_lognormal, batch_size, int((epoch + 1) * 1000))
    ds_train_nbody = get_iterable_dataset(train_dataset_nbody, batch_size, epoch)
    rng_epoch = np.random.default_rng(args.seed + epoch)

    for batch_logn, batch_nbody in zip(ds_train_logn, ds_train_nbody):
        batch = get_ot_batch([batch_logn, batch_nbody], rng_epoch, eps, device, args.ot_reg, ot_method=args.ot_method)

        for mb in iter_microbatches(batch, micro_bs):
            loss = train_step(
                unet,
                optimizer,
                mb['x0'],
                mb['x1'],
                mb['theta_x0'],
                mb['t'],
                sigma,
            )
            step += 1
            pbar.update(1)
            if step >= max_steps:
                break
            wandb.log({
                "train_loss": loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
            })
            if step % 500 == 0:
                # basic scheduler step per epoch
                scheduler.step()

                rng_eval = np.random.default_rng(args.seed + 1234)
                ds_test_lognormal = get_iterable_dataset(test_dataset_lognormal, micro_bs, 1)
                ds_test_nbody = get_iterable_dataset(test_dataset_nbody, micro_bs, 0)
                batch_lognormal = next(ds_test_lognormal)
                batch_nbody = next(ds_test_nbody)
                batch = get_ot_batch([batch_lognormal, batch_nbody], rng_eval, eps, device, args.ot_reg, ot_method=args.ot_method)

                with torch.no_grad():
                    lt = flow_matching_loss(unet, batch['x0'], batch['x1'], batch['theta_x0'], batch['t'], sigma)
                    loss_t = float(lt.detach().cpu().item())

                wandb.log({
                    "val_loss": loss_t,
                    "train_loss_epoch": float(loss),
                    "epoch": epoch
                })

                if args.best_ckpt_metric == "val_loss" and loss_t < best_val_loss:
                    best_val_loss = loss_t
                    _save_best_ckpt()

            if step % nb_checkpoints == 0:
                ds_test_lognormal = get_iterable_dataset(test_dataset_lognormal, micro_bs, 0)
                ds_test_nbody = get_iterable_dataset(test_dataset_nbody, micro_bs, 0)
                batch_lognormal = next(ds_test_lognormal)
                batch_nbody = next(ds_test_nbody)
                batch_test = get_ot_batch([batch_lognormal, batch_nbody], rng_epoch, eps, device, args.ot_reg, ot_method=args.ot_method)

                x_1_pred = solve_ode_forward(batch_test['x0'], unet, batch_test['theta_x0'], device)

                # Higher DPI for crisper images in W&B; moderate figsize to reduce browser downscale blur
                fig = plt.figure(figsize=(12, 6), dpi=300, constrained_layout=True)
                # Prepare numpy arrays for plotting (avoid cuda tensors in np.concatenate)
                x0_np = batch_test['x0'][0].detach().cpu().squeeze(0).numpy()  # (H,W)
                x1_np = batch_test['x1'][0].detach().cpu().squeeze(0).numpy()  # (H,W)
                pred_np = x_1_pred[-1, 0]  # (H,W) numpy
                # Use consistent color scaling across panels to aid visual comparison
                all_vals = np.concatenate([
                    x0_np.ravel(),
                    pred_np.ravel(),
                    x1_np.ravel()
                ])
                vmin, vmax = np.percentile(all_vals, [1, 99])

                ax1 = fig.add_subplot(5, 1, 1)
                im1 = ax1.imshow(x0_np.T, cmap="viridis", vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im1, ax=ax1); cbar.ax.tick_params(labelsize=12)
                ax1.set_title("Beginning of ODE", fontsize=16)
                ax1.axis("off")

                ax2 = fig.add_subplot(5, 1, 2)
                im2 = ax2.imshow(pred_np.T, cmap="viridis", vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im2, ax=ax2); cbar.ax.tick_params(labelsize=12)
                ax2.set_title("End of ODE", fontsize=16)
                ax2.axis("off")

                ax3 = fig.add_subplot(5, 1, 3)
                im3 = ax3.imshow(x1_np.T, cmap="viridis", vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im3, ax=ax3); cbar.ax.tick_params(labelsize=12)
                ax3.set_title("Truth", fontsize=16)
                ax3.axis("off")

                ax4 = fig.add_subplot(5, 1, 4)
                im4 = ax4.imshow((pred_np - x1_np).T, cmap="viridis")
                cbar = fig.colorbar(im4, ax=ax4); cbar.ax.tick_params(labelsize=12)
                ax4.set_title("Residuals", fontsize=16)
                ax4.axis("off")

                ax5 = fig.add_subplot(5, 1, 5)
                im5 = ax5.imshow((x0_np - pred_np).T, cmap="viridis")
                cbar = fig.colorbar(im5, ax=ax5); cbar.ax.tick_params(labelsize=12)
                ax5.set_title("Learned correction", fontsize=16)
                ax5.axis("off")

                # Save to disk at high resolution to avoid any in-memory backend inconsistencies
                img_path = fig_dir / f"fm_diag_epoch_{epoch}.png"
                fig.savefig(img_path, dpi=300, bbox_inches="tight")
                log_dict = {"fm_diagnostics": wandb.Image(str(img_path), caption=f"fm_diagnostics epoch {epoch}"), "epoch": epoch}
                try:
                    wandb.log(log_dict)
                except Exception:
                    pass
                plt.close(fig)

            # PQMass evaluation at its own interval (controls best-checkpoint tracking)
            if step > 0 and step % pqm_step_interval == 0:
                chi2_epoch = _run_pqm(
                    unet, f"step {step}", 99, 0,
                    fig_dir / f"pqm_unet_step_{step}.png",
                    "pqm/chi2_unet_mean", "pqm/plot_unet",
                    extra_log={"epoch": epoch},
                )
                if args.best_ckpt_metric == "pqm_chi2" and chi2_epoch is not None and chi2_epoch < best_pqm_chi2:
                    best_pqm_chi2 = chi2_epoch
                    _save_best_ckpt()

        if step >= max_steps:
            break
    epoch += 1
pbar.close()

# Final trained model
try:
    final_ckpt = str(ckpt_dir / "unet_FINAL.pth")
    torch.save(unet.state_dict(), final_ckpt)
    art = wandb.Artifact("unet-final", type="model")
    art.add_file(final_ckpt)
    wandb.log_artifact(art)
except Exception:
    pass

# Post-training PQMass: evaluate both last and best checkpoints on a fixed test set
_run_pqm(unet, "last", 42, 42, fig_dir / "pqm_final_last.png", "pqm/chi2_last_mean", "pqm/plot_last")

best_ckpt_path = ckpt_dir / "unet_best.pth"
if best_ckpt_path.exists():
    unet_best = build_unet2d_condition_with_y(config).to(device)
    unet_best.load_state_dict(torch.load(str(best_ckpt_path), map_location=device))
    unet_best.eval()
    _run_pqm(unet_best, "best", 42, 42, fig_dir / "pqm_final_best.png", "pqm/chi2_best_mean", "pqm/plot_best")

wandb.finish()