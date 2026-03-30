import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist
from pqm import pqm_chi2
from cosmoford import SURVEY_MASK
from cosmoford.dataset import reshape_field_numpy


def iter_microbatches(batch: dict[str, torch.Tensor], micro_bs: int):
    """Yield smaller batch dicts by slicing along the batch dimension.
    If micro_bs <= 0 or micro_bs >= B, yield the original batch once.
    """
    B = batch['x0'].shape[0]
    if micro_bs <= 0 or micro_bs >= B:
        yield batch
        return
    for start in range(0, B, micro_bs):
        end = min(B, start + micro_bs)
        yield {
            'x0': batch['x0'][start:end],
            'x1': batch['x1'][start:end],
            't': batch['t'][start:end],
            'theta_x0': batch['theta_x0'][start:end],
            'theta_x1': batch['theta_x1'][start:end],
        }

def preprocess_batch(batch, rng: np.random.Generator):
    """Prepare NumPy batches with minimal conversions before UNet.

    - For lognormal: select one of the 10 maps per sim, reshape via NumPy helper
    - For nbody: reshape via NumPy helper
    - Return NumPy arrays; torch conversion happens right before UNet
    """
    batch_lognormal, batch_nbody = batch
    idx = rng.integers(low=0, high=10)

    print(batch_lognormal.keys())

    if isinstance(batch_lognormal, dict) and 'maps' in batch_lognormal:
        batch_lognormal = {'kappa': batch_lognormal['maps'], 'theta': batch_lognormal['theta']}

    shape_dataset = batch_lognormal['kappa'].shape
    # Lognormal maps: (B, 10, H, W) -> pick idx -> (B, H, W) -> reshape (B, 1834, 88)
    kappa_logn = batch_lognormal['kappa']
    y_logn = batch_lognormal['theta']
    if len(shape_dataset) == 4:
        kappa_logn = kappa_logn[:, idx, :, :]
        y_logn = y_logn[:, 1:]
    x_logn = reshape_field_numpy(kappa_logn)

    # N-body maps: (B, H, W) -> reshape (B, 1834, 88)
    kappa_nbody = batch_nbody['kappa']
    x_nbody = reshape_field_numpy(kappa_nbody)
    y_nbody = batch_nbody['theta'][:, [0, 1, -1]]

    return {'maps': x_logn, 'theta': y_logn}, {'maps': x_nbody, 'theta': y_nbody}

def split_rng(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Create n independent child RNGs from a parent Generator by sampling new seeds.
    Deterministic given the parent's state.
    """
    return [np.random.default_rng(int(rng.integers(0, 2**63))) for _ in range(n)]

def augmentation_data_numpy(
    maps: np.ndarray,
    rng: np.random.Generator | None = None,
    p_flip: float = 0.5,
    vmask: np.ndarray | None = None,
    hmask: np.ndarray | None = None,
):
    """Random vertical and horizontal flips per-sample using NumPy ops before OT pairing.
    Accepts (B, H, W) or (B, H, W, 1); returns (maps, vmask, hmask).
    If vmask/hmask are provided they are reused, otherwise sampled from rng.
    """
    if maps.ndim not in (3, 4):
        raise ValueError(f"Expected maps with ndim 3 or 4, got shape {maps.shape}")
    add_channel = maps.ndim == 3
    if add_channel:
        maps = maps[..., None]
    # print(rng.random(B))
    B = maps.shape[0]
    # Vertical flips on axis=1
    if vmask is None:
        vmask = rng.random(B) < p_flip
    if vmask.any():
        maps[vmask] = maps[vmask, ::-1, :, :]
    # Horizontal flips on axis=2
    if hmask is None:
        hmask = rng.random(B) < p_flip
    if hmask.any():
        maps[hmask] = maps[hmask, :, ::-1, :]

    if add_channel:
        maps = maps[..., 0]
    return maps, vmask, hmask

def apply_mask(x, vmask=None, hmask=None):
    """Apply survey mask to x, aligning flips per-sample using vmask/hmask.

    x is expected to be NumPy with shape (B,H,W) or (B,H,W,1).
    vmask/hmask are boolean arrays of length B indicating flips applied to the maps.
    """
    base_mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])  # (H,W)
    # Add batch dim and repeat to match x's batch size
    B = x.shape[0]
    mask = base_mask[None, ...]  # (1,H,W)
    if mask.shape[0] != B:
        mask = np.repeat(mask, B, axis=0)  # (B,H,W)

    # If flip metadata provided, apply the same flips to the mask per-sample
    if vmask is not None and hmask is not None:
        mask, _, _ = augmentation_data_numpy(mask, vmask=vmask, hmask=hmask)

    # Match channel if x has a trailing channel dim
    if x.ndim == 4 and x.shape[-1] == 1:
        mask = mask[..., None]

    return x * mask


def pqm_evaluate(maps_ref, maps_gen, num_refs=100, re_tessellation=20):
    """Compare two sets of maps using PQMass.

    Parameters
    ----------
    maps_ref : np.ndarray, shape (N, H, W)
        Reference maps (e.g., N-body validation set).
    maps_gen : np.ndarray, shape (M, H, W)
        Generated maps (e.g., UNet ODE output).
    num_refs : int
        Number of reference cells for the Voronoi tessellation.
    re_tessellation : int
        Number of independent tessellations; returns a distribution of chi².

    Returns
    -------
    chi2_vals : list of float
        Chi² values, one per tessellation.
    fig : matplotlib.figure.Figure
        Histogram of chi² values vs the expected chi²(dof=num_refs-1) PDF.
    """
    ref_flat = maps_ref.reshape(len(maps_ref), -1).astype(np.float32)
    gen_flat = maps_gen.reshape(len(maps_gen), -1).astype(np.float32)

    chi2_vals = pqm_chi2(ref_flat, gen_flat, num_refs=num_refs, re_tessellation=re_tessellation)

    dof = num_refs - 1
    x = np.linspace(max(0, dof - 4 * np.sqrt(2 * dof)), dof + 6 * np.sqrt(2 * dof), 300)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(chi2_vals, bins=15, density=True, alpha=0.7, color="steelblue", label="PQMass χ²")
    ax.plot(x, chi2_dist.pdf(x, df=dof), "r-", lw=2, label=f"χ²(dof={dof})")
    ax.axvline(np.mean(chi2_vals), color="steelblue", linestyle="--", lw=1.5,
               label=f"mean = {np.mean(chi2_vals):.1f}")
    ax.set_xlabel("χ² statistic")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)
    ax.set_title("PQMass — χ² distribution")

    return chi2_vals, fig