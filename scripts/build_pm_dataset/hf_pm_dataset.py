import argparse
import glob
import math
import os
import re
from pathlib import Path

import datasets
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from astropy.table import Table
from datasets import Dataset, Features, Array2D, Value, Sequence
from tqdm import tqdm

from compute_kappa_pm import compute_kappa

parser = argparse.ArgumentParser()
parser.add_argument('--sims_path', type=str, default='/home/juzgh/scratch/nbody_sims', help='Path to the simulations')
parser.add_argument('--params_path', type=str, default='notebooks/cosmo_parameters_uniform.npy',
                    help='Path to the cosmological parameters  file')
parser.add_argument('--nz_path', type=str, default='notebooks/nz.fits', help='Path to redshift distribution FITS file')
parser.add_argument('--output_dir', type=str, default='artifacts/kappa',
                    help='Directory to save processed kappa patches')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for processing')
parser.add_argument('--chunk_size', type=int, default=500, help='Number of samples per chunk')
args = parser.parse_args()

print("Loading data...")

data_dir = args.sims_path
files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))


# Build a mapping from seed -> file path, extracting the seed from filenames
def _extract_seed_from_filename(path: str):
    base = os.path.basename(path)
    # Prefer trailing digits before .npy; fallback to the last digit group anywhere
    m = re.search(r'(\d+)(?=\.npy$)', base)
    if m:
        return int(m.group(1))
    all_nums = re.findall(r'\d+', base)
    if not all_nums:
        return None
    return int(all_nums[-1])


seed_to_file = {}
for p in files:
    s = _extract_seed_from_filename(p)
    if s is not None and s not in seed_to_file:
        seed_to_file[s] = p

# Load parameters: column 0 = seed, columns 1-2 = cosmological params
params_all = np.load(args.params_path)
seeds_all = params_all[:, 0].astype(int)

# Keep only rows that have a corresponding file by seed
mask_has_file = np.array([seed in seed_to_file for seed in seeds_all])
params = params_all[mask_has_file]
seeds = params[:, 0].astype(int)
nz_hsc = Table.read(args.nz_path)
# Pre-extract JAX arrays for z and bins to avoid string indexing inside JAX transforms
z_mid = jnp.array(np.asarray(nz_hsc['Z_MID'], dtype=np.float32))
bins_arr = jnp.stack([
    jnp.array(np.asarray(nz_hsc[f'BIN{i}'], dtype=np.float32)) for i in range(1, 4)
], axis=0)
# Build the base KDE n(z) objects once — systematic_shift will apply delta_z per sim
base_nzs = [
    jc.redshift.kde_nz(z_mid, bins_arr[i], bw=0.015)
    for i in range(3)
]

batch_size = args.batch_size
num_samples = params.shape[0]
num_chunks = math.ceil(num_samples / args.chunk_size)
dataset = None
for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
    chunk_params = []
    chunk_maps = []

    # Determine this chunk's rows
    chunk_start = chunk_idx * args.chunk_size
    chunk_end = min((chunk_idx + 1) * args.chunk_size, num_samples)
    params_chunk = params[chunk_start:chunk_end]
    seeds_chunk = seeds[chunk_start:chunk_end]

    # Number of batches for this chunk
    num_in_chunk = params_chunk.shape[0]
    num_batches = math.ceil(num_in_chunk / batch_size)

    # Generate data for this chunk batch by batch
    for b in tqdm(range(num_batches), desc=f"Generating batches for chunk {chunk_idx + 1}/{num_chunks}"):
        b_start = b * batch_size
        b_end = min((b + 1) * batch_size, num_in_chunk)
        if b_start >= b_end:
            continue

        batch_params = params_chunk[b_start:b_end]  # (B, 3): [seed, Omega_c, S8]
        batch_seeds = seeds_chunk[b_start:b_end]

        # Load simulation data by seed
        data_list = []
        for seed in batch_seeds:
            fp = seed_to_file.get(int(seed))
            if fp is None:
                continue
            data_list.append(np.load(fp, allow_pickle=True))
        if not data_list:
            continue

        data = jnp.stack(data_list, axis=0)  # shape (B, 3, 1024, 1024)

        # Sample redshift uncertainty delta_z ~ Normal(0, 0.022) per simulation
        B_actual = len(data_list)
        delta_z_batch = np.random.normal(loc=0.0, scale=0.022, size=(B_actual, 1))
        cosmo_params_chunk = jnp.array(
            np.concatenate([batch_params[:B_actual], delta_z_batch], axis=1)
        )  # (B, 4): [seed, Omega_c, S8, delta_z]

        kappas = compute_kappa(data, cosmo_params_chunk, base_nzs, bins_arr)  # (B, Npatch, 1424, 176)

        maps_np = np.array(kappas)
        B, Npatch = maps_np.shape[0], maps_np.shape[1]
        maps_flat = maps_np.reshape(B * Npatch, 1424, 176)

        # Keep (Omega_c, S8, delta_z) and repeat per patch
        theta_batch = np.concatenate(
            [np.array(batch_params)[:B_actual, 1:3], delta_z_batch], axis=1
        )  # (B, 3): [Omega_c, S8, delta_z]
        theta_rep = np.repeat(theta_batch, Npatch, axis=0)

        chunk_params.append(theta_rep.astype(np.float16))
        chunk_maps.append(maps_flat.astype(np.float16))

    if chunk_params:  # Only process if we have data
        # Concatenate the batches for this chunk
        chunk_params = np.concatenate(chunk_params, axis=0).tolist()
        chunk_maps = np.concatenate(chunk_maps, axis=0)

        # Create chunk dataset
        chunk_dataset = Dataset.from_dict({
            'theta': chunk_params,
            'maps': chunk_maps,
        }, features=Features({
            'theta': Sequence(Value(dtype='float16'), length=3),
            'maps': Array2D(dtype='float16', shape=(1424, 176)),
        }))

        # For the first chunk, create the initial dataset
        if dataset is None:
            dataset = chunk_dataset
        else:
            # Concatenate with existing dataset
            dataset = datasets.concatenate_datasets([dataset, chunk_dataset])

dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Save the dataset
save_path = str(Path(args.output_dir) / "hf_pmnbody")
dataset.save_to_disk(save_path)
print(f"Dataset saved to: {save_path}")
print("Done!")
