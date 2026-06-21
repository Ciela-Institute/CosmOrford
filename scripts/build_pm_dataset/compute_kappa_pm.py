import argparse
import glob
import math
import os
import re
import time
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import astropy.units as u
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
import numpy as np
from astropy.table import Table
from jax.image import scale_and_translate
from jax.scipy.ndimage import map_coordinates
from tqdm import tqdm


def _load_batch_payload(batch_params, batch_seeds, seed_to_file):
    """
    Runs in a background thread. Loads raw numpy density planes from disk
    and packages them with their corresponding cosmological parameters.
    """
    t0 = time.perf_counter()

    loaded_data = []
    loaded_seed = []
    loaded_theta = []
    loaded_files = []

    for row, seed in zip(batch_params, batch_seeds):
        fp = seed_to_file.get(int(seed))
        if fp is None:
            continue
        loaded_data.append(np.load(fp, allow_pickle=False))
        loaded_seed.append(float(row[0]))
        loaded_theta.append(row[1:3].astype(np.float32))  # [Omega_c, S8]
        loaded_files.append(fp)

    load_s = time.perf_counter() - t0

    if not loaded_data:
        return None, load_s

    # Stack data and prepare random redshift uncertainty perturbations
    data_np = np.stack(loaded_data, axis=0).astype(np.float32, copy=False)
    B_actual = data_np.shape[0]
    delta_z_batch = np.random.normal(loc=0.0, scale=0.022, size=(B_actual, 1)).astype(np.float32)
    seed_col = np.asarray(loaded_seed, dtype=np.float32).reshape(-1, 1)
    theta_2 = np.asarray(loaded_theta, dtype=np.float32)

    payload = {
        "data_np": data_np,
        "delta_z_batch": delta_z_batch,
        "seed_col": seed_col,
        "theta_2": theta_2,
        "loaded_files": loaded_files,
    }
    return payload, load_s


def _save_batch_payload(loaded_files, kappas, seed_col, theta_2, delta_z_batch, output_dir):
    """
    Runs in a background thread. Compresses and writes the computed kappa arrays
    to disk as .npz files.
    """
    t0 = time.perf_counter()
    for i, in_fp in enumerate(loaded_files):
        out_fp = _output_file_for_input(in_fp, output_dir)
        np.savez_compressed(
            out_fp,
            maps=kappas[i],
            seed=seed_col[i],
            theta_2=theta_2[i],
            delta_z=delta_z_batch[i],
            theta=np.concatenate([theta_2[i], delta_z_batch[i]], axis=0).astype(np.float32),
            source_file=np.array([os.path.basename(in_fp)]),
        )
    return time.perf_counter() - t0


def convergence_Born(cosmo, density_planes, r, a, dx, dz, coords, z_source):
    """
    Compute the Born convergence
    Args:
      cosmo: `Cosmology`, cosmology object.
      density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use
      coords: a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
      z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
      name: `string`, name of the operation.
    Returns:
      `Tensor` of shape [batch_size, N, Nz], of convergence values.
    """
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c) ** 2
    r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

    convergence = 0
    n_planes = len(r)

    # Use lax.scan for efficient looping over density planes in JAX
    def scan_fn(carry, i):
        density_planes, a, r = carry

        p = density_planes[:, :, i]
        density_normalization = dz * r[i] / a[i]
        p = (p - p.mean()) / p.mean() * constant_factor * density_normalization

        im = map_coordinates(p, coords * r[i] / dx - 0.5, order=1, mode="wrap")

        return carry, im * jnp.clip(1.0 - (r[i] / r_s), 0, 1000).reshape([-1, 1, 1])

    _, convergence = jax.lax.scan(scan_fn, (density_planes, a, r), jnp.arange(n_planes))

    return convergence.sum(axis=0)


def simps_manual(f, a, b, N):
    """
    Manual Simpson's rule implementation for numerical integration in JAX.
    """
    x = jnp.linspace(a, b, N + 1)
    dx = (b - a) / N
    y = jnp.stack([f(xi) for xi in x])
    return dx / 3 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)


def upsample(image):
    """
    Upsamples the density planes to a higher resolution (1424x1424)
    using Lanczos3 interpolation.
    """
    new_shape = (1424, 1424)
    old_shape = image[0, 0].shape
    scale = jnp.array([new_shape[0] / old_shape[0], new_shape[1] / old_shape[1]])
    translation = jnp.array([0.0, 0.0])

    resampling_function = partial(
        scale_and_translate,
        shape=new_shape,
        spatial_dims=(0, 1),
        scale=scale,
        translation=translation,
        method="lanczos3",
        antialias=True)

    # Apply across the batch and channel dimensions
    return jax.vmap(jax.vmap(resampling_function))(image=image)


# Compile the core logic to the GPU and vectorize across the batch dimension
@jax.jit
@partial(jax.vmap, in_axes=(0, 0, None, None))
def compute_kappa(data, cosmo_params, base_nzs, bins_arr):
    box_size = [1000, 1000, 1000]
    mesh_shape = [1424, 1424, 36]
    field_size = 50

    dx = box_size[0] / mesh_shape[0]
    dz = 102.5390625

    # Base cosmological constants
    n_s = 0.965
    Omega_b = 0.0224
    h = 0.67

    # Extract varied parameters for this specific simulation
    oc = cosmo_params[1]
    S8 = cosmo_params[2]
    delta_z = cosmo_params[3]

    om = oc + Omega_b
    sigma8 = S8 / jnp.sqrt(om / 0.3)

    cosmo = jc.Planck15(Omega_c=oc, sigma8=sigma8, Omega_b=Omega_b, n_s=n_s, h=h)

    # Generate the angular coordinate grid
    xgrid, ygrid = np.meshgrid(
        np.linspace(0, field_size, mesh_shape[0], endpoint=False),
        np.linspace(0, field_size, mesh_shape[1], endpoint=False),
    )
    coords = jnp.array((np.stack([xgrid, ygrid], axis=0) * u.deg).to(u.rad))

    r = (jnp.arange(mesh_shape[-1]) + 0.5) * dz
    a = jc.background.a_of_chi(cosmo, r)

    data = upsample(data)

    # OPTIMIZATION: Shift and compute ONLY for Bin 2 (index 1) to save GPU time
    nz_bin2 = jc.redshift.systematic_shift(base_nzs[1], delta_z)

    kappas = []
    # Loop over the 3 input lightcones
    for i in range(3):
        lightcone = data[:, i, :, :]
        lightcone = jnp.transpose(lightcone, axes=(1, 2, 0))

        # Integrate convergence along the lightcone for bin 2 only
        kappa_bin2 = simps_manual(
            lambda z: nz_bin2(z).reshape([-1, 1, 1])
                      * convergence_Born(cosmo, lightcone, r, a, dx, dz, coords, z),
            0.01,
            2,
            N=32,
        ).squeeze()

        # Slice into patches
        for j in range(1424 // 176):
            kappa_bin_slice = kappa_bin2[:, j * 176: (j + 1) * 176]
            kappas.append(kappa_bin_slice)

    kappas = jnp.stack(kappas, axis=0)

    return kappas


def _output_file_for_input(input_path: str, output_dir: str) -> Path:
    """Constructs the output filename by replacing .npy with _kappa.npz"""
    return Path(output_dir) / f"{Path(input_path).stem}_kappa.npz"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sims_path', type=str, default='/home/juzgh/scratch/nbody_sims',
                        help='Path to the simulations')
    parser.add_argument('--params_path', type=str, default='cosmo_parameters_uniform_30000.npy',
                        help='Path to the cosmological parameters file')
    parser.add_argument('--nz_path', type=str, default='nz.fits', help='Path to redshift distribution FITS file')
    parser.add_argument('--output_dir', type=str, default='/home/rouzib/scratch/kappa',
                        help='Directory to save processed kappa patches')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--chunk_size', type=int, default=500, help='Number of samples per chunk (Standard Mode)')
    # Add arguments to support splitting the dataset across array jobs
    parser.add_argument('--total_chunks', type=int, default=None,
                        help='Total number of array jobs (overrides chunk_size)')
    parser.add_argument('--chunk_idx', type=int, default=None, help='Specific array job ID to run (0-indexed)')
    args = parser.parse_args()

    print(f"Loading data from {args.sims_path}...")

    data_dir = args.sims_path
    files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))

    print(f"Found {len(files)} files.")


    def _extract_seed_from_filename(path: str):
        """Parses the simulation seed number from the filename to match with cosmology params."""
        base = os.path.basename(path)
        m = re.search(r'(\d+)(?=\.npy$)', base)
        if m:
            return int(m.group(1))
        all_nums = re.findall(r'\d+', base)
        if not all_nums:
            return None
        return int(all_nums[-1])


    # Create a mapping of seeds to file paths
    seed_to_file = {}
    for p in files:
        s = _extract_seed_from_filename(p)
        if s is not None and s not in seed_to_file:
            seed_to_file[s] = p

    # Load cosmological parameters and filter out missing files
    params_all = np.load(args.params_path)
    seeds_all = params_all[:, 0].astype(int)

    mask_has_file = np.array([seed in seed_to_file for seed in seeds_all])
    params = params_all[mask_has_file]
    seeds = params[:, 0].astype(int)

    # Load and prep redshift distributions (n(z))
    nz_hsc = Table.read(args.nz_path)
    z_mid = jnp.array(np.asarray(nz_hsc['Z_MID'], dtype=np.float32))
    bins_arr = jnp.stack([
        jnp.array(np.asarray(nz_hsc[f'BIN{i}'], dtype=np.float32)) for i in range(1, 4)
    ], axis=0)

    # Build the base KDE n(z) objects once. Shift will be applied per-sim inside the JIT compilation.
    base_nzs = [
        jc.redshift.kde_nz(z_mid, bins_arr[i], bw=0.015)
        for i in range(3)
    ]

    num_samples = params.shape[0]

    # --- ARRAY JOB LOGIC ---
    # If array arguments are provided, divide the data globally and only process the requested chunk.
    if args.total_chunks is not None and args.chunk_idx is not None:
        print(f"Running in Array Job Mode: Processing chunk {args.chunk_idx + 1} of {args.total_chunks}")
        chunk_size = math.ceil(num_samples / args.total_chunks)
        chunk_start = args.chunk_idx * chunk_size
        chunk_end = min((args.chunk_idx + 1) * chunk_size, num_samples)
        chunks_to_process = [(args.chunk_idx, chunk_start, chunk_end)]
    else:
        # Standard sequential mode
        print("Running in Standard Mode (Sequential chunking)")
        chunk_size = args.chunk_size
        num_total_chunks = math.ceil(num_samples / chunk_size)
        chunks_to_process = [(i, i * chunk_size, min((i + 1) * chunk_size, num_samples)) for i in
                             range(num_total_chunks)]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    timing_totals = {"load": 0.0, "compute": 0.0, "d2h": 0.0, "save": 0.0, "wall": 0.0}
    timed_batches = 0

    # Process data in chunks to manage memory footprint
    for chunk_idx, chunk_start, chunk_end in chunks_to_process:
        params_chunk = params[chunk_start:chunk_end]
        seeds_chunk = seeds[chunk_start:chunk_end]

        num_in_chunk = params_chunk.shape[0]
        num_batches = math.ceil(num_in_chunk / args.batch_size)

        # Determine start and end indices for each batch
        batch_ranges = []
        for b in range(num_batches):
            b_start = b * args.batch_size
            b_end = min((b + 1) * args.batch_size, num_in_chunk)
            if b_start < b_end:
                batch_ranges.append((b_start, b_end))

        if not batch_ranges:
            print(f"Chunk {chunk_idx} has no data. Skipping.")
            continue

        # Setup the thread limits.
        # Match save workers to loader workers to keep the pipeline balanced.
        loader_workers = 3
        save_workers = loader_workers
        prefetch_depth = max(4, 2 * loader_workers)
        max_save_inflight = 2 * save_workers

        # Context managers for the two thread pools
        with ThreadPoolExecutor(max_workers=loader_workers) as load_pool, \
                ThreadPoolExecutor(max_workers=save_workers) as save_pool:

            pending = list(enumerate(batch_ranges))
            inflight = {}
            save_inflight = set()


            def submit_one():
                """Pops a batch off the pending list and submits it to the background loader."""
                if not pending:
                    return False
                bi, (s, e) = pending.pop(0)
                fut = load_pool.submit(
                    _load_batch_payload,
                    params_chunk[s:e],
                    seeds_chunk[s:e],
                    seed_to_file,
                )
                inflight[fut] = bi
                return True


            # Prime the loader queue before entering the main loop
            for _ in range(min(prefetch_depth, len(pending))):
                submit_one()

            pbar = tqdm(total=len(batch_ranges), desc=f"Batches for chunk {chunk_idx + 1}")

            while inflight:
                # Wait for any background loader thread to finish a batch
                t_wait0 = time.perf_counter()
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                wait_s = time.perf_counter() - t_wait0

                for fut in done:
                    bi = inflight.pop(fut)

                    t_wall0 = time.perf_counter()
                    payload, load_s = fut.result()
                    timing_totals["load"] += load_s

                    # Immediately submit the next batch to keep the loader queue full
                    while len(inflight) < prefetch_depth and submit_one():
                        pass

                    if payload is None:
                        pbar.update(1)
                        continue

                    data_np = payload["data_np"]
                    delta_z_batch = payload["delta_z_batch"]
                    seed_col = payload["seed_col"]
                    theta_2 = payload["theta_2"]
                    loaded_files = payload["loaded_files"]

                    # Send data to GPU and compute
                    t0 = time.perf_counter()
                    cosmo_params_chunk = jnp.array(np.concatenate([seed_col, theta_2, delta_z_batch], axis=1))
                    kappa_dev = compute_kappa(jnp.asarray(data_np), cosmo_params_chunk, base_nzs, bins_arr)
                    kappa_dev.block_until_ready()  # Wait for GPU to finish for accurate timing
                    t1 = time.perf_counter()

                    # Transfer results back from Device to Host (GPU to CPU)
                    kappas = np.asarray(kappa_dev, dtype=np.float32)
                    t2 = time.perf_counter()

                    # Throttle the save queue to prevent Out-Of-Memory (OOM) errors.
                    # If the queue is at capacity, block until at least one save finishes.
                    while len(save_inflight) >= max_save_inflight:
                        done_saves, save_inflight = wait(save_inflight, return_when=FIRST_COMPLETED)
                        for save_fut in done_saves:
                            timing_totals["save"] += save_fut.result()

                    # Submit the processed batch to the background save thread pool
                    fut_save = save_pool.submit(
                        _save_batch_payload,
                        loaded_files, kappas, seed_col, theta_2, delta_z_batch, args.output_dir
                    )
                    save_inflight.add(fut_save)

                    t3 = time.perf_counter()

                    batch_wall = t3 - t_wall0
                    compute_s = t1 - t0
                    d2h_s = t2 - t1

                    timing_totals["compute"] += compute_s
                    timing_totals["d2h"] += d2h_s
                    timing_totals["wall"] += batch_wall
                    timed_batches += 1

                    print(
                        f"[chunk {chunk_idx + 1} batch {bi + 1}/{len(batch_ranges)}] "
                        f"wait={wait_s:.3f}s load={load_s:.3f}s "
                        f"compute={compute_s:.3f}s d2h={d2h_s:.3f}s wall={batch_wall:.3f}s "
                        f"load_inflight={len(inflight)} save_inflight={len(save_inflight)}"
                    )

                    pbar.update(1)

            pbar.close()

            # Ensure all background saves complete before closing the pools and moving to the next chunk
            if save_inflight:
                print(
                    f"Waiting for remaining {len(save_inflight)} background saves to finish for chunk {chunk_idx + 1}...")
                done_saves, _ = wait(save_inflight, return_when=ALL_COMPLETED)
                for save_fut in done_saves:
                    timing_totals["save"] += save_fut.result()

    if timed_batches > 0:
        print("\n=== Timing summary ===")
        print(f"batches: {timed_batches}")
        print(f"load total:    {timing_totals['load']:.3f}s | avg: {timing_totals['load'] / timed_batches:.3f}s")
        print(f"compute total: {timing_totals['compute']:.3f}s | avg: {timing_totals['compute'] / timed_batches:.3f}s")
        print(f"d2h total:     {timing_totals['d2h']:.3f}s | avg: {timing_totals['d2h'] / timed_batches:.3f}s")
        print(
            f"save total:    {timing_totals['save']:.3f}s | avg: {timing_totals['save'] / timed_batches:.3f}s (background CPU time)")
        print(f"wall total:    {timing_totals['wall']:.3f}s | avg: {timing_totals['wall'] / timed_batches:.3f}s")

    print(f"Per-sample files saved in: {args.output_dir}")
    print("Done!")
