"""
Create a Gaussian Random Field (GRF) convergence dataset.

For each cosmology drawn from the prior, computes the convergence power
spectrum C_l^kk using CAMB's lensing source windows, draws a GRF on
the sphere with healpy.synfast, and extracts flat-sky patches.

Much faster than the lognormal pipeline because there are no shells
or iterative convergence calculations -- just one CAMB call per cosmology
and one healpy.synfast call per realisation.
"""

import numpy as np
import camb
from camb.sources import SplinedSourceWindow
from scipy.optimize import minimize_scalar
import healpy as hp
import astropy.units as u
import os
import sys
import time
import yaml
from tqdm import tqdm
from argparse import ArgumentParser

# Add parent directory to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build_lognormal_dataset"))
from projector import projector_func
from get_hsc_redshift_distribution import get_redshift_distribution

N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))
THIS_WORKER = int(os.getenv("SLURM_ARRAY_TASK_ID", 1))


def get_prior_from_str(string, params):
    """Sample from a prior distribution specified by a string."""
    if string == "uniform":
        sample = np.random.uniform(low=params[0], high=params[1], size=(1,))
    if string == "normal":
        sample = params[0] + params[1] * np.random.randn(1)
    return sample.squeeze()


def get_sigma8_from_As(log_As, cosmo_params):
    """Compute |sigma8 - target| for a given log(As), used in minimisation."""
    h = cosmo_params["little_h"]
    Omega_m = cosmo_params["Omega_m"]
    Omega_b = cosmo_params["Omega_b"]
    ns = cosmo_params.get("n_s", 0.965)
    m_nu = cosmo_params.get("m_nu", 0.06)
    w = cosmo_params.get("w", -1)
    sigma_8_target = cosmo_params["sigma_8"]

    Omega_nu = (m_nu / 93.14) / h**2 if m_nu else 0
    Omega_c = Omega_m - Omega_b - Omega_nu

    pars = camb.set_params(
        H0=100 * h,
        omch2=Omega_c * h**2,
        ombh2=Omega_b * h**2,
        ns=ns,
        mnu=m_nu,
        w=w,
        As=np.exp(log_As),
        WantTransfer=True,
        NonLinear=camb.model.NonLinear_both,
    )
    results = camb.get_results(pars)
    sigma8 = results.get_sigma8()
    return abs(sigma8 - sigma_8_target)


def compute_convergence_cl(cosmo_params, z, dndz, lmax=4000):
    """
    Compute the convergence power spectrum C_l^kk using CAMB.

    This is the expensive step (CAMB Boltzmann solver). The returned C_l
    can be reused to draw many GRF realisations cheaply.

    Parameters
    ----------
    cosmo_params : dict
        Must contain little_h, Omega_m, Omega_b, and either sigma_8 or S_8.
    z, dndz : array
        Redshift distribution of sources.
    lmax : int
        Maximum multipole for CAMB. Default 4000 is sufficient for
        2 arcmin pixels (Nyquist ell ~ 5400) since small-scale power
        is subdominant. Higher values are much slower.

    Returns
    -------
    cl_kappa : array of shape (lmax+1,)
    """
    cosmo_params = cosmo_params.copy()

    # Convert S_8 to sigma_8 if needed
    if "sigma_8" not in cosmo_params:
        S8 = cosmo_params["S_8"]
        cosmo_params["sigma_8"] = S8 * (0.3 / cosmo_params["Omega_m"]) ** 0.5

    h = cosmo_params["little_h"]
    Omega_m = cosmo_params["Omega_m"]
    Omega_b = cosmo_params["Omega_b"]
    ns = cosmo_params.get("n_s", 0.965)
    m_nu = cosmo_params.get("m_nu", 0.06)
    w = cosmo_params.get("w", -1)

    Omega_nu = (m_nu / 93.14) / h**2 if m_nu else 0
    Omega_c = Omega_m - Omega_b - Omega_nu

    # Find As that gives the target sigma_8
    log_As = minimize_scalar(
        lambda x: get_sigma8_from_As(x, cosmo_params),
        bounds=[np.log(1e-11), np.log(2e-8)],
        tol=1e-11,
    ).x

    # Set up CAMB with a lensing source window for our n(z)
    pars = camb.set_params(
        H0=100 * h,
        omch2=Omega_c * h**2,
        ombh2=Omega_b * h**2,
        ns=ns,
        mnu=m_nu,
        w=w,
        As=np.exp(log_As),
        WantTransfer=True,
        NonLinear=camb.model.NonLinear_both,
        Want_cl_2D_array=True,
        max_l=lmax,
    )
    pars.SourceWindows = [
        SplinedSourceWindow(z=z, W=dndz, source_type="lensing")
    ]

    results = camb.get_results(pars)
    cls_dict = results.get_source_cls_dict(lmax=lmax, raw_cl=True)
    return cls_dict["W1xW1"]


def draw_grf(cl_kappa, nside, seed=None):
    """
    Draw a GRF realisation on the sphere from a power spectrum.

    This is cheap — just a healpy.synfast call.
    """
    if seed is not None:
        np.random.seed(seed)
    return hp.synfast(cl_kappa, nside=nside, lmax=len(cl_kappa) - 1, new=True)


def correct_kappa(kappa):
    """Deconvolve the HEALPix pixel window function from a map."""
    nside = hp.get_nside(kappa)
    lmax = 3 * nside - 1
    alm_obs = hp.map2alm(kappa, lmax=lmax)
    pw = hp.pixwin(nside, lmax=lmax)
    l, m = hp.Alm.getlm(lmax)
    alm_true = alm_obs / pw[l]
    return hp.alm2map(alm_true, nside=nside, lmax=lmax)


def main(args):
    # Storage arrays
    results = np.empty(
        (args.num_indep_sims, args.num_patches, 1424, 176), dtype=np.float16
    )
    labels = np.empty((args.num_indep_sims, 4), dtype=np.float32)

    # Load redshift distribution
    print("Loading the redshift distributions")
    fits_file = os.path.join(args.input_dir, "hsc_y3/nz.fits")
    z, dndz = get_redshift_distribution(fits_file)

    # Load prior config
    config_path = os.path.join(os.path.dirname(__file__), "prior_config.yaml")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    cosmo_params_prior = data["cosmo_params"]
    redshift_shift_prior = data["redshift_shift"]

    nside = args.nside
    lmax = 3 * nside - 1

    for i in tqdm(range(args.num_indep_sims)):
        seed = THIS_WORKER * args.num_indep_sims + i

        # Sample cosmological parameters
        Omega_m = get_prior_from_str(
            cosmo_params_prior["prior"], cosmo_params_prior["prior_params"][0]
        )
        S_8 = get_prior_from_str(
            cosmo_params_prior["prior"], cosmo_params_prior["prior_params"][1]
        )
        delta_z = get_prior_from_str(
            redshift_shift_prior["prior"], redshift_shift_prior["prior_params"]
        )

        print(
            f"Running for seed = {seed}, Omega_m = {Omega_m:.2f}, "
            f"S8 = {S_8:.2f}, delta_z = {delta_z:.2f}"
        )

        h = 0.7
        Omega_bh2 = 0.0224
        cosmo_params = {
            "little_h": h,
            "Omega_m": Omega_m,
            "S_8": S_8,
            "Omega_b": Omega_bh2 / h**2,
        }

        start = time.time()
        z_shifted = np.clip(z - delta_z, a_min=0, a_max=None)

        cl_kappa = compute_convergence_cl(cosmo_params, z_shifted, dndz, lmax=lmax)
        kappa_sphere = draw_grf(cl_kappa, nside=nside, seed=seed)
        elapsed = time.time() - start
        print(f"GRF convergence map generated in {elapsed:.1f}s")

        # Extract patches
        kappa_corrected = correct_kappa(kappa_sphere)
        kappa_patches = np.empty((args.num_patches, 1424, 176))
        rng = np.random.default_rng(seed)
        lon_cs = rng.uniform(low=0, high=360, size=(args.num_patches,))
        lat_cs = rng.uniform(low=-90, high=90, size=(args.num_patches,))

        for j, patch_center in enumerate(zip(lon_cs, lat_cs)):
            kappa_proj = projector_func(
                kappa_corrected,
                patch_center=patch_center,
                patch_shape=(1424, 176),
                reso=args.reso * u.arcmin.to(u.degree),
                proj_mode="gnomonic",
            )
            kappa_patches[j] = kappa_proj.T

        results[i] = kappa_patches
        labels[i] = [seed, Omega_m, S_8, delta_z]

        if args.debug_mode:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(
            args.output_dir,
            f"grf_sim_{THIS_WORKER}_{args.reso}arcmin_{nside}nside.npz",
        ),
        train=results,
        label=labels,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--num_patches", default=5, type=int)
    parser.add_argument("--nside", default=2048, type=int)
    parser.add_argument("--reso", default=2, type=float, help="Resolution in arcmin")
    parser.add_argument("--num_indep_sims", required=True, type=int)
    parser.add_argument("--debug_mode", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
