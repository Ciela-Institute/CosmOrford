#!/usr/bin/env python3
"""Regenerate posterior contour plots from saved posterior samples using getdist."""

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from getdist import MCSamples
from getdist import plots as getdist_plots


def _iter_results_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("**/results.json")):
        yield path.parent


def _write_getdist_contour(samples_phys: np.ndarray, out_path: Path, title: str) -> None:
    gd_samples = MCSamples(
        samples=np.asarray(samples_phys, dtype=np.float64),
        names=["omegam", "s8"],
        labels=[r"\Omega_m", r"S_8"],
    )
    plotter = getdist_plots.get_single_plotter(width_inch=5.8, ratio=0.85)
    plotter.settings.num_plot_contours = 2
    plotter.settings.alpha_filled_add = 0.65
    plotter.plot_2d(gd_samples, "omegam", "s8", filled=True)
    ax = plt.gca()
    ax.set_title(title)
    plotter.fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.fig.savefig(out_path, dpi=160)
    plt.close(plotter.fig)


def process_results_dir(results_dir: Path, overwrite: bool) -> int:
    results_file = results_dir / "results.json"
    if not results_file.exists():
        return 0

    payload = json.loads(results_file.read_text())
    samples_file = results_dir / "posterior_samples_phys.npy"
    if not samples_file.exists():
        return 0

    samples = np.load(samples_file)
    if samples.ndim != 3 or samples.shape[-1] != 2:
        return 0

    contour_files = []
    written = 0
    for obs_idx in range(samples.shape[0]):
        out_path = results_dir / f"posterior_fiducial_obs{obs_idx:02d}_contour.png"
        if out_path.exists() and not overwrite:
            contour_files.append(out_path.name)
            continue
        _write_getdist_contour(
            samples[obs_idx],
            out_path=out_path,
            title=f"Posterior contours (fiducial realization {obs_idx})",
        )
        written += 1
        contour_files.append(out_path.name)

    payload["posterior_contour_backend"] = "getdist"
    payload["posterior_contour_files"] = contour_files
    results_file.write_text(json.dumps(payload, indent=2))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate contour plots with getdist")
    parser.add_argument(
        "--results-root",
        default=str(Path.home() / "experiments" / "npe_results"),
        help="Root directory containing run_name/budget-*/results.json outputs",
    )
    parser.add_argument(
        "--run-name-glob",
        default="hos_npe_*",
        help="Glob selecting run directories under --results-root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing contour files",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    run_roots = sorted(root.glob(args.run_name_glob))
    total_dirs = 0
    total_written = 0
    for run_root in run_roots:
        for results_dir in _iter_results_dirs(run_root):
            total_dirs += 1
            total_written += process_results_dir(results_dir, overwrite=args.overwrite)

    print(f"Processed results dirs: {total_dirs}")
    print(f"Contours written: {total_written}")


if __name__ == "__main__":
    main()
