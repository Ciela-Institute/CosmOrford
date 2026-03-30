#!/usr/bin/env python3
"""Build a readable markdown report for HOS-NPE ablation-transfer runs."""

import argparse
import csv
import datetime as dt
import pathlib
import statistics
import sys
from collections import defaultdict, Counter


def read_status(path: pathlib.Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _to_float(x):
    if x in (None, "", "None"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _fmt(x, ndigits):
    if x is None:
        return "NA"
    return f"{x:.{ndigits}f}"


def _is_true(x):
    return str(x).lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser(description="Write markdown report from HOS-NPE ablation status CSV")
    parser.add_argument("--status-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="HOS-NPE Ablation Transfer Monitoring Report")
    args = parser.parse_args()

    rows = read_status(pathlib.Path(args.status_csv))
    if not rows:
        raise ValueError(f"Empty status CSV: {args.status_csv}")

    by_regime = defaultdict(list)
    state_counts = Counter()
    issues = []
    for r in rows:
        by_regime[r["regime_id"]].append(r)
        state_counts[r["state"]] += 1
        if r["state"].startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")):
            issues.append(f"Job `{r['job_id']}` ({r['regime_id']}, seed {r['seed']}) ended with `{r['state']}`.")
        if r["state"].startswith("COMPLETED") and not _is_true(r["results_exists"]):
            issues.append(
                f"Job `{r['job_id']}` ({r['regime_id']}, seed {r['seed']}) completed but `results.json` is missing."
            )
        if _is_true(r["results_exists"]):
            if not _is_true(r.get("posterior_samples_norm")) or not _is_true(r.get("posterior_samples_phys")):
                issues.append(
                    f"Run `{r['run_name']}` missing one of posterior sample files (`norm`/`phys`)."
                )
            if int(float(r.get("n_plots") or 0)) == 0 or int(float(r.get("n_contours") or 0)) == 0:
                issues.append(
                    f"Run `{r['run_name']}` missing posterior plots/contours."
                )

    lines = []
    lines.append(f"# {args.title}")
    lines.append("")
    lines.append(f"_Generated: {dt.datetime.utcnow().isoformat()}Z_")
    lines.append("")
    lines.append("## Overall status")
    lines.append("")
    lines.append(f"- Total submitted runs tracked: **{len(rows)}**")
    for k, v in sorted(state_counts.items()):
        lines.append(f"- `{k}`: **{v}**")
    lines.append("")

    lines.append("## Per-regime summary")
    lines.append("")
    lines.append("| Regime | Runs | Completed | With results | Mean FoM | Std FoM | Mean best val NLL |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for regime_id in sorted(by_regime.keys()):
        g = by_regime[regime_id]
        completed = sum(1 for x in g if x["state"].startswith("COMPLETED"))
        with_results = [x for x in g if _is_true(x["results_exists"])]
        foms = [_to_float(x["fom_mean"]) for x in with_results]
        foms = [x for x in foms if x is not None]
        nlls = [_to_float(x["best_val_nll"]) for x in with_results]
        nlls = [x for x in nlls if x is not None]
        mean_fom = statistics.mean(foms) if foms else None
        std_fom = statistics.pstdev(foms) if len(foms) > 1 else (0.0 if len(foms) == 1 else None)
        mean_nll = statistics.mean(nlls) if nlls else None
        lines.append(
            f"| `{regime_id}` | {len(g)} | {completed} | {len(with_results)} | "
            f"{_fmt(mean_fom, 3)} | "
            f"{_fmt(std_fom, 3)} | "
            f"{_fmt(mean_nll, 4)} |"
        )
    lines.append("")

    lines.append("## Artifact checks")
    lines.append("")
    missing_artifacts = []
    for r in rows:
        if not _is_true(r["results_exists"]):
            continue
        np_norm = _is_true(r.get("posterior_samples_norm"))
        np_phys = _is_true(r.get("posterior_samples_phys"))
        n_plots = int(float(r.get("n_plots") or 0))
        n_contours = int(float(r.get("n_contours") or 0))
        if not (np_norm and np_phys and n_plots > 0 and n_contours > 0):
            missing_artifacts.append(
                f"- `{r['run_name']}`: norm={np_norm}, phys={np_phys}, plots={n_plots}, contours={n_contours}"
            )
    if missing_artifacts:
        lines.extend(missing_artifacts)
    else:
        lines.append("- All completed runs with `results.json` also have posterior samples and posterior plot artifacts.")
    lines.append("")

    lines.append("## Issues and actions")
    lines.append("")
    if issues:
        lines.append("### Observed issues")
        lines.append("")
        lines.extend([f"- {i}" for i in issues])
        lines.append("")
        lines.append("### Recommended actions")
        lines.append("")
        lines.append(
            "- Resubmit failed/cancelled/timeout jobs with:\n"
            "  `python scripts/rerun_failed_hos_npe_ablation.py --manifest <manifest.csv>`"
        )
        lines.append("- Re-run status + report generation after reruns complete.")
    else:
        lines.append("- No blocking issues detected in current snapshot.")
    lines.append("")

    lines.append("## Detailed run table")
    lines.append("")
    lines.append("| Regime | Seed | Job | State | FoM mean | FoM std | NLL | Results dir |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---|")
    for r in sorted(rows, key=lambda x: (x["regime_id"], int(x["seed"]))):
        lines.append(
            f"| `{r['regime_id']}` | {r['seed']} | `{r['job_id']}` | `{r['state']}` | "
            f"{r['fom_mean'] or 'NA'} | {r['fom_std'] or 'NA'} | {r['best_val_nll'] or 'NA'} | "
            f"`{r['results_dir']}` |"
        )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote markdown report: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
