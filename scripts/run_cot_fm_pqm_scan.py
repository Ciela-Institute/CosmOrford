"""Parallel PQMass num_refs scan for cot_fm.py.

Launches one cot_fm.py process per num_refs value simultaneously, assigning
GPUs in round-robin order. Designed for clusters where you submit a single
multi-GPU job and want all configurations training in parallel.

The original cot_fm.py is untouched — this script is just a launcher.

Usage
-----
# Scan num_refs 10, 50, 100, 200 on GPUs 0,1,2,3:
python scripts/run_cot_fm_pqm_scan.py \\
    --exp_config configs/experiments/unet_exp/baseline.yaml \\
    --num_refs 10 50 100 200 \\
    --gpus 0 1 2 3

# Fix simulation budget while scanning num_refs:
python scripts/run_cot_fm_pqm_scan.py \\
    --exp_config configs/experiments/unet_exp/baseline.yaml \\
    --num_refs 10 50 100 200 \\
    --sim_budget 5000

# Limit to at most 2 concurrent jobs:
python scripts/run_cot_fm_pqm_scan.py \\
    --exp_config configs/experiments/unet_exp/baseline.yaml \\
    --num_refs 10 50 100 200 500 \\
    --max_parallel 2

Logs
----
Each num_refs run writes to <log_dir>/num_refs_<N>.log (stdout+stderr combined).
Default log_dir: cot_fm_pqm_scan_logs/ next to this script.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Parallel PQMass num_refs scan launcher for cot_fm.py"
)
parser.add_argument(
    "--exp_config",
    type=str,
    required=True,
    help="Path to experiment YAML passed to cot_fm.py",
)
parser.add_argument(
    "--num_refs",
    type=int,
    nargs="+",
    default=[10, 50, 100, 200, 500],
    help="List of num_refs values to scan (default: 10 50 100 200 500)",
)
parser.add_argument(
    "--sim_budget",
    type=int,
    default=None,
    help="Fix the simulation budget for all runs (passed through to cot_fm.py).",
)
parser.add_argument(
    "--gpus",
    type=str,
    nargs="*",
    default=None,
    help="GPU IDs to use (e.g. --gpus 0 1 2). Auto-detected if not set.",
)
parser.add_argument(
    "--max_parallel",
    type=int,
    default=None,
    help="Cap on simultaneous jobs. Defaults to number of GPUs (or num_refs count if CPU-only).",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="Directory for per-run log files. Default: cot_fm_pqm_scan_logs/ next to this script.",
)
parser.add_argument(
    "--python",
    type=str,
    default=sys.executable,
    help="Python interpreter to use (default: same as this script).",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
COT_FM = REPO_ROOT / "cosmoford" / "emulator" / "cot_fm.py"

if not COT_FM.exists():
    sys.exit(f"ERROR: cot_fm.py not found at {COT_FM}")

log_dir = Path(args.log_dir) if args.log_dir else SCRIPT_DIR / "cot_fm_pqm_scan_logs"
log_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpus() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []

if args.gpus is not None:
    gpu_ids = [str(g) for g in args.gpus]
else:
    gpu_ids = detect_gpus()

cpu_only = len(gpu_ids) == 0
if cpu_only:
    print("WARNING: no GPUs detected — running on CPU (will be very slow).")

max_parallel = args.max_parallel or (len(gpu_ids) if not cpu_only else len(args.num_refs))
max_parallel = max(1, max_parallel)

# ---------------------------------------------------------------------------
# Launch jobs
# ---------------------------------------------------------------------------

num_refs_list = args.num_refs
print(f"num_refs : {num_refs_list}")
print(f"GPUs     : {gpu_ids if not cpu_only else 'CPU'}")
print(f"Max parallel: {max_parallel}")
print(f"Logs     : {log_dir}")
print()

procs: list[tuple[int, subprocess.Popen, Path]] = []  # (num_refs, proc, log_path)
pending = list(enumerate(num_refs_list))

def _start(slot: int, num_refs: int) -> tuple[int, subprocess.Popen, Path]:
    gpu = gpu_ids[slot % len(gpu_ids)] if not cpu_only else None
    log_path = log_dir / f"num_refs_{num_refs}.log"
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    cmd = [args.python, str(COT_FM), "--exp_config", args.exp_config, "--num_refs", str(num_refs)]
    if args.sim_budget is not None:
        cmd += ["--sim_budget", str(args.sim_budget)]

    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
    gpu_label = f"GPU {gpu}" if gpu is not None else "CPU"
    print(f"[num_refs={num_refs:>5}] Started PID {proc.pid} on {gpu_label}  →  {log_path.name}")
    return num_refs, proc, log_path

# Fill up to max_parallel slots
slot = 0
while pending and len(procs) < max_parallel:
    _, num_refs = pending.pop(0)
    procs.append(_start(slot, num_refs))
    slot += 1

# Poll and refill
failed: list[int] = []
succeeded: list[int] = []

while procs:
    time.sleep(5)
    still_running = []
    for num_refs, proc, log_path in procs:
        rc = proc.poll()
        if rc is None:
            still_running.append((num_refs, proc, log_path))
        else:
            if rc == 0:
                print(f"[num_refs={num_refs:>5}] Finished OK")
                succeeded.append(num_refs)
            else:
                print(f"[num_refs={num_refs:>5}] FAILED (exit {rc}) — see {log_path}")
                failed.append(num_refs)
            if pending:
                _, next_num_refs = pending.pop(0)
                still_running.append(_start(slot, next_num_refs))
                slot += 1
    procs = still_running

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 50)
print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed.")
if succeeded:
    print(f"  OK     : {succeeded}")
if failed:
    print(f"  FAILED : {failed}")
    sys.exit(1)
