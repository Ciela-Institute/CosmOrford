"""Parallel simulation-budget scan for cot_fm.py.

Launches one cot_fm.py process per budget simultaneously, assigning GPUs
in round-robin order. Designed for clusters where you submit a single
multi-GPU job and run everything in one go.

The original cot_fm.py is untouched — this script is just a launcher.

Usage
-----
# Run budgets 100, 1000, 5000 on GPUs 0,1,2:
python scripts/run_cot_fm_budget_scan.py \\
    --exp_config configs/experiments/unet_exp/final.yaml \\
    --budgets 100 1000 5000 \\
    --gpus 0 1 2

# Use all available GPUs (auto-detected):
python scripts/run_cot_fm_budget_scan.py \\
    --exp_config configs/experiments/unet_exp/final.yaml \\
    --budgets 100 1000 5000 10000 20200

# Limit to at most 2 concurrent jobs:
python scripts/run_cot_fm_budget_scan.py \\
    --exp_config configs/experiments/unet_exp/final.yaml \\
    --budgets 100 1000 5000 \\
    --max_parallel 2

Logs
----
Each budget writes to <log_dir>/budget_<N>.log (stdout+stderr combined).
Default log_dir: cot_fm_scan_logs/ next to this script.
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
    description="Parallel budget scan launcher for cot_fm.py"
)
parser.add_argument(
    "--exp_config",
    type=str,
    required=True,
    help="Path to experiment YAML passed to cot_fm.py",
)
parser.add_argument(
    "--budgets",
    type=int,
    nargs="+",
    default=[100, 1000, 5000, 10000, 20200],
    help="List of sim budgets to run (default: 100 1000 5000 10000 20200)",
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
    help="Cap on simultaneous jobs. Defaults to number of GPUs (or budgets if CPU-only).",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="Directory for per-budget log files. Default: cot_fm_scan_logs/ next to this script.",
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

log_dir = Path(args.log_dir) if args.log_dir else SCRIPT_DIR / "cot_fm_scan_logs"
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

max_parallel = args.max_parallel or (len(gpu_ids) if not cpu_only else len(args.budgets))
max_parallel = max(1, max_parallel)

# ---------------------------------------------------------------------------
# Launch jobs
# ---------------------------------------------------------------------------

budgets = args.budgets
print(f"Budgets : {budgets}")
print(f"GPUs    : {gpu_ids if not cpu_only else 'CPU'}")
print(f"Max parallel: {max_parallel}")
print(f"Logs    : {log_dir}")
print()

procs: list[tuple[int, subprocess.Popen, Path]] = []  # (budget, proc, log_path)
pending = list(enumerate(budgets))  # (gpu_slot_index, budget)

def _start(slot: int, budget: int) -> tuple[int, subprocess.Popen, Path]:
    gpu = gpu_ids[slot % len(gpu_ids)] if not cpu_only else None
    log_path = log_dir / f"budget_{budget}.log"
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    cmd = [args.python, str(COT_FM), "--exp_config", args.exp_config, "--sim_budget", str(budget)]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
    gpu_label = f"GPU {gpu}" if gpu is not None else "CPU"
    print(f"[budget={budget:>6}] Started PID {proc.pid} on {gpu_label}  →  {log_path.name}")
    return budget, proc, log_path

# Fill up to max_parallel slots
slot = 0
while pending and len(procs) < max_parallel:
    _, budget = pending.pop(0)
    procs.append(_start(slot, budget))
    slot += 1

# Poll and refill
failed: list[int] = []
succeeded: list[int] = []

while procs:
    time.sleep(5)
    still_running = []
    for budget, proc, log_path in procs:
        rc = proc.poll()
        if rc is None:
            still_running.append((budget, proc, log_path))
        else:
            if rc == 0:
                print(f"[budget={budget:>6}] Finished OK")
                succeeded.append(budget)
            else:
                print(f"[budget={budget:>6}] FAILED (exit {rc}) — see {log_path}")
                failed.append(budget)
            # Start a new job from the pending queue if any remain
            if pending:
                _, next_budget = pending.pop(0)
                still_running.append(_start(slot, next_budget))
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
