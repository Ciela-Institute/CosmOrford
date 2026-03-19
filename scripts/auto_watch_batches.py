#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import pathlib
import re
import subprocess
import time


def sacct_state(job_id: str) -> str:
    try:
        out = subprocess.run(
            ["sacct", "-n", "-P", "-j", job_id, "-o", "State"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except subprocess.CalledProcessError:
        return "UNKNOWN"
    for line in out:
        s = line.strip().split("|")[0]
        if s:
            return s
    return "UNKNOWN"


def read_manifest(path: pathlib.Path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return rows


def write_manifest(path: pathlib.Path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def resubmit(config: str, seed: str, exclude_nodes: str = "") -> str:
    cmd = ["sbatch"]
    if exclude_nodes:
        cmd.append(f"--exclude={exclude_nodes}")
    cmd.extend([f"--export=ALL,CONFIG={config},SEED={seed}", "scripts/submit_job.sh"])
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    m = re.search(r"Submitted batch job (\d+)", p.stdout)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output: {p.stdout}")
    return m.group(1)


def _job_id_as_int(job_id: str) -> int:
    try:
        return int(job_id)
    except (TypeError, ValueError):
        return -1


def latest_attempt_rows(rows):
    latest = {}
    for row in rows:
        key = (row.get("config", ""), row.get("seed", ""))
        jid = _job_id_as_int(row.get("job_id", ""))
        prev = latest.get(key)
        if prev is None or jid > _job_id_as_int(prev.get("job_id", "")):
            latest[key] = row
    return list(latest.values())


def has_newer_attempt(rows, config, seed, old_job_id):
    old = _job_id_as_int(old_job_id)
    for r in rows:
        if r["config"] == config and r["seed"] == seed and r.get("job_id"):
            if _job_id_as_int(r["job_id"]) > old:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Auto-watch and repair ablation batch failures.")
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--max-cycles", type=int, default=72)
    parser.add_argument("--status-out", default="jobout/auto_watch_status.txt")
    parser.add_argument("--exclude-nodes", default="", help="Optional comma-separated nodes to exclude on auto-resubmit.")
    args = parser.parse_args()

    status_path = pathlib.Path(args.status_out)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    for cycle in range(1, args.max_cycles + 1):
        stamp = dt.datetime.utcnow().isoformat() + "Z"
        lines = [f"[{stamp}] cycle={cycle}"]
        all_done = True
        for manifest in args.manifests:
            mp = pathlib.Path(manifest)
            rows = read_manifest(mp)
            changed = False
            counts = {"COMPLETED": 0, "RUNNING": 0, "PENDING": 0, "FAILED": 0, "OTHER": 0}
            for r in latest_attempt_rows(rows):
                jid = r.get("job_id", "")
                if not jid:
                    continue
                state = sacct_state(jid)
                if state.startswith("COMPLETED"):
                    counts["COMPLETED"] += 1
                elif state.startswith("RUNNING"):
                    counts["RUNNING"] += 1
                    all_done = False
                elif state.startswith("PENDING"):
                    counts["PENDING"] += 1
                    all_done = False
                elif state.startswith("FAILED") or state.startswith("CANCELLED") or state.startswith("TIMEOUT"):
                    counts["FAILED"] += 1
                    all_done = False
                    if not has_newer_attempt(rows, r["config"], r["seed"], jid):
                        new_jid = resubmit(r["config"], r["seed"], exclude_nodes=args.exclude_nodes)
                        rows.append(
                            {
                                "timestamp_utc": "auto-resubmit",
                                "config": r["config"],
                                "seed": r["seed"],
                                "job_id": new_jid,
                                "dry_run_cmd_or_output": f"auto_resubmitted_from={jid}",
                            }
                        )
                        changed = True
                        lines.append(
                            f"  auto-resubmit manifest={mp.name} config={r['config']} seed={r['seed']} old={jid} new={new_jid}"
                        )
                else:
                    counts["OTHER"] += 1
                    all_done = False
            if changed:
                write_manifest(mp, rows)
            lines.append(
                f"  {mp.name}: completed={counts['COMPLETED']} running={counts['RUNNING']} pending={counts['PENDING']} failed={counts['FAILED']} other={counts['OTHER']}"
            )

        status_path.write_text("\n".join(lines) + "\n")
        if all_done:
            return
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
