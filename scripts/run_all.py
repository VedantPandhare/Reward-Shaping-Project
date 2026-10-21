"""Parallel sweep over variants x seeds. Each run is its own process (`rewardlab.train`).

    python scripts/run_all.py                      # every variant, seeds from base.yaml
    python scripts/run_all.py --variants baseline hover_hack delta_fix --workers 9
    python scripts/run_all.py --steps 20000        # smoke test
Finished runs (run.json present) are skipped unless --force.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rewardlab.envs import load_base, load_variants
from rewardlab.train import run_dir_for

p = argparse.ArgumentParser()
p.add_argument("--variants", nargs="*", default=None)
p.add_argument("--seeds", nargs="*", type=int, default=None)
p.add_argument("--algo", default="ppo")
p.add_argument("--steps", type=int, default=None)
p.add_argument("--workers", type=int, default=os.cpu_count())
p.add_argument("--force", action="store_true")
p.add_argument("--smoke", action="store_true", help="write to runs/_smoke instead of runs/")
args = p.parse_args()

base = load_base()
variants = args.variants or list(load_variants())
seeds = args.seeds if args.seeds is not None else base["seeds"]
jobs = []
for v in variants:
    for s in seeds:
        out = Path("runs/_smoke") / v / f"seed{s}" if args.smoke else run_dir_for(v, s, args.algo)
        if (out / "run.json").exists() and not args.force:
            print(f"skip {v} seed{s} (done)")
            continue
        cmd = [sys.executable, "-m", "rewardlab.train", "--variant", v, "--seed", str(s),
               "--algo", args.algo, "--out", str(out)]
        if args.steps:
            cmd += ["--steps", str(args.steps)]
        jobs.append((v, s, cmd))

print(f"{len(jobs)} runs, {args.workers} workers")


def run(job):
    v, s, cmd = job
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return v, s, r.returncode, r.stderr[-500:] if r.returncode else ""


with ThreadPoolExecutor(args.workers) as ex:
    for fut in as_completed([ex.submit(run, j) for j in jobs]):
        v, s, code, err = fut.result()
        print(f"{'ok  ' if code == 0 else 'FAIL'} {v} seed{s} {err}", flush=True)
