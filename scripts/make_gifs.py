"""Record a GIF + reward-breakdown JSON for each variant (seed 0).

    python scripts/make_gifs.py [--variants baseline hover_hack delta_fix] [--seed 0]
"""

import argparse
from pathlib import Path

from rewardlab.envs import load_variants
from rewardlab.record import record
from rewardlab.train import run_dir_for

p = argparse.ArgumentParser()
p.add_argument("--variants", nargs="*", default=list(load_variants()))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval-seed", type=int, default=10000)
a = p.parse_args()

for v in a.variants:
    run = run_dir_for(v, a.seed)
    if not (run / "model.zip").exists():
        print(f"skip {v}: no model")
        continue
    out = Path("results/gifs") / f"{v}.gif"
    print(v, record(str(run), v, str(out), a.eval_seed))
