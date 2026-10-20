"""Phase 3/4: train the baseline PPO on the built-in reward, evaluate on the raw env, record a GIF.

    python scripts/02_train_baseline.py [--steps 1000000] [--seed 0]
Watch training with: tensorboard --logdir runs
"""

import argparse
import json

from rewardlab.record import record
from rewardlab.train import run_dir_for, train

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=1_000_000)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

info = train("baseline", a.seed, a.steps)
print(json.dumps(info, indent=2))
print(record(str(run_dir_for("baseline", a.seed)), "baseline", "results/gifs/baseline.gif"))
