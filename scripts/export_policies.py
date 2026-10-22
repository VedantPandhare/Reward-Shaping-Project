"""Export trained PPO policies (seed 0) as JSON for the in-browser 3D game, plus parity fixtures.

    python scripts/export_policies.py [--seed 0]
Writes web/results/ (ablation table + plots), web/models/policies.json (weights) and web/models/parity.json (observations with the
action SB3 picks, used by web/tests/parity.mjs to check the JS forward pass).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rewardlab.envs import load_variants, make_env
from rewardlab.train import run_dir_for

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

out = Path("web/models")
out.mkdir(parents=True, exist_ok=True)
policies, parity = {}, {}


def arr(t):
    return np.round(t.detach().cpu().numpy(), 6).tolist()


for variant in load_variants():
    model = PPO.load(run_dir_for(variant, a.seed) / "model.zip", device="cpu")
    sd = model.policy.state_dict()
    policies[variant] = {
        "w1": arr(sd["mlp_extractor.policy_net.0.weight"]), "b1": arr(sd["mlp_extractor.policy_net.0.bias"]),
        "w2": arr(sd["mlp_extractor.policy_net.2.weight"]), "b2": arr(sd["mlp_extractor.policy_net.2.bias"]),
        "w3": arr(sd["action_net.weight"]), "b3": arr(sd["action_net.bias"]),
    }
    env = make_env(variant, wrapped=False)
    obs, _ = env.reset(seed=123)
    samples = []
    for _ in range(300):
        act = int(model.predict(obs, deterministic=True)[0])
        samples.append({"obs": [float(x) for x in obs], "action": act})
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            obs, _ = env.reset()
    parity[variant] = samples[::6]
    print(variant, "exported")

(out / "policies.json").write_text(json.dumps(policies, separators=(",", ":")))
(out / "parity.json").write_text(json.dumps(parity, separators=(",", ":")))
res = Path("web/results")
res.mkdir(parents=True, exist_ok=True)
pd.read_csv("results/ablation.csv").to_json(res / "ablation.json", orient="records", indent=1)
for name in ("learning_curves.png", "tradeoff.png"):
    shutil.copy(Path("results") / name, res / name)
print("policies.json bytes:", (out / "policies.json").stat().st_size)
