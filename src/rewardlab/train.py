"""Train one (variant, seed): `python -m rewardlab.train --variant hover_hack --seed 0`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from rewardlab.detector import HackingCallback
from rewardlab.envs import load_base, make_train_vec_env, variant_cfg
from rewardlab.evaluate import evaluate_run

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

DQN_KWARGS = dict(learning_rate=6.3e-4, batch_size=128, buffer_size=50000, learning_starts=0,
                  gamma=0.99, target_update_interval=250, train_freq=4, gradient_steps=-1,
                  exploration_fraction=0.12, exploration_final_eps=0.1,
                  policy_kwargs=dict(net_arch=[256, 256]))


def run_dir_for(variant: str, seed: int, algo: str = "ppo") -> Path:
    tag = "" if algo == "ppo" else f"_{algo}"
    return RUNS_DIR / f"{variant}{tag}" / f"seed{seed}"


def train(variant: str, seed: int, steps: int | None = None, algo: str = "ppo",
          n_envs: int | None = None, evaluate: bool = True, out_dir: Path | None = None) -> dict:
    base = load_base()
    cfg = variant_cfg(variant)
    steps = steps or cfg.get("total_timesteps", base["total_timesteps"])
    n_envs = n_envs or (base["n_envs"] if algo == "ppo" else 1)
    out = Path(out_dir) if out_dir else run_dir_for(variant, seed, algo)
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    np.random.seed(seed)
    gamma = base["ppo"]["gamma"] if algo == "ppo" else DQN_KWARGS["gamma"]
    env = make_train_vec_env(variant, seed, n_envs, gamma=gamma)
    det_cfg = base["detector"]
    callback = HackingCallback(
        episodes_path=out / "episodes.csv", window=det_cfg["window"],
        timeout_rate_threshold=det_cfg["timeout_rate_threshold"],
        custom_share_threshold=det_cfg["custom_share_threshold"])

    if algo == "ppo":
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                    tensorboard_log=str(out / "tb"), **base["ppo"])
    elif algo == "dqn":
        model = DQN("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                    tensorboard_log=str(out / "tb"), **DQN_KWARGS)
    else:
        raise ValueError(f"unsupported algo {algo!r}")

    t0 = time.time()
    model.learn(total_timesteps=steps, callback=callback, tb_log_name="run")
    train_seconds = time.time() - t0
    model.save(out / "model.zip")
    callback.save()
    env.close()

    info = {"variant": variant, "seed": seed, "algo": algo, "steps": steps, "n_envs": n_envs,
            "train_seconds": round(train_seconds, 1), "train_flagged": callback.ever_flagged}
    if evaluate:
        summary = evaluate_run(out, variant, algo, base["eval_episodes"], base["eval_seed_base"],
                               gamma, {k: v for k, v in det_cfg.items() if k != "window"})
        info["eval"] = summary
    (out / "run.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    info = train(a.variant, a.seed, a.steps, a.algo, a.n_envs, not a.no_eval, a.out)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
