"""Evaluate a saved model. Scores come from the unmodified env reward (wrapper OFF for scoring).

The reward wrapper never alters observations, termination or truncation (see tests), so we can
run it purely as an auditor to collect custom-term totals for the hacking detector while the
reported return is `info["raw_reward"]`, i.e. identical to running the raw env.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO, SAC

from rewardlab.detector import HackingDetector
from rewardlab.envs import load_base, make_env

ALGOS = {"ppo": PPO, "dqn": DQN, "sac": SAC}
X_PAD = 0.2  # |x| of the final obs within this counts as "on the pad"


def fuel_cost(actions) -> float:
    a = np.asarray(actions)
    return float(0.3 * np.sum(a == 2) + 0.03 * np.sum((a == 1) | (a == 3)))


def run_episode(predict, env, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    done, ret, actions, custom, abs_total, last_r, terminated, truncated = (
        False, 0.0, [], 0.0, 0.0, 0.0, False, False)
    while not done:
        action = predict(obs)
        obs, r, terminated, truncated, info = env.step(action)
        raw = info.get("raw_reward", r)
        ret += raw
        last_r = raw
        actions.append(int(action) if np.ndim(action) == 0 else action)
        terms = info.get("reward_terms", {})
        custom += sum(v for k, v in terms.items() if k != "env")
        abs_total += abs(r)
        done = terminated or truncated
    landed = bool(terminated and last_r >= 99.9)
    return {
        "return": ret, "length": len(actions), "terminated": bool(terminated),
        "truncated": bool(truncated), "landed": landed,
        "crashed": bool(terminated and last_r <= -99.9),
        "on_pad": bool(landed and abs(obs[0]) <= X_PAD),
        "fuel": fuel_cost(actions) if actions and np.ndim(actions[0]) == 0 else float("nan"),
        "custom_sum": custom, "abs_total": abs_total,
    }


def evaluate_model(model, variant: str, n_episodes: int = 100, seed_base: int = 10000,
                   deterministic: bool = True, gamma: float = 0.999, **env_kwargs) -> pd.DataFrame:
    env = make_env(variant, wrapped=True, gamma=gamma, **env_kwargs)

    def predict(obs):
        return model.predict(obs, deterministic=deterministic)[0]

    rows = [{"episode": i, "seed": seed_base + i, **run_episode(predict, env, seed_base + i)}
            for i in range(n_episodes)]
    env.close()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, detector_kwargs: dict | None = None) -> dict:
    d = df.rename(columns={"return": "raw_return"}).assign(shaped_return=0.0)
    det = HackingDetector.analyze(d, **(detector_kwargs or {}))
    return {
        "mean_return": float(df["return"].mean()), "std_return": float(df["return"].std()),
        "landing_rate": float(df["landed"].mean()), "on_pad_rate": float(df["on_pad"].mean()),
        "crash_rate": float(df["crashed"].mean()), "timeout_rate": float(df["truncated"].mean()),
        "mean_length": float(df["length"].mean()), "mean_fuel": float(df["fuel"].mean()),
        "custom_share": det["custom_share"], "hack_flag": bool(det["flagged"]),
    }


def load_model(path: str, algo: str = "ppo"):
    return ALGOS[algo].load(path, device="cpu")


def evaluate_run(run_dir: Path, variant: str, algo: str, n_episodes: int, seed_base: int,
                 gamma: float, detector_kwargs: dict | None = None) -> dict:
    model = load_model(str(run_dir / "model.zip"), algo)
    df = evaluate_model(model, variant, n_episodes, seed_base, gamma=gamma)
    df.to_csv(run_dir / "eval_episodes.csv", index=False)
    summary = summarize(df, detector_kwargs)
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    base = load_base()
    p = argparse.ArgumentParser(description="Evaluate a saved model on the raw env")
    p.add_argument("run_dir")
    p.add_argument("--variant", required=True, help="variant the model was trained with (audit)")
    p.add_argument("--algo", default=base["algo"])
    p.add_argument("--episodes", type=int, default=base["eval_episodes"])
    a = p.parse_args()
    print(json.dumps(evaluate_run(Path(a.run_dir), a.variant, a.algo, a.episodes,
                                  base["eval_seed_base"], base["ppo"]["gamma"],
                                  {k: v for k, v in base["detector"].items() if k != "window"}),
                     indent=2))


if __name__ == "__main__":
    main()
