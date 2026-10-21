"""Render a trained agent to GIF and dump its per-frame reward breakdown.

    python -m rewardlab.record runs/hover_hack/seed0 --variant hover_hack --out results/gifs/hover.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from rewardlab.envs import load_base, make_env
from rewardlab.evaluate import load_model
from rewardlab.wrappers import env_breakdown


def rollout_with_frames(model, variant: str, seed: int, max_steps: int = 1000, gamma: float = 0.999):
    """Run one deterministic episode. Returns (frames, per-step records, summary)."""
    env = make_env(variant, wrapped=True, gamma=gamma, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    steps, ret, fuel, last_r, term, trunc = [], 0.0, 0.0, 0.0, False, False
    for t in range(max_steps):
        action = int(model.predict(obs, deterministic=True)[0])
        nobs, r, term, trunc, info = env.step(action)
        raw = info["raw_reward"]
        terms = env_breakdown(obs, nobs, action, raw)
        custom = {k: v for k, v in info["reward_terms"].items() if k != "env"}
        steps.append({"t": t, "action": action, **{k: float(v) for k, v in terms.items()}, **custom, "raw_reward": raw,
                      "total_reward": float(r)})
        fuel += 0.3 if action == 2 else 0.03 if action in (1, 3) else 0.0
        ret += raw
        last_r = raw
        obs = nobs
        frames.append(env.render())
        if term or trunc:
            break
    env.close()
    summary = {"return": ret, "length": len(steps), "landed": bool(term and last_r >= 99.9),
               "crashed": bool(term and last_r <= -99.9), "timed_out": bool(trunc),
               "fuel": fuel}
    return frames, steps, summary


def save_gif(frames, path, fps: int = 25, stride: int = 2, shrink: int = 2, max_seconds: float = 12):
    frames = [f[::shrink, ::shrink] for f in frames[::stride]]
    frames = frames[: int(fps * max_seconds)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)


def record(run_dir: str, variant: str, out: str, seed: int = 10000, algo: str = "ppo") -> dict:
    model = load_model(str(Path(run_dir) / "model.zip"), algo)
    frames, steps, summary = rollout_with_frames(model, variant, seed,
                                                 gamma=load_base()["ppo"]["gamma"])
    save_gif(frames, out)
    Path(out).with_suffix(".json").write_text(json.dumps({"summary": summary, "steps": steps}))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--variant", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--algo", default="ppo")
    a = p.parse_args()
    print(record(a.run_dir, a.variant, a.out, a.seed, a.algo))


if __name__ == "__main__":
    main()
