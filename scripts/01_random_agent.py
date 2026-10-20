"""Phase 2: watch a random agent fail. Use --human for a window (needs a display)."""

import argparse

import gymnasium as gym
import numpy as np

LABELS = ["x", "y", "vx", "vy", "angle", "ang_vel", "left_leg", "right_leg"]

p = argparse.ArgumentParser()
p.add_argument("--human", action="store_true")
p.add_argument("--episodes", type=int, default=20)
args = p.parse_args()

env = gym.make("LunarLander-v3", render_mode="human" if args.human else "rgb_array")
returns = []
for ep in range(args.episodes):
    obs, _ = env.reset(seed=ep)
    if ep == 0:
        print("first observation:", {k: round(float(v), 3) for k, v in zip(LABELS, obs)})
    done, total = False, 0.0
    while not done:
        obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
        total += r
        done = terminated or truncated
    returns.append(total)
print(f"random policy over {args.episodes} episodes: mean return {np.mean(returns):.1f} "
      f"(min {np.min(returns):.1f}, max {np.max(returns):.1f})")
