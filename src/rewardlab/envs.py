"""Environment construction helpers."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rewardlab.wrappers import RewardShapingWrapper, make_reward_fn

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_base() -> dict:
    return load_yaml("base.yaml")


def load_variants() -> dict:
    return load_yaml("variants.yaml")


def variant_cfg(variant: str) -> dict:
    variants = load_variants()
    if variant not in variants:
        raise KeyError(f"unknown variant {variant!r}; choose from {sorted(variants)}")
    return variants[variant]


def make_env(variant: str | None = "baseline", *, wrapped: bool = True, gamma: float = 0.999,
             env_id: str = "LunarLander-v3", render_mode: str | None = None, **env_kwargs):
    """One env. `wrapped=False` gives the raw env (used for evaluation)."""
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    if wrapped and variant:
        cfg = variant_cfg(variant)
        env = RewardShapingWrapper(env, make_reward_fn(cfg["fn"], cfg.get("params"), gamma=gamma))
    return env


def make_train_vec_env(variant: str, seed: int, n_envs: int = 16, gamma: float = 0.999,
                       monitor_dir: str | None = None, env_id: str = "LunarLander-v3"):
    """Vectorised training env. Monitor sits outside the wrapper, so ep_rew_mean is the *shaped*
    return (what the agent optimises); the raw return is in info["raw_reward"]."""
    def factory(i):
        def _init():
            env = make_env(variant, wrapped=True, gamma=gamma, env_id=env_id)
            path = None if monitor_dir is None else str(Path(monitor_dir) / f"monitor_{i}")
            return Monitor(env, path)
        return _init

    venv = DummyVecEnv([factory(i) for i in range(n_envs)])
    venv.seed(seed)
    return venv
