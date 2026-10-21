"""Reward-shaping wrapper and the reward variants used in the Lunar Lander lab."""

from __future__ import annotations

import math
from collections.abc import Callable

import gymnasium as gym

MAIN_ENGINE_COST = 0.3
SIDE_ENGINE_COST = 0.03


class RewardFn:
    """Base class: returns (extra_reward, {term_name: value}) for each step."""

    def reset(self, obs) -> None:
        pass

    def __call__(self, obs, action, reward, terminated, truncated):
        return 0.0, {}


class RewardShapingWrapper(gym.Wrapper):
    """Adds `extra` reward from `reward_fn` without touching obs / termination / truncation.

    `reward_fn(obs, action, reward, terminated, truncated) -> (extra, terms)`; it may expose a
    `reset(obs)` method for per-episode state. The unmodified reward is always available as
    `info["raw_reward"]` and every custom term as `info["reward_terms"]` (plus the key "env").
    """

    def __init__(self, env: gym.Env, reward_fn: Callable):
        super().__init__(env)
        self.reward_fn = reward_fn

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        extra, terms = self.reward_fn(obs, action, reward, terminated, truncated)
        info["reward_terms"] = {"env": float(reward), **{k: float(v) for k, v in terms.items()}}
        info["raw_reward"] = float(reward)
        return obs, reward + extra, terminated, truncated, info


def _speed(obs) -> float:
    return math.hypot(obs[2], obs[3])


def _is_main(action) -> bool:
    return int(action) == 2


def _is_side(action) -> bool:
    return int(action) in (1, 3)


class TimePenalty(RewardFn):
    def __init__(self, penalty: float = 0.05):
        self.penalty = penalty

    def __call__(self, obs, action, reward, terminated, truncated):
        return -self.penalty, {"time_penalty": -self.penalty}


class HoverHack(RewardFn):
    """+bonus every frame the lander is upright and slow. Hovering beats landing."""

    def __init__(self, bonus=1.0, max_abs_angle=0.2, max_speed=0.3):
        self.bonus, self.max_abs_angle, self.max_speed = bonus, max_abs_angle, max_speed

    def __call__(self, obs, action, reward, terminated, truncated):
        ok = abs(obs[4]) < self.max_abs_angle and _speed(obs) < self.max_speed
        b = self.bonus if ok else 0.0
        return b, {"hover_bonus": b}


class TiltOnlyHack(RewardFn):
    def __init__(self, bonus=1.0, max_abs_angle=0.2):
        self.bonus, self.max_abs_angle = bonus, max_abs_angle

    def __call__(self, obs, action, reward, terminated, truncated):
        b = self.bonus if abs(obs[4]) < self.max_abs_angle else 0.0
        return b, {"tilt_bonus": b}


def upright_slow_potential(obs) -> float:
    return -(abs(obs[4]) + _speed(obs))


class DeltaFix(RewardFn):
    """Potential-based shaping scale*(gamma*Phi(s') - Phi(s)) plus an optional time penalty.

    Phi = -(|angle| + speed); Phi of a terminal state is 0 (episodic convention). Truncation is
    not terminal, so Phi is kept.
    """

    def __init__(self, scale=10.0, time_penalty=0.05, gamma=0.999):
        self.scale, self.time_penalty, self.gamma = scale, time_penalty, gamma
        self._prev = 0.0

    def reset(self, obs):
        self._prev = upright_slow_potential(obs)

    def __call__(self, obs, action, reward, terminated, truncated):
        phi = 0.0 if terminated else upright_slow_potential(obs)
        shaping = self.scale * (self.gamma * phi - self._prev)
        self._prev = phi
        terms = {"delta_shaping": shaping}
        if self.time_penalty:
            terms["time_penalty"] = -self.time_penalty
        return sum(terms.values()), terms


class FuelEfficient(RewardFn):
    """Multiplies the built-in fuel cost by `multiplier` (extra = (multiplier-1) * cost)."""

    def __init__(self, multiplier=2.0):
        self.multiplier = multiplier

    def __call__(self, obs, action, reward, terminated, truncated):
        cost = MAIN_ENGINE_COST if _is_main(action) else SIDE_ENGINE_COST if _is_side(action) else 0
        extra = -(self.multiplier - 1.0) * cost
        return extra, {"extra_fuel": extra}


class SoftLanding(RewardFn):
    """Penalty on vertical speed and tilt at the last frame of a (non-out-of-bounds) episode."""

    def __init__(self, scale=10.0):
        self.scale = scale

    def __call__(self, obs, action, reward, terminated, truncated):
        if terminated and abs(obs[0]) < 1.0:
            p = -self.scale * (abs(obs[3]) + abs(obs[4]))
            return p, {"touchdown_penalty": p}
        return 0.0, {}


REWARD_FNS = {
    "none": RewardFn,
    "time_penalty": TimePenalty,
    "hover_hack": HoverHack,
    "delta_fix": DeltaFix,
    "tilt_only_hack": TiltOnlyHack,
    "fuel_efficient": FuelEfficient,
    "soft_landing": SoftLanding,
}


def make_reward_fn(fn: str, params: dict | None = None, **extra) -> RewardFn:
    if fn not in REWARD_FNS:
        raise KeyError(f"unknown reward fn {fn!r}; choose from {sorted(REWARD_FNS)}")
    kwargs = {**(params or {}), **extra}
    if fn != "delta_fix":
        kwargs.pop("gamma", None)
    return REWARD_FNS[fn](**kwargs)


def lunar_shaping(obs) -> dict:
    """Components of LunarLander's built-in position score (matches the env source)."""
    return {
        "distance": -100 * math.hypot(obs[0], obs[1]),
        "speed": -100 * _speed(obs),
        "tilt": -100 * abs(obs[4]),
        "legs": 10 * obs[6] + 10 * obs[7],
    }


def env_breakdown(prev_obs, obs, action, reward) -> dict:
    """Split the built-in per-frame reward into improvement terms (for the demo chart)."""
    a, b = lunar_shaping(prev_obs), lunar_shaping(obs)
    terms = {k: b[k] - a[k] for k in a}
    terms["fuel"] = (
        -MAIN_ENGINE_COST if _is_main(action) else -SIDE_ENGINE_COST if _is_side(action) else 0.0
    )
    terms["terminal"] = float(reward) - sum(terms.values())
    if abs(terms["terminal"]) < 1e-3:
        terms["terminal"] = 0.0
    return terms
