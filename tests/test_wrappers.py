import gymnasium as gym
import numpy as np
import pytest

from rewardlab.wrappers import (
    DeltaFix, HoverHack, RewardShapingWrapper, TimePenalty, env_breakdown, make_reward_fn,
)


def rollout(env, actions, seed=0):
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        out.append((obs, r, term, trunc, info))
        if term or trunc:
            break
    return out


ACTIONS = np.random.default_rng(0).integers(0, 4, 300).tolist()


@pytest.mark.parametrize("fn", ["none", "time_penalty", "hover_hack", "delta_fix",
                                "tilt_only_hack", "fuel_efficient", "soft_landing"])
def test_wrapper_never_changes_dynamics(fn):
    raw = rollout(gym.make("LunarLander-v3"), ACTIONS)
    wrapped = rollout(RewardShapingWrapper(gym.make("LunarLander-v3"), make_reward_fn(fn)), ACTIONS)
    assert len(raw) == len(wrapped)
    np.testing.assert_array_equal(raw[0], wrapped[0])
    for (o1, r1, t1, tr1, _), (o2, r2, t2, tr2, info) in zip(raw[1:], wrapped[1:]):
        np.testing.assert_array_equal(o1, o2)
        assert (t1, tr1) == (t2, tr2)
        assert info["raw_reward"] == pytest.approx(r1)
        assert info["reward_terms"]["env"] == pytest.approx(r1)


def test_time_penalty_changes_reward_exactly():
    raw = rollout(gym.make("LunarLander-v3"), ACTIONS)
    w = rollout(RewardShapingWrapper(gym.make("LunarLander-v3"), TimePenalty(0.05)), ACTIONS)
    for (_, r1, *_), (_, r2, *_rest) in zip(raw[1:], w[1:]):
        assert r2 == pytest.approx(r1 - 0.05)


def test_hover_bonus_condition():
    fn = HoverHack(bonus=1.0, max_abs_angle=0.2, max_speed=0.3)
    obs = np.zeros(8)
    assert fn(obs, 0, 0, False, False)[0] == 1.0
    obs[4] = 0.5  # tilted
    assert fn(obs, 0, 0, False, False)[0] == 0.0
    obs[4], obs[3] = 0.0, -0.5  # fast
    assert fn(obs, 0, 0, False, False)[0] == 0.0


def test_delta_fix_telescopes():
    """Without a time penalty and gamma=1, shaping over an episode sums to -scale*Phi(s0)
    (terminal potential 0): there is nothing to farm by hovering."""
    fn = DeltaFix(scale=10.0, time_penalty=0.0, gamma=1.0)
    rng = np.random.default_rng(1)
    obs0 = rng.normal(size=8) * 0.1
    fn.reset(obs0)
    total = 0.0
    for t in range(50):
        obs = rng.normal(size=8) * 0.1
        total += fn(obs, 0, 0, t == 49, False)[0]
    assert total == pytest.approx(-10.0 * (-(abs(obs0[4]) + np.hypot(obs0[2], obs0[3]))))


def test_delta_fix_state_resets_between_episodes():
    env = RewardShapingWrapper(gym.make("LunarLander-v3"), DeltaFix(time_penalty=0.0))
    env.reset(seed=0)
    env.step(0)
    env.reset(seed=1)
    _, _, _, _, info = env.step(0)
    assert abs(info["reward_terms"]["delta_shaping"]) < 5  # no huge jump from stale potential


def test_breakdown_sums_to_env_reward():
    env = gym.make("LunarLander-v3")
    obs, _ = env.reset(seed=3)
    for a in ACTIONS[:200]:
        nobs, r, term, trunc, _ = env.step(a)
        assert sum(env_breakdown(obs, nobs, a, r).values()) == pytest.approx(r, abs=1e-3)
        obs = nobs
        if term or trunc:
            break


def test_unknown_fn():
    with pytest.raises(KeyError):
        make_reward_fn("nope")
