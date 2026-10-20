"""rewardlab: reward shaping wrappers and a reward-hacking detector."""

from rewardlab.detector import HackingCallback, HackingDetector
from rewardlab.wrappers import RewardShapingWrapper, make_reward_fn

__all__ = ["HackingCallback", "HackingDetector", "RewardShapingWrapper", "make_reward_fn"]
