import gymnasium as gym
import stable_baselines3
import torch

print("gymnasium", gym.__version__)
print("stable-baselines3", stable_baselines3.__version__)
print("torch", torch.__version__)
env = gym.make("LunarLander-v3")
obs, _ = env.reset(seed=0)
print("LunarLander-v3 ok, obs shape", obs.shape)
