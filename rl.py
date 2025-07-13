import gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from model import NEAXNetwork

class GymCompatibilityWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

# === Feature Extractor NEAX ===
class NEAXExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.neax = NEAXNetwork(in_dim=obs_dim, out_dim=features_dim)

    def forward(self, obs):
        return self.neax(obs)

env = make_vec_env(lambda: GymCompatibilityWrapper(gym.make("CartPole-v1")), n_envs=1)

policy_kwargs = dict(
    features_extractor_class=NEAXExtractor,
    features_extractor_kwargs=dict(features_dim=64),
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

model.learn(total_timesteps=500_000)

# === Test ===
test_env = gym.make("CartPole-v1", render_mode="human")
obs, _ = test_env.reset()
total_reward = 0
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    total_reward += reward
    test_env.render()

print(f"\nEpisode as finished - Total reward: {total_reward}")
test_env.close()
