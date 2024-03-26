'''
This is an implementation of the PPO algorithm to solve the CartPole-v1 environment from OpenAI Gym.
'''

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np

env_name = 'CartPole-v1'
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

results = evaluate_policy(model, env, n_eval_episodes=1000, return_episode_rewards=True, warn=False)

per_episode_rewards = results[0]

# plotting the rewards
plt.plot(per_episode_rewards)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards per episode from PPO Algorithm')
plt.savefig('ppo_rewards.png')
plt.show()

# rolling average
windowSize=100
rollingAverage=np.convolve(per_episode_rewards, np.ones(windowSize)/windowSize, mode='valid')
plt.plot(rollingAverage)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rolling average of rewards per episode from PPO Algorithm')
plt.savefig('ppo_rolling_average.png')
plt.show()

env.close()