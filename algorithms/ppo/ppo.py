'''
This is a simple example of using the PPO algorithm to solve the CartPole-v1 environment from OpenAI Gym.
'''

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode='human')
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

print(evaluate_policy(model, env, n_eval_episodes=10))

env.close()