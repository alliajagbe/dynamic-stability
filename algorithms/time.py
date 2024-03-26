# comparing the time elapsed by both ppo and qlearning algorithms for 10000 episodes

from qlearning.qlearning3 import Q_Learning
from stable_baselines3 import PPO
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

import timeit

# PPO
env_name = 'CartPole-v1'
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)
start_time_ppo = timeit.default_timer()
model.learn(total_timesteps=10000)
end_time_ppo = timeit.default_timer()

ppo_time = end_time_ppo - start_time_ppo

# Q-Learning
env=gym.make('CartPole-v1')
(state,_)=env.reset()
 
upper_bound=env.observation_space.high
lower_bound=env.observation_space.low
min_cart_velocity=-3
max_cart_velocity=3
min_pole_angle=-10
max_pole_angle=10
upper_bound[1]=max_cart_velocity
upper_bound[3]=max_pole_angle
lower_bound[1]=min_cart_velocity
lower_bound[3]=min_pole_angle

n_bins_position = n_bins_velocity = n_bins_angle = n_bins_angle_velocity = 30

n_bins=[n_bins_position,n_bins_velocity,n_bins_angle,n_bins_angle_velocity]
 
alpha=0.1
gamma=1
epsilon=0.2
n_episodes=10000
 
Q1=Q_Learning(env,alpha,gamma,epsilon,n_episodes,n_bins,lower_bound,upper_bound)
start_time_q = timeit.default_timer()
Q1.train()
end_time_q = timeit.default_timer()

q_time = end_time_q - start_time_q

# visualizing the time elapsed
plt.bar(['PPO', 'Q-Learning'], [ppo_time, q_time], color=['blue', 'red'])
plt.xlabel('Algorithm')
plt.ylabel('Time elapsed')
plt.title('Time elapsed by PPO and Q-Learning for 10000 episodes')
plt.savefig('time_elapsed.png')
plt.show()