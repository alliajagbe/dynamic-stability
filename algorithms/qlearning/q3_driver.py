import gym
import numpy as np
import matplotlib.pyplot as plt 
from qlearning3 import Q_Learning
 
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
Q1.train()

cumulative_rewards,env=Q1.test()
 
plt.figure(figsize=(12, 5))
plt.plot(Q1.total_rewards,color='blue',linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per episode from Q-Learning Algorithm')
plt.savefig('rewards.png')
plt.show()
 
 
env.close()

# rolling average
windowSize=100
rollingAverage=np.convolve(Q1.total_rewards, np.ones(windowSize)/windowSize, mode='valid')
plt.figure(figsize=(12, 5))
plt.plot(rollingAverage,color='blue',linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rolling average of rewards per episode from Q-Learning Algorithm')
plt.savefig('rolling_average.png')
plt.show()