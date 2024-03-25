'''
Implementing Q-Learning algorithm for the CartPole-v1 environment from OpenAI Gym.
'''
import gym 
import numpy as np
import matplotlib.pyplot as plt
import random, math

env = gym.make('CartPole-v1', render_mode='human')
bins = list(zip(env.observation_space.low, env.observation_space.high))
bins[1] = (-0.5, 0.5)
bins[3] = (-math.radians(50), math.radians(50))
n_bins = (1,1,6,3)
Q = np.zeros(n_bins + (env.action_space.n,))
alpha = 0.1
max_episodes = 1000
gamma = 0.99
max_time_steps = 250
epsilon = 0.1
n_streaks = 0
solved_time = 199
streak_to_end = 120




def discretize(state):
    indices = []
    for i in range(len(state)):
        if state[i] <= bins[i][0]:
            index = 0
        elif state[i] >= bins[i][1]:
            index = n_bins[i] - 1
        else:
            bound_width = bins[i][1] - bins[i][0]
            offset = (n_bins[i]-1) * bins[i][0] / bound_width
            scaling = (n_bins[i]-1) / bound_width
            index = int(round(scaling * state[i] - offset))
        indices.append(index)
    return tuple(indices)

def epsilon_greedy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])
    

def train():
    for episode in range(max_episodes):
        observation = env.reset()
        # print("From the outer for loop",observation)
        start_state = discretize(observation[0])
        prev_state = start_state
        done = False
        time_step = 0

        while not done:
            env.render()
            action = epsilon_greedy(Q, prev_state, epsilon)
            observation, reward, done, info, _ = env.step(action)
            # print("From the inner while loop",observation)
            best_q = np.max(Q[discretize(observation)])
            Q[prev_state][action] += alpha * (reward + gamma * best_q - Q[prev_state][action])
            state = discretize(observation)
            prev_state = state
            time_step += 1

        if time_step >= solved_time:
            n_streaks += 1
        else:
            n_streaks = 0

        if n_streaks > streak_to_end:
            print(f'Solved in {episode} episodes')
            break

    env.close()

train()

    
