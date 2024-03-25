import gym
import numpy as np

# Initialize the CartPole environment
env = gym.make('CartPole-v1')

# Discretize the state space
num_buckets = (1, 1, 6, 3)  # (position, velocity, angle, angular velocity)
num_actions = env.action_space.n
Q_table = np.zeros(num_buckets + (num_actions,))

# Define the Q-learning parameters
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
exploration_decay_rate = 0.99
min_exploration_rate = 0.01

# Discretize the state
def discretize_state(state):
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -np.radians(50)]
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], np.radians(50)]
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    discrete_state = [int(round((num_buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    discrete_state = [min(num_buckets[i] - 1, max(0, discrete_state[i])) for i in range(len(state))]
    return tuple(discrete_state)

# Choose action using epsilon-greedy policy
def choose_action(state, exploration_rate):
    if np.random.random() < exploration_rate:
        return env.action_space.sample()  # Explore action space
    else:
        return np.argmax(Q_table[state])  # Exploit learned values

# Update Q-values using Q-learning update rule
def update_Q_value(state, action, reward, next_state, next_action):
    best_next_action = np.argmax(Q_table[next_state])
    target = reward + discount_factor * Q_table[next_state + (best_next_action,)]
    Q_table[state + (action,)] += learning_rate * (target - Q_table[state + (action,)])

# Train the agent
num_episodes = 10000
for episode in range(num_episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay_rate)

    done = False
    total_reward = 0
    while not done:
        action = choose_action(state, exploration_rate)
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        update_Q_value(state, action, reward, next_state, action)
        state = next_state
        total_reward += reward

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Test the trained agent
total_rewards = []
num_test_episodes = 100
for _ in range(num_test_episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(Q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        state = next_state
        total_reward += reward
    total_rewards.append(total_reward)

print(f"Average Total Reward: {np.mean(total_rewards)}")