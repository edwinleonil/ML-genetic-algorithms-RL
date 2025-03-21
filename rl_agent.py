import numpy as np
import random

# Define the environment
grid_size = 4
num_states = grid_size * grid_size
num_actions = 4  # Up, Down, Left, Right

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# Rewards matrix
rewards = np.full((grid_size, grid_size), -1)
rewards[3, 3] = 10  # Goal state with reward

# Parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2
num_episodes = 1000

# Q-table initialization
Q = np.zeros((num_states, num_actions))


def state_to_index(state):
    return state[0] * grid_size + state[1]


def move(state, action):
    i, j = state
    if action == UP:
        i = max(i - 1, 0)
    elif action == DOWN:
        i = min(i + 1, grid_size - 1)
    elif action == LEFT:
        j = max(j - 1, 0)
    elif action == RIGHT:
        j = min(j + 1, grid_size - 1)
    return (i, j)


# Training with Q-learning
for episode in range(num_episodes):
    state = (0, 0)  # Start state

    while state != (3, 3):  # Until reaching the goal
        state_idx = state_to_index(state)

        # Exploration vs Exploitation
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_actions - 1)  # Explore
        else:
            action = np.argmax(Q[state_idx])  # Exploit

        next_state = move(state, action)
        next_state_idx = state_to_index(next_state)

        reward = rewards[next_state]

        # Q-learning update rule
        Q[state_idx, action] += learning_rate * (
            reward + discount_factor *
            np.max(Q[next_state_idx]) - Q[state_idx, action]
        )

        state = next_state

# Print the learned Q-values
print("Trained Q-values:")
print(Q.reshape(grid_size, grid_size, num_actions))
