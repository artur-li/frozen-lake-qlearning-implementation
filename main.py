import gym
import numpy as np
import random

# initialize environment
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)

# initialize Q-table
qtable = np.zeros((16, 4))
print("Q-table before training: ")
print(qtable)

# training paramaters
iterations = 1000
learning_rate = 0.5
discount_factor = 0.9
epsilon = 1
epsilon_decay = 0.001

# Training
for _ in range(iterations):
    state, info = env.reset()
    done = False

    # until the agent gets stuck or reaches the goal, keep training it
    while not done:

        # implement exploration over exploitation logic (based on epsilon value)
        random_int = np.random.random()
        if random_int < epsilon:
          action = env.action_space.sample()
        else:
          action = np.argmax(qtable[state])
    
        # make selected move
        next_state, reward, done, truncated, info = env.step(action)

        # execute (update value) formula
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_factor * np.max(qtable[next_state]) - qtable[state, action])

        # update current state
        state = next_state
        
    # decay epsilon
    epsilon = max(epsilon - epsilon_decay, 0)

print()
print("Q-table after training")
print(qtable)


eval_episodes = 100
nb_success = 0

# Evaluation
for _ in range(eval_episodes):
    state, info = env.reset()
    done = False
    
    # until the agent gets stuck or reaches the goal, keep training it
    while not done:
        # choose the action with the highest value in the current state
        action = np.argmax(qtable[state])

        # implement this action
        next_state, reward, done, truncated, info = env.step(action)

        # update current state
        state = next_state

        # if reward attained, it means agent solved the game
        nb_success += reward

# check the success rate
print (f"Success rate after evaluation = {nb_success/eval_episodes*100}%")

