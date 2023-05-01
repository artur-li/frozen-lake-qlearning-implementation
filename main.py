import gym
import numpy as np

# initializing environment
env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)

# initializing Q-table
qtable = np.zeros((16, 4))
print("Q-table before training: ")
print(qtable)

# initializing paramaters
iterations = 1000
learning_rate = 0.5
discount_factor = 0.9
number_of_completions = 0

# Training
for _ in range(iterations):
    state, info = env.reset()
    done = False

    # keep training until iteration terminated
    while not done:

        # make highest yielding move if applicable, else: random move
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])  
        else:
            action = env.action_space.sample()
    
        # actually make the move
        next_state, reward, done, truncated, info = env.step(action)

        # execute (update value) formula
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_factor * np.max(qtable[next_state]) - qtable[state, action])

        # update state
        state = next_state

        # if completed add 1 to completions
        if reward == 1:
            number_of_completions += 1

print()
print("Q-table after training")
print(qtable)
print()
print(f"Number of completions: {number_of_completions} (after 1000 iterations)")
