import numpy as np
import matplotlib.pyplot as plt

# Define the factors
factors_1 = ['red', 'blue', 'green']
factors_2 = ['circle', 'square', 'triangle']

# Define the Q-table
num_factors_1 = len(factors_1)
num_factors_2 = len(factors_2)
num_agents = 2
q_table = np.zeros((num_factors_1, num_factors_2, num_agents, 4))

# Define the hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Define the actions
actions = ['up', 'down', 'left', 'right']

# Define the agent's initial positions
agent_positions = [(0, 0), (num_factors_1 - 1, num_factors_2 - 1)]

# Define the play parameters
num_plays = 10
max_steps = 100

# Track the positions of the agents after each play
play_positions = []

# Track the cumulative rewards for each play
cumulative_rewards = []

# Track the accuracy of the winning agent
winning_agent_accuracy = []

# Play loop
for play in range(num_plays):
    print(f"Play {play + 1}/{num_plays}")

    # Reset the agent positions
    agent_positions = [(0, 0), (num_factors_1 - 1, num_factors_2 - 1)]

    # Track the rewards for each step in the play
    step_rewards = []

    for step in range(max_steps):
        # Select an action for each agent
        for agent_id in range(num_agents):
            position = agent_positions[agent_id]
            x, y = position
            if np.random.uniform() < epsilon:
                # Explore: randomly select an action
                action = np.random.choice(actions)
            else:
                # Exploit: select the action with the maximum Q-value
                action = actions[np.argmax(q_table[x, y, agent_id])]

            # Move the agent
            if action == 'up' and x > 0:
                x -= 1
            elif action == 'down' and x < num_factors_1 - 1:
                x += 1
            elif action == 'left' and y > 0:
                y -= 1
            elif action == 'right' and y < num_factors_2 - 1:
                y += 1

            # Update the agent's position
            agent_positions[agent_id] = (x, y)

        # Update the Q-values
        new_positions = agent_positions.copy()
        for agent_id in range(num_agents):
            x, y = new_positions[agent_id]
            reward = 0  # Define the reward based on the factors and other conditions

            # Update the Q-value using the Bellman equation
            q_value = q_table[x, y, agent_id, actions.index(action)]
            max_q_value = np.max(q_table[x, y, agent_id])
            new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * max_q_value)
            q_table[x, y, agent_id, actions.index(action)] = new_q_value

            # Check if the play is finished
            if (x, y) == (num_factors_1 - 1, num_factors_2 - 1):
                break

        if (x, y) == (num_factors_1 - 1, num_factors_2 - 1):
            print("Play finished!")
            break

        # Calculate the cumulative reward for the step
        step_reward = np.sum(
            [q_table[x, y, agent_id, actions.index(action)] for agent_id, (x, y) in enumerate(agent_positions)])
        step_rewards.append(step_reward)

    # Add the agent positions to the list
    play_positions.append(agent_positions)

    # Calculate the cumulative reward for the play
    cumulative_reward = np.sum(step_rewards)
    cumulative_rewards.append(cumulative_reward)

    # Calculate the accuracy of the winning agent
    winning_agent_id = 0 if agent_positions[0] == (num_factors_1 - 1, num_factors_2 - 1) else 1
    winning_agent_accuracy.append(step_rewards[-1] / (max_steps * num_agents))

# Plot the cumulative rewards
plt.plot(range(1, num_plays + 1), cumulative_rewards)
plt.xlabel('Play')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward per Play')
plt.show()

# Determine the winning agent based on accuracy
winning_agent_index = np.argmax(winning_agent_accuracy)
winning_agent = f"Agent {winning_agent_index + 1}"
print(
    f"The winning agent is {winning_agent} with an accuracy of {winning_agent_accuracy[winning_agent_index] * 100:.2f}%.")

print("Training complete!")
