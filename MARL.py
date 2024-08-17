import numpy as np
import random

# Define the environment
class CellEnvironment:
    def __init__(self, data):
        self.data = data
        self.n_cells, self.n_genes = data.shape
        self.n_clusters = 2

    def reset(self):
        self.cell_clusters = np.zeros(self.n_cells, dtype=int)
        return self.cell_clusters

    def step(self, actions):
        for i in range(self.n_cells):
            self.cell_clusters[i] = actions[i]
        rewards = self.calculate_rewards()
        return self.cell_clusters, rewards, False

    def calculate_rewards(self):
        rewards = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            cluster = self.cell_clusters[i]
            rewards[i] = -np.mean((self.data[i] - np.mean(self.data[self.cell_clusters == cluster], axis=0))**2)
        return rewards

# Define the agents
class CellAgent:
    def __init__(self, env, agent_idx):
        self.env = env
        self.idx = agent_idx
        self.action_space = np.arange(env.n_clusters)
        self.state = env.reset()

    def act(self, state, epsilon=0.1):
        if random.uniform(0, 1) < epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.greedy_action(state)
        return action

    def greedy_action(self, state):
        rewards = []
        for action in self.action_space:
            new_state = state.copy()
            new_state[self.idx] = action
            rewards.append(np.mean(self.env.calculate_rewards()[new_state == action]))
        return np.argmax(rewards)

# Run the simulation
def run_simulation(env, agents, n_steps=100):
    for i in range(n_steps):
        actions = [agent.act(env.cell_clusters) for agent in agents]
        env.step(actions)
    return env.cell_clusters

# Load the data
data = np.random.randn(100, 500)

# Initialize the environment and agents
env = CellEnvironment(data)
agents = [CellAgent(env, i) for i in range(env.n_cells)]

# Run the simulation
cell_clusters = run_simulation(env, agents)

# Print the results
print(cell_clusters)
