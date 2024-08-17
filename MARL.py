import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_rl.ddpg import DDPGAgent
from keras_rl.sac import SACAgent
from keras_rl.multi_agent import MultiAgent
from keras_rl.memory import SequentialMemory
from keras_rl.callbacks import ModelIntervalCheckpoint, FileLogger
from sklearn.preprocessing import StandardScaler


class MicrobiomeEnv:
    def __init__(self, num_agents, num_species, data_file):
        self.num_agents = num_agents
        self.num_species = num_species
        self.scaler = StandardScaler()
        self.data = pd.read_csv(data_file)
        self.reset()

    def reset(self):
        self.state = self.scaler.fit_transform(self.data.iloc[:, :self.num_species * (self.num_agents - 1)].values)
        self.action = self.scaler.fit_transform(self.data.iloc[:, self.num_species * (self.num_agents - 1):].values)
        self.rewards = np.zeros(self.num_agents)
        self.done = False
        self.step_count = 0
        return self.state

    def step(self, actions):
        self.rewards = np.zeros(self.num_agents)
        self.done = False
        self.step_count += 1
        for i in range(self.num_agents):
            joint_actions = np.hstack((actions[:i], self.action[i], actions[i:]))
            joint_state = np.hstack((self.state[:i], self.state[i + 1:])).flatten()
            joint_corr = np.corrcoef(joint_actions, joint_state.reshape((self.num_agents - 1) * self.num_species))
            self.rewards[i] = np.sum(joint_corr[:self.num_species, self.num_species:]) / (self.num_agents - 1)
        if self.step_count >= 100:
            self.done = True
        return self.state, self.rewards, self.done, {}

def build_ddpg_agent(state_size, action_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(state_size,)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(action_size, activation='tanh'))
    model.add(tf.keras.layers.Lambda(lambda x: x * 2))
    model.add(tf.keras.layers.Lambda(lambda x: x + 1))
    memory = SequentialMemory(limit=10000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=action_size)
    ddpg_agent = DDPGAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup_critic=100,
                           nb_steps_warmup_actor=100, random_process=random_process, gamma=0.99, target_model_update=1e-3)
    ddpg_agent.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['mae'])
    return ddpg_agent


def build_sac_agent(state_size, action_size):
    model = build_actor_critic(state_size, action_size)
    memory = SequentialMemory(limit=10000, window_length=1)
    sac_agent = SACAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup_critic=100,
                          nb_steps_warmup_actor=100, actor_lr=1e-3, critic_lr=1e-3)
    sac_agent.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['mae'])

    # Train the agent
    history = sac_agent.fit(env, nb_steps=10000, visualize=False, verbose=1)

    # Plot the performance graph
    plt.plot(history.history['episode_reward'])
    plt.title('Performance')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Evaluate the agent
    scores = sac_agent.test(env, nb_episodes=10, visualize=False)
    print(np.mean(scores.history['episode_reward']))
