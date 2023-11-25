import torch
from code.MineSweeper.DQN_CNN import DQN_CNN
from code.MineSweeper.MineSweeper_CNN import MineSweeperEnv
import numpy as np
from math import exp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(5)
torch.manual_seed(5)

# MineSweeper Size
width, height = 6, 6
env = MineSweeperEnv(width, height, 0.1)

# Environment parameters
n_actions = width * height
n_states = width * height

# Hyper parameters
n_hidden = 50
batch_size = 64
lr = 0.01
start_epsilon = 0.9
end_epsilon = 0.01
gamma = 0.99  # It's typically less than 1
target_replace_iter = 10000
memory_capacity = 20000
n_episodes = 100000
epsilon_decay = n_episodes/10

# 建立 DQN
dqn = DQN_CNN(n_states, n_actions, n_hidden, batch_size, lr, start_epsilon, gamma, target_replace_iter, memory_capacity, width, height).to(device)

# 學習
isDoneCount = 0
rewards = 0
loss = 0
for i_episode in range(n_episodes):
    t = 0
    state = env.reset()
    state = np.array(state)

    while True:
        action = dqn.choose_action(state)
        next_state, reward, done = env.click(action // width, action % height)

        dqn.store_transition(state, action, reward, next_state)

        rewards += reward

        if dqn.memory_counter > memory_capacity:
            # if dqn.memory_counter % 10 == 0:
            loss = dqn.learn()

        state = np.array(next_state)

        if done:
            new_epsilon = end_epsilon + (start_epsilon - end_epsilon) * exp(-i_episode / epsilon_decay)
            dqn.update_epsilon(new_epsilon)
            if reward == 30:
                isDoneCount += 1
            break
        t += 1

    if i_episode % 100 == 0 and i_episode != 0:
        print("MemoryCounter: {} Episode: {}, Win Rate: {}%, averageRewards:{} Loss: {}"
              .format(dqn.memory_counter,i_episode, (isDoneCount / 100) * 100, rewards/100,loss))
        rewards = 0
        isDoneCount = 0
