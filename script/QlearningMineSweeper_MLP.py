import torch
from code.MineSweeper.DQN import DQN
from code.MineSweeper.MineSweeper import MineSweeperEnv
import numpy as np
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(5)
torch.manual_seed(5)

# MineSweeper Size
width = 7
height = 7


env = MineSweeperEnv(width, height, 0.1)

# Environment parameters
n_actions = width * height
n_states = width * height
# print(env.reset(), n_actions)


# Hyper parameters
n_hidden = 50
batch_size = 1000
lr = 0.001                 # learning rate
epsilon = 0.2             # epsilon-greedy
gamma = 1               # reward discount factor
target_replace_iter = 1000 # target network 更新間隔
memory_capacity = 1000
n_episodes = 20000

# 建立 DQN
dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

# 學習
isDoneCount = 0
for i_episode in range(n_episodes):
    if i_episode % 1000 == 0 and i_episode != 0:
        print("Win rate:", str(isDoneCount/10)+"%")
        isDoneCount = 0
    t = 0
    rewards = 0
    state = env.reset()

    while True:
        # 選擇 action
        # print(state)
        state = np.array(state)
        action = dqn.choose_action(state)
        next_state, reward, done = env.click(action//width, action % height)

        # 儲存 experience
        dqn.store_transition(state, action, reward, next_state)

        # 累積 reward
        rewards += reward

        # 有足夠 experience 後進行訓練
        if dqn.memory_counter > memory_capacity:
            dqn.learn()

        # 進入下一 state
        state = next_state

        if done:
            if reward == 1:
                isDoneCount += 1
            # print('Episode {} :{} timesteps, total rewards:{}'.format(i_episode,t+1, rewards))
            break
        t += 1
