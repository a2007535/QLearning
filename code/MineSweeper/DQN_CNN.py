import numpy as np
import torch.nn as nn
import torch
from code.MineSweeper.Net_CNN import Net


class DQN_CNN(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter,
                 memory_capacity, width, height):
        super(DQN_CNN, self).__init__()
        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden, input_size=(width, height)), Net(n_states, n_actions, n_hidden, input_size=(width, height))

        self.state_memory = np.zeros([memory_capacity,int(n_states**0.5),int(n_states**0.5)])  # 每個 memory 中的 experience 大小為 (state + next state + reward + action)
        self.nextstate_memory = np.zeros([memory_capacity,int(n_states**0.5),int(n_states**0.5)])
        self.action_memory = np.zeros((memory_capacity,1))
        self.reward_memory = np.zeros((memory_capacity, 1))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.CrossEntropyLoss()
        self.memory_counter = 0
        self.learn_step_counter = 0  # 讓 target network 知道什麼時候要更新

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()  # 增加批次維度

        def is_valid_action(state, action):
            row = action // int(self.n_states ** 0.5)
            col = action % int(self.n_states ** 0.5)
            # 確認動作是否有效的邏輯，取決於你的狀態表示
            # 例如，在Minesweeper中，檢查對應的網格是否未被揭示
            # print(state[row, col])
            return state[row, col] == -1
        # epsilon-greedy策略
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
            # print(action, state)
            while not is_valid_action(state, action):
                action = np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                # print(state_tensor)
                actions_value = self.eval_net(state_tensor)
                # print(actions_value.max(1)[1])
                action = actions_value.max(1)[1].cpu().item()  # 選擇價值最高的動作
                while not is_valid_action(state, action):
                    # 將已經探索過的動作價值設為極低，再次選擇
                    actions_value[0][action] = -float('inf')
                    action = actions_value.max(1)[1].cpu().item()

        return action



    def store_transition(self, state, action, reward, next_state):

        # 存進 memory；舊 memory 可能會被覆蓋
        index = self.memory_counter % self.memory_capacity
        self.state_memory[index, :] = state
        self.nextstate_memory[index, :] = next_state
        self.action_memory[index, :] = action
        self.reward_memory[index, :] = reward

        self.memory_counter += 1

    def learn(self):
        # 確保有足夠的記憶體進行抽樣
        if self.memory_counter < self.batch_size:
            return
        sample_index = np.random.choice(min(self.memory_counter, self.memory_capacity), self.batch_size)
        b_state = torch.FloatTensor(self.state_memory[sample_index, :]).cuda()
        b_action = torch.LongTensor(self.action_memory[sample_index, :].astype(int)).cuda()
        b_reward = torch.FloatTensor(self.reward_memory[sample_index, :]).cuda()
        b_next_state = torch.FloatTensor(self.nextstate_memory[sample_index, :]).cuda()

        # DQN的目標Q值和預測Q值
        q_eval = self.eval_net(b_state).gather(1, b_action)
        q_next = self.target_net(b_next_state).detach()
        q_target = b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        # 使用均方誤差MSE作為損失函數
        loss = nn.MSELoss()(q_eval, q_target)


        # 反向傳播和優化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目標網絡
        self.learn_step_counter += 1
        # print(self.learn_step_counter)
        if self.learn_step_counter % self.target_replace_iter == 0:
            print("更新目標網路")
            self.target_net.load_state_dict(self.eval_net.state_dict())
        return loss