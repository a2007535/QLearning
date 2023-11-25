import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        # 輸入層 (state) 到隱藏層，隱藏層到輸出層 (action)
        self.fc1 = nn.Linear(n_states, 200).cuda()
        self.fc2 = nn.Linear(200, 200).cuda()
        self.fc3 = nn.Linear(200, 200).cuda()
        self.fc4 = nn.Linear(200, 200).cuda()
        self.fc5 = nn.Linear(200, 200).cuda()
        self.out = nn.Linear(200, n_actions).cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        x = self.fc2(x)
        x = F.relu(x)  # ReLU activation
        x = self.fc3(x)
        x = F.relu(x) # ReLU activation
        x = self.fc4(x)
        x = F.relu(x) # ReLU activation
        x = self.fc5(x)
        x = F.softmax(input=x, dim=1) # ReLU activation
        actions_value = self.out(x)
        return actions_value