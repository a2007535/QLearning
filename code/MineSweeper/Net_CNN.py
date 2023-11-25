import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_channels, n_actions, n_hidden, input_size=(6, 6)):
        self.input_size = input_size
        super(Net, self).__init__()
        def conv(in_channels,out_channels,pool):
            layers = [nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1).cuda(),
                      nn.BatchNorm2d(out_channels).cuda(),
                      nn.ReLU(inplace=True).cuda()]
            if pool:
                layers.append(nn.MaxPool2d(2).cuda())
            return nn.Sequential(*layers)

        self.conv1 = conv(1,32,False)
        self.conv2 = conv(32,64, False)
        self.res1 = nn.Sequential(conv(64, 64, False), conv(64, 64, False))
        self.conv3 = conv(64, 128, True)
        self.res2 = nn.Sequential(conv(128, 128, False), conv(128, 128, False))
        self.conv4 = conv(128, 64, True)
        # 計算全連接層輸入維度
        self.fc_input_dim = self.calculate_fc_input_dim(input_size)

        self.fc1 = nn.Linear(self.fc_input_dim, n_hidden)
        self.activation_func_3 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def calculate_fc_input_dim(self, input_size):
        # 經過兩個卷積層和池化層後的尺寸計算
        size = input_size
        size = (size[0] - 2 + 2) // 2  # 第一次卷積和池化
        size = (size - 2 + 2) // 2  # 第二次卷積和池化
        return size * size * 64  # 64是最後一個卷積層的輸出通道數

    def forward(self, x):
        x = x.view(-1, 1, self.input_size[0], self.input_size[1])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.res2(x) + x
        x = self.conv4(x)
        x = x.view(-1, self.fc_input_dim)  # Flatten the tensor for FC layer
        x = self.activation_func_3(self.fc1(x))
        x = self.fc2(x)
        # print(x)

        return x
