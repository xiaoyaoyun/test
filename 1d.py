import torch
import torch.nn as nn
import numpy as np


from torch.utils.data import DataLoader, TensorDataset

X_continuous = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
X_categorical = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

dataset = TensorDataset(X_continuous, X_categorical, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class MyModel(nn.Module):
    def __init__(self, num_continuous_features, num_categorical_features):
        super(MyModel, self).__init__()
        self.fc_continuous = nn.Linear(num_continuous_features, 1)
        self.fc_categorical = nn.Linear(num_categorical_features, 1)
        self.fc_combined = nn.Linear(2, 1)  # 合并连续和离散特征

    def forward(self, x_continuous, x_categorical):
        out_continuous = self.fc_continuous(x_continuous)
        out_categorical = self.fc_categorical(x_categorical)
        combined = torch.cat((out_continuous, out_categorical), dim=1)
        output = self.fc_combined(combined)
        return output
model = MyModel(num_continuous_features=1, num_categorical_features=3)


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 10
output_size = 1
hidden_size = 64

model = SimpleNN(input_size, hidden_size, output_size)

criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

x = torch.randn(1, input_size)  # 输入数据，1个样本，每个样本有10个特征
y = torch.randn(1, output_size)  # 目标数据，1个样本，1个输出值

outputs = model(x)

loss = criterion(outputs, y)

optimizer.zero_grad()  # 清除梯度
loss.backward()  # 反向传播
optimizer.step()  # 更新权重

print(f'损失：{loss.item()}')


def normalize_feature(feature):
    min_val = np.min(feature)
    max_val = np.max(feature)
    normalized_feature = (feature - min_val) / (max_val - min_val)
    return normalized_feature

raw_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]])
normalized_data = np.apply_along_axis(normalize_feature, axis=1, arr=raw_data)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

input_size = 10
output_size = 1

hidden_size = 64
num_layers = 2

model = SimpleLSTM(input_size, hidden_size, output_size, num_layers)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

x = torch.randn(1, 1, input_size) 
y = torch.randn(1, output_size)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=10)

print(model)
