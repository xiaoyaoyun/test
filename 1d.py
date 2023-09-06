import torch
import torch.nn as nn
import numpy as np

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

