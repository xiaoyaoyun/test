import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import Transformer, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

# 自定义数据集类
class CustomDataset(data.Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __getitem__(self, index):
        return self.input_data[index], self.target_data[index]

    def __len__(self):
        return len(self.input_data)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()

        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src = src.permute(1, 0, 2)  # 转换输入形状为(sequence_length, batch_size, input_dim)
        tgt = tgt.permute(1, 0, 2)  # 转换目标形状为(sequence_length, batch_size, output_dim)

        encoder_output = self.transformer_encoder(src)
        decoder_output = self.transformer_decoder(tgt, encoder_output)
        
        decoder_output = decoder_output.permute(1, 0, 2)  # 恢复输出形状为(batch_size, sequence_length, d_model)
        output = self.fc(decoder_output[:, -1, :])  # 取最后一个时刻的输出作为结果
        return output

input_data = torch.randn(100, 10, 32)  # 假设输入数据形状为(batch_size, sequence_length, input_dim)
target_data = torch.randn(100, 10, 1)  # 假设目标数据形状为(batch_size, sequence_length, output_dim)
batch_size = 10
dataset = CustomDataset(input_data, target_data)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 设置模型参数
input_dim = 32
output_dim = 1
d_model = 64
nhead = 4
num_layers = 2

# 创建模型并移动到设备上
model = TransformerModel(input_dim, output_dim, d_model, nhead, num_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模型训练
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, targets)

        loss = criterion(outputs, targets[:, -1, :])  # 计算损失时只使用最后一个时刻的目标作为对比
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss}")

# 测试模型
test_input = torch.randn(1, 10, 32)  # 假设测试输入数据形状为(1, sequence_length, input_dim)
test_input = test_input.to(device)
with torch.no_grad():
    model.eval()
    predicted_output = model(test_input, None)  # 在测试阶段不需要目标数据，传入None
    print("Predicted Output:", predicted_output)

'''
	一：
	Embedding层：词表大小乘以隐藏层 V * h。
	位置编码：假设位置编码是冻结和计算的，只使用一个位置编码，固定不变的，维度为（s*h）, 参数数量为 s * h。
	Multihead_attention：3*a*h*h (3: qkv)  (假设只有考虑encoder只有一个multihead_attention)
	FFN: 假设2个Linear, 中间的维度为 l_dim：a*h*l_dim+l_dim*h
	最终模型参数:
	l * (3 * a * h*h + a*h*l_dim+l_dim*h )+ s * h + V * h
	注： 如果多头注意力时隐藏层维度变为 h / a, 再concat（暂定保持维度为h,输入到linear），那么multihead_atten层参数量需要除以a, 线性第一层也需要

'''