import torch

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def binary_cross_entropy(pred, target):
    epsilon = 1e-7
    loss = -target * torch.log(pred + epsilon) - (1 - target) * torch.log(1 - pred + epsilon)
    loss = torch.mean(loss)
    return loss

def grad(pred, target):
    # 创建一个与预测值形状相同的张量，用于存储导数值
    d_pred = torch.zeros_like(pred)
    print(pred.shape)
    # 计算损失关于每个预测值的导数
    delta = 1e-7

    for i in range(len(pred)):
        # 在索引i处增加一个小的扰动
        pred_plus_delta = pred.clone()
        pred_plus_delta[i] += delta

        # 使用有限差分近似计算导数（中心差分）
        loss = binary_cross_entropy(pred, target)
        loss_plus_delta = binary_cross_entropy(pred_plus_delta, target)
        d_pred[i] = (loss_plus_delta - loss) / delta

    return d_pred

# 示例数据
input_data = torch.tensor([[0.2], [-0.7]])  # 预测值
target_data = torch.tensor([[0], [1]])     # 真实标签

# 计算预测值
pred = sigmoid(input_data)
print('pred: ', pred)
# 计算损失
loss = binary_cross_entropy(pred, target_data)
print('loss: ', loss)

# 手动求导
grads = grad(pred, target_data)

# 输出导数值
print(grads)