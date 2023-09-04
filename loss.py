import torch
import torch.nn as nn

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def binary_cross_entropy(pred, target):
    epsilon = 1e-7
    loss = -target * torch.log(pred + epsilon) - (1 - target) * torch.log(1 - pred + epsilon)
    loss = torch.mean(loss)
    return loss

def grad(pred, target):
    d_pred = torch.zeros_like(pred)
    print(pred.shape)
    delta = 1e-7
    for i in range(len(pred)):
        pred_plus_delta = pred.clone()
        pred_plus_delta[i] += delta

        loss = binary_cross_entropy(pred, target)
        loss_plus_delta = binary_cross_entropy(pred_plus_delta, target)
        d_pred[i] = (loss_plus_delta - loss) / delta

    return d_pred

def softmax(x):
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
    softmax_x = exp_x / sum_exp_x
    return softmax_x

# L(x1, x2, y) = y * D(x1, x2) + (1 - y) * max(0, margin - D(x1, x2))
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, x1, x2, y):
        euclidean_distance = F.pairwise_distance(x1, x2)
        loss_contrastive = torch.mean((1 - y) * torch.pow(euclidean_distance, 2) +
                                      y * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# LLM
'''
    if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
'''



predicted = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
target = torch.tensor([1, 1, 1, 0], dtype=torch.float32)
x1 = torch.tensor([[1.0, 2.0, 3.0]])
x2 = torch.tensor([[4.0, 5.0, 6.0]])
y = torch.tensor([1.0])  # 1表示相似

contrastive_loss = ContrastiveLoss(margin=2.0)

loss = contrastive_loss(x1, x2, y)
print("Contrastive Loss:", loss.item())

input_data = torch.tensor([[0.2], [-0.7]])  # 预测值
target_data = torch.tensor([[0], [1]])     # 真实标签

pred = sigmoid(input_data)
print('pred: ', pred)
loss = binary_cross_entropy(pred, target_data)
print('loss: ', loss)

grads = grad(pred, target_data)

# 输出导数值
print(grads)