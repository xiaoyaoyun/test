import torch
import torch.nn as nn

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

teacher_model = TeacherModel()
student_model = StudentModel()

criterion = nn.CrossEntropyLoss()
def knowledge_distillation_loss(student_logits, teacher_logits, temperature):
    loss_ce = criterion(student_logits, teacher_logits.argmax(dim=1))
    loss_kd = nn.KLDivLoss()(torch.log_softmax(student_logits / temperature, dim=1),
                             torch.softmax(teacher_logits / temperature, dim=1))
    loss = loss_ce + loss_kd
    return loss

input_data = torch.randn(100, 10)
teacher_logits = teacher_model(input_data)
temperature = 10.0  # 控制软目标的温度参数
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.01)

for epoch in range(100):
    student_logits = student_model(input_data)
    loss = knowledge_distillation_loss(student_logits, teacher_logits, temperature)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item()}')
