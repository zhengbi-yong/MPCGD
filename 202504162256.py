# 2025年4月16日
# 使用 MLP 进行 MNIST 分类，监控中间层 Loss (通过共享最终输出头计算)，仅使用最终 Loss 更新梯度

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 创建结果文件夹 ---
now = datetime.datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
results_base_dir = "results"
results_dir = "mnist_shared_head_final_loss"
results_dir = os.path.join(results_base_dir, results_dir, timestamp_str)
os.makedirs(results_dir, exist_ok=True)
print(f"结果将保存在: {results_dir}")
# ---------------------

# 1. 超参数设置
input_dim = 28 * 28   # MNIST image dimension (flattened)
output_dim = 10       # Number of classes (digits 0-9)
hidden_dim = 128      # Hidden layer dimension (can be tuned)
num_hidden_layers = 4 # Number of hidden layers (e.g., 4 means 1 input + 4 hidden + 1 output head structure)
num_epochs = 20       # Number of training epochs (adjust as needed)
batch_size = 128      # Batch size for training
learning_rate = 1e-3  # Learning rate

# 确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 2. 数据准备 (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 定义 MLP 模型 (****** 修改后的版本 ******)
class MNIST_MLPWithSharedOutputHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(MNIST_MLPWithSharedOutputHead, self).__init__()
        self.activation = nn.ReLU()

        # 输入层
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # 隐藏层 (N = num_hidden_layers 个)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 共享的最终输出头
        self.final_output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1) # 展平

        representations_to_evaluate = []

        # 计算并存储输入层激活后的表示
        h = self.activation(self.input_layer(x))
        representations_to_evaluate.append(h) # 表示 0 (Input Act后)

        # 计算并存储每个隐藏层激活后的表示
        current_h = h
        for hidden_layer in self.hidden_layers: # 循环 N 次 (N=num_hidden_layers)
            current_h = self.activation(hidden_layer(current_h))
            representations_to_evaluate.append(current_h) # 表示 1 到 N (Hidden Act后)

        # representations_to_evaluate 列表现在包含 N+1 个表示

        # 使用共享的 final_output_head 计算所有表示的预测
        all_outputs = []
        for h_repr in representations_to_evaluate:
            prediction = self.final_output_head(h_repr)
            all_outputs.append(prediction)

        # 返回包含 N+1 个预测的列表
        return all_outputs


# 4. 实例化模型、损失函数和优化器
model = MNIST_MLPWithSharedOutputHead(input_dim, hidden_dim, output_dim, num_hidden_layers).to(device)
criterion = nn.CrossEntropyLoss()
# 优化器需要优化所有参数，包括输入层、隐藏层和共享的输出头
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练循环
# 存储每个表示（通过共享头计算）的平均损失历史
losses_history = [[] for _ in range(num_hidden_layers + 1)] # N+1 个损失历史

for epoch in range(num_epochs):
    model.train()
    # 存储当前 epoch 内每个表示的 batch 损失
    epoch_batch_losses = [[] for _ in range(num_hidden_layers + 1)]

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播：获取所有 N+1 个表示通过共享头产生的预测
        all_predictions = model(data)

        current_batch_losses_values = []
        final_loss_tensor = None # 用于反向传播的损失（基于最后一个表示）

        # 计算每个预测的损失
        for i in range(num_hidden_layers + 1): # 循环 N+1 次
            loss = criterion(all_predictions[i], target)
            loss_item = loss.item()
            current_batch_losses_values.append(loss_item)
            epoch_batch_losses[i].append(loss_item) # 记录 batch loss

            # 存储最后一个表示产生的损失（用于梯度计算）
            if i == num_hidden_layers:
                final_loss_tensor = loss

        # 反向传播和优化（仅使用最后一个表示的损失）
        optimizer.zero_grad()
        if final_loss_tensor is not None:
            final_loss_tensor.backward()
        else:
             raise ValueError("Final loss tensor was not calculated.")
        optimizer.step()

        # 打印 batch 进度
        if batch_idx % 100 == 0:
             print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                   f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                   f'Final Rep. Loss (for backprop): {final_loss_tensor.item():.6f}')

    # 计算并记录 epoch 的平均损失
    avg_epoch_losses = [np.mean(batch_losses) if len(batch_losses) > 0 else 0 for batch_losses in epoch_batch_losses] # Handle potential empty list if epoch is short
    print(f"\nEpoch {epoch+1} Average Losses (Shared Head):")
    # L1=InputAct后, L2=H1Act后, ..., L(N+1)=HNAct后
    loss_str = " | ".join([f"L{i+1}: {l:.4f}" for i, l in enumerate(avg_epoch_losses)])
    print(f"[{loss_str}]\n")
    for i in range(num_hidden_layers + 1):
        losses_history[i].append(avg_epoch_losses[i])


# 6. 绘图并保存 Loss 历史
plt.figure(figsize=(12, 6))
for i in range(num_hidden_layers + 1):
    # 更新标签以反映表示的来源
    label = f'Loss from Rep. after Input Layer Act.' if i == 0 else f'Loss from Rep. after Hidden Layer {i} Act.'
    if i == num_hidden_layers:
        label += ' (Used for Backprop)'
    else:
        label += ' (Recorded Only, Shared Head)'

    plt.plot(range(1, num_epochs + 1), losses_history[i], label=label, marker='o', linestyle='-')

plt.title('MNIST Training Loss History (Shared Output Head, Only Final Loss for Backprop)')
plt.xlabel('Epoch')
plt.ylabel('Average Cross-Entropy Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend(fontsize='small') # Adjust legend font size if needed
plt.grid(True)
plt.yscale('log') # Log scale often helpful for loss

loss_plot_path = os.path.join(results_dir, 'mnist_loss_history_shared_head.png') # 使用新文件名
plt.savefig(loss_plot_path)
print(f"Loss history plot saved to: {loss_plot_path}")
# plt.show()
# plt.close()

# 7. 评估模型性能
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # 前向传播获取所有预测
        all_outputs = model(data)
        # 使用最后一个预测进行评估
        final_output = all_outputs[-1]
        test_loss += criterion(final_output, target).item() # 累加 batch loss
        pred = final_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader) # 计算平均测试损失

print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
      f'({100. * correct / len(test_loader.dataset):.2f}%)\n')