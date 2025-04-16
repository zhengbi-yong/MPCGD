# 2025年4月16日
# 使用 MLP 进行 MNIST 分类，监控中间层 Loss (通过共享最终输出头计算)，仅使用最终 Loss 更新梯度
# 新增：可视化单个样本在各层的预测结果

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
results_base_dir = "results_mnist_shared_head" # 修改文件夹基础名
results_dir = os.path.join(results_base_dir, timestamp_str)
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
# --- 用于训练/测试的变换 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# --- 用于可视化的反向变换 (近似) ---
# Note: Denormalization might not perfectly restore the original pixel values
# due to clamping or floating point inaccuracies, but it's good for visualization.
inv_normalize = transforms.Normalize(
   mean=(-0.1307/0.3081,),
   std=(1.0/0.3081,)
)
# --- 用于获取原始图像的变换 (仅 ToTensor) ---
vis_transform = transforms.ToTensor()


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# --- 获取原始图像和标签用于可视化 ---
train_dataset_vis = datasets.MNIST(root='./data', train=True, download=True, transform=vis_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 定义 MLP 模型
class MNIST_MLPWithSharedOutputHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(MNIST_MLPWithSharedOutputHead, self).__init__()
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.final_output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        representations_to_evaluate = []
        h = self.activation(self.input_layer(x))
        representations_to_evaluate.append(h)
        current_h = h
        for hidden_layer in self.hidden_layers:
            current_h = self.activation(hidden_layer(current_h))
            representations_to_evaluate.append(current_h)
        all_outputs = []
        for h_repr in representations_to_evaluate:
            prediction = self.final_output_head(h_repr)
            all_outputs.append(prediction)
        return all_outputs


# 4. 实例化模型、损失函数和优化器
model = MNIST_MLPWithSharedOutputHead(input_dim, hidden_dim, output_dim, num_hidden_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练循环
losses_history = [[] for _ in range(num_hidden_layers + 1)]
for epoch in range(num_epochs):
    model.train()
    epoch_batch_losses = [[] for _ in range(num_hidden_layers + 1)]
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        all_predictions = model(data)
        final_loss_tensor = None
        for i in range(num_hidden_layers + 1):
            loss = criterion(all_predictions[i], target)
            epoch_batch_losses[i].append(loss.item())
            if i == num_hidden_layers:
                final_loss_tensor = loss
        optimizer.zero_grad()
        if final_loss_tensor is not None:
            final_loss_tensor.backward()
        else:
             raise ValueError("Final loss tensor was not calculated.")
        optimizer.step()
        if batch_idx % 100 == 0:
             print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                   f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                   f'Final Rep. Loss (for backprop): {final_loss_tensor.item():.6f}')
    avg_epoch_losses = [np.mean(batch_losses) if len(batch_losses) > 0 else 0 for batch_losses in epoch_batch_losses]
    print(f"\nEpoch {epoch+1} Average Losses (Shared Head):")
    loss_str = " | ".join([f"L{i+1}: {l:.4f}" for i, l in enumerate(avg_epoch_losses)])
    print(f"[{loss_str}]\n")
    for i in range(num_hidden_layers + 1):
        losses_history[i].append(avg_epoch_losses[i])


# 6. 绘图并保存 Loss 历史
plt.figure(figsize=(12, 6))
for i in range(num_hidden_layers + 1):
    label = f'Loss from Rep. after Input Layer Act.' if i == 0 else f'Loss from Rep. after Hidden Layer {i} Act.'
    if i == num_hidden_layers: label += ' (Used for Backprop)'
    else: label += ' (Recorded Only, Shared Head)'
    plt.plot(range(1, num_epochs + 1), losses_history[i], label=label, marker='o', linestyle='-')
plt.title('MNIST Training Loss History (Shared Output Head, Only Final Loss for Backprop)')
plt.xlabel('Epoch')
plt.ylabel('Average Cross-Entropy Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend(fontsize='small')
plt.grid(True)
plt.yscale('log')
loss_plot_path = os.path.join(results_dir, 'mnist_loss_history_shared_head.png')
plt.savefig(loss_plot_path)
print(f"Loss history plot saved to: {loss_plot_path}")
# plt.show() #注释掉show，防止阻塞后续代码执行，除非你想在运行时看到图
plt.close() # 关闭图形，释放内存

# 7. 评估模型性能
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        all_outputs = model(data)
        final_output = all_outputs[-1]
        test_loss += criterion(final_output, target).item()
        pred = final_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
test_loss /= len(test_loader)
print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
      f'({100. * correct / len(test_loader.dataset):.2f}%)\n')


# 8. 单个样本可视化 (****** 新增部分 ******)
print("Visualizing single sample predictions...")
model.eval() # 确保模型在评估模式

# --- 选择一个样本 ---
sample_idx = 0 # 可以修改这个索引来查看不同的样本
# 从 transform=vis_transform 的数据集中获取原始图像tensor
original_image_tensor, true_label = train_dataset_vis[sample_idx]
# 从 transform=transform 的数据集中获取标准化后的图像tensor用于模型输入
normalized_image_tensor, _ = train_dataset[sample_idx]

print(f"Selected sample index: {sample_idx}, True Label: {true_label}")

# --- 预处理用于模型的输入 ---
sample_input = normalized_image_tensor.unsqueeze(0).to(device) # 添加batch维度并移到设备

# --- 模型推理 ---
with torch.no_grad():
    all_sample_outputs_logits = model(sample_input) # 获取所有层的logits输出列表

# --- 后处理预测结果 ---
probabilities_list = []
predicted_classes_list = []
for logits in all_sample_outputs_logits:
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy() # 转为概率,移除batch维,转numpy
    pred_class = np.argmax(probs)
    probabilities_list.append(probs)
    predicted_classes_list.append(pred_class)

# --- 可视化 ---
num_layers_to_plot = num_hidden_layers + 1
# 创建子图: 1行, 1个图像位 + (N+1)个概率图位
fig, axes = plt.subplots(1, 1 + num_layers_to_plot, figsize=(4 * (1 + num_layers_to_plot), 4))

# 绘制原始图像
ax_img = axes[0]
# 显示原始图像 (不需要反标准化，因为我们从vis_transform获取)
ax_img.imshow(original_image_tensor.squeeze(), cmap='gray')
ax_img.set_title(f"Original Image\nTrue Label: {true_label}")
ax_img.axis('off')

# 绘制每个表示层的预测概率分布
for i in range(num_layers_to_plot):
    ax_prob = axes[i+1]
    probs = probabilities_list[i]
    pred_class = predicted_classes_list[i]

    # 创建条形图
    bars = ax_prob.bar(range(10), probs, color='skyblue')
    ax_prob.set_xticks(range(10))
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("Probability")
    ax_prob.grid(axis='y', linestyle='--', alpha=0.7)

    # 突出显示真实标签和预测标签的条形
    bars[true_label].set_color('green')
    if pred_class != true_label:
        bars[pred_class].set_color('red')
    else: # 如果预测正确，红色条会被绿色覆盖，再设一次确保绿色
        bars[pred_class].set_color('green')


    # 设置标题
    layer_name = f'Input Layer Act.' if i == 0 else f'Hidden Layer {i} Act.'
    title = f"Pred. from {layer_name}\nPredicted: {pred_class} (P={probs[pred_class]:.2f})"
    ax_prob.set_title(title, fontsize=9)

plt.tight_layout(pad=2.0) # 调整布局，增加间距

# 保存可视化结果图
vis_plot_path = os.path.join(results_dir, f'sample_{sample_idx}_predictions.png')
plt.savefig(vis_plot_path)
print(f"Single sample visualization saved to: {vis_plot_path}")
# plt.show() # 显示图片
# plt.close() # 关闭图形