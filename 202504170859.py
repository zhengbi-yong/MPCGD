# 2025年4月16日
# 使用 MLP 进行 MNIST 分类，监控中间层 Loss (通过共享最终输出头计算)
# 修改：将所有中间层和最终层的 Loss 加起来，用于梯度更新
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
results_base_dir = "results"
results_dir = "mnist_shared_head_combined_loss"
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
# --- 用于训练/测试的变换 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# --- 用于可视化的反向变换 (近似) ---
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
        self.final_output_head = nn.Linear(hidden_dim, output_dim) # Shared Head

    def forward(self, x):
        x = x.view(x.size(0), -1)
        representations_to_evaluate = []
        h = self.activation(self.input_layer(x))
        representations_to_evaluate.append(h) # Representation after Input Layer Activation
        current_h = h
        for hidden_layer in self.hidden_layers:
            current_h = self.activation(hidden_layer(current_h))
            representations_to_evaluate.append(current_h) # Rep. after Hidden Layer 1, 2, ... N Activation
        all_outputs = []
        # Use the *shared* head to get predictions from each representation
        for h_repr in representations_to_evaluate:
            prediction = self.final_output_head(h_repr)
            all_outputs.append(prediction)
        # Returns a list of output tensors, one for each representation layer
        # [pred_from_input_act, pred_from_h1_act, ..., pred_from_hN_act]
        return all_outputs


# 4. 实例化模型、损失函数和优化器
model = MNIST_MLPWithSharedOutputHead(input_dim, hidden_dim, output_dim, num_hidden_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练循环 (****** 修改的部分 ******)
losses_history = [[] for _ in range(num_hidden_layers + 1)] # Store avg loss from each layer per epoch
for epoch in range(num_epochs):
    model.train()
    epoch_batch_losses = [[] for _ in range(num_hidden_layers + 1)] # Store individual batch losses for averaging
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # --- Forward Pass ---
        # Get predictions from *all* specified layers using the shared head
        all_predictions = model(data)

        # --- Loss Calculation ---
        # Calculate loss for each layer's prediction and sum them up for backprop
        total_loss_for_backprop = torch.tensor(0.0, device=device, requires_grad=True) # Initialize sum tensor
        # The length of all_predictions is num_hidden_layers + 1
        for i in range(num_hidden_layers + 1):
            loss = criterion(all_predictions[i], target) # Calculate individual loss
            epoch_batch_losses[i].append(loss.item()) # Log individual loss value
            total_loss_for_backprop = total_loss_for_backprop + loss # Accumulate loss tensor for gradient calculation

        # --- Backpropagation ---
        optimizer.zero_grad()
        # Use the *summed* loss to compute gradients
        total_loss_for_backprop.backward()
        optimizer.step()

        # --- Logging ---
        if batch_idx % 100 == 0:
             # Log the total loss used for backprop
             print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                   f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                   f'Total Loss (Summed, for backprop): {total_loss_for_backprop.item():.6f}') # Updated print statement

    # --- Epoch Summary ---
    avg_epoch_losses = [np.mean(batch_losses) if len(batch_losses) > 0 else 0 for batch_losses in epoch_batch_losses]
    print(f"\nEpoch {epoch+1} Average Individual Losses (Shared Head, Summed for Backprop):")
    # Display the average *individual* component losses for monitoring
    loss_str = " | ".join([f"L{i+1}: {l:.4f}" for i, l in enumerate(avg_epoch_losses)])
    print(f"[{loss_str}]\n")

    # Store the history of average individual losses
    for i in range(num_hidden_layers + 1):
        losses_history[i].append(avg_epoch_losses[i])


# 6. 绘图并保存 Loss 历史 (****** 修改标签 ******)
plt.figure(figsize=(12, 7)) # Increased height slightly for legend
for i in range(num_hidden_layers + 1):
    label = f'Loss Component from Rep. after Input Layer Act.' if i == 0 else f'Loss Component from Rep. after Hidden Layer {i} Act.'
    # Update label to reflect that all losses are summed for backprop
    label += ' (Component, Summed for Backprop)'
    plt.plot(range(1, num_epochs + 1), losses_history[i], label=label, marker='o', linestyle='-')

# Add a line representing the sum of average component losses (approximation of total loss trend)
sum_of_avg_losses = [sum(epoch_losses) for epoch_losses in zip(*losses_history)]
plt.plot(range(1, num_epochs + 1), sum_of_avg_losses, label='Sum of Avg. Loss Components (Approx. Total Loss Trend)', marker='x', linestyle='--', color='black')


plt.title('MNIST Training Loss History (Shared Head, Sum of All Losses for Backprop)') # Updated title
plt.xlabel('Epoch')
plt.ylabel('Average Cross-Entropy Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend(fontsize='small') # Keep legend concise
plt.grid(True)
plt.yscale('log') # Keep log scale as losses can vary significantly
loss_plot_path = os.path.join(results_dir, 'mnist_loss_history_summed_loss.png') # Updated filename
plt.savefig(loss_plot_path)
print(f"Loss history plot saved to: {loss_plot_path}")
# plt.show()
plt.close() # 关闭图形，释放内存

# 7. 评估模型性能
# Note: Evaluation typically uses only the *final* layer's output for accuracy calculation,
# even if training used summed losses. This reflects standard practice.
model.eval()
test_loss = 0 # Based on final layer output
correct = 0   # Based on final layer output
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        all_outputs = model(data)
        final_output = all_outputs[-1] # Evaluate using the prediction from the last representation
        test_loss += criterion(final_output, target).item() # Calculate test loss using final output only
        pred = final_output.argmax(dim=1, keepdim=True) # Get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader) # Use average loss over batches
print(f'\nTest set: Average loss (final layer): {test_loss:.4f}, Accuracy (final layer): {correct}/{len(test_loader.dataset)} '
      f'({100. * correct / len(test_loader.dataset):.2f}%)\n')


# 8. 单个样本可视化 (No changes needed here, it visualizes predictions from all layers)
print("Visualizing single sample predictions...")
model.eval() # Ensure model is in eval mode

# --- 选择一个样本 ---
sample_idx = 0 # You can change this index
original_image_tensor, true_label = train_dataset_vis[sample_idx]
normalized_image_tensor, _ = train_dataset[sample_idx]

print(f"Selected sample index: {sample_idx}, True Label: {true_label}")

# --- 预处理用于模型的输入 ---
sample_input = normalized_image_tensor.unsqueeze(0).to(device)

# --- 模型推理 ---
with torch.no_grad():
    all_sample_outputs_logits = model(sample_input)

# --- 后处理预测结果 ---
probabilities_list = []
predicted_classes_list = []
for logits in all_sample_outputs_logits:
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_class = np.argmax(probs)
    probabilities_list.append(probs)
    predicted_classes_list.append(pred_class)

# --- 可视化 ---
num_layers_to_plot = num_hidden_layers + 1
fig, axes = plt.subplots(1, 1 + num_layers_to_plot, figsize=(4 * (1 + num_layers_to_plot), 4))

# 绘制原始图像
ax_img = axes[0]
ax_img.imshow(original_image_tensor.squeeze(), cmap='gray')
ax_img.set_title(f"Original Image\nTrue Label: {true_label}")
ax_img.axis('off')

# 绘制每个表示层的预测概率分布
for i in range(num_layers_to_plot):
    ax_prob = axes[i+1]
    probs = probabilities_list[i]
    pred_class = predicted_classes_list[i]

    bars = ax_prob.bar(range(10), probs, color='skyblue')
    ax_prob.set_xticks(range(10))
    ax_prob.set_ylim(0, 1)
    ax_prob.set_ylabel("Probability")
    ax_prob.grid(axis='y', linestyle='--', alpha=0.7)

    bars[true_label].set_color('green')
    if pred_class != true_label:
        bars[pred_class].set_color('red')
    else:
        bars[pred_class].set_color('green')

    layer_name = f'Input Layer Act.' if i == 0 else f'Hidden Layer {i} Act.'
    title = f"Pred. from {layer_name}\nPredicted: {pred_class} (P={probs[pred_class]:.2f})"
    ax_prob.set_title(title, fontsize=9)

plt.tight_layout(pad=2.0)

vis_plot_path = os.path.join(results_dir, f'sample_{sample_idx}_predictions_summed_loss_train.png') # Updated filename
plt.savefig(vis_plot_path)
print(f"Single sample visualization saved to: {vis_plot_path}")
# plt.show()
plt.close()