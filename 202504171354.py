# 2025年4月17日
# 使用 MLP 进行 MNIST 超分辨率 (LR -> HR)，监控中间层 Loss (通过共享最终输出头计算MSE)
# 修改：使用所有中间层和最终层的 Loss 之和来更新梯度
# 新增：可视化单个样本在各层的重建结果
# 修正：确保可视化中的 LR 和 HR Ground Truth 来源于同一个样本

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Dataset
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
results_dir = "mnist_sr_shared_head_combined_loss"
results_dir = os.path.join(results_base_dir, results_dir, timestamp_str)
os.makedirs(results_dir, exist_ok=True)
print(f"结果将保存在: {results_dir}")
# ---------------------

# 1. 超参数设置
hr_size = 28
lr_scale = 2 # Downscaling factor
lr_size = hr_size // lr_scale
input_dim = lr_size * lr_size   # LR image dimension (flattened)
output_dim = hr_size * hr_size  # HR image dimension (flattened)
hidden_dim = 256     # Hidden layer dimension
num_hidden_layers = 4 # Number of hidden layers
num_epochs = 30
batch_size = 128
learning_rate = 1e-3

# 确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 2. 数据准备 (MNIST SR)

# --- Transforms ---
norm_mean = (0.1307,)
norm_std = (0.3081,)

hr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

lr_transform = transforms.Compose([
    transforms.Resize((lr_size, lr_size), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

vis_transform = transforms.ToTensor()

# --- Custom Dataset for SR ---
class MNIST_SR_Dataset(Dataset):
    def __init__(self, mnist_dataset, lr_transform, hr_transform):
        self.mnist_dataset = mnist_dataset
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.base_data = mnist_dataset.data
        self.base_targets = mnist_dataset.targets

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        pil_image = transforms.ToPILImage()(self.base_data[idx])
        lr_image = self.lr_transform(pil_image)
        hr_image_transformed = self.hr_transform(pil_image)
        return lr_image, hr_image_transformed

# --- Create Datasets and Loaders ---
base_train_dataset = datasets.MNIST(root='./data', train=True, download=True)
base_test_dataset = datasets.MNIST(root='./data', train=False, download=True)

train_dataset = MNIST_SR_Dataset(base_train_dataset, lr_transform, hr_transform)
test_dataset = MNIST_SR_Dataset(base_test_dataset, lr_transform, hr_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 定义 MLP 模型 for SR (No change in model architecture)
class SR_MLPWithSharedOutputHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(SR_MLPWithSharedOutputHead, self).__init__()
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.final_output_head = nn.Linear(hidden_dim, output_dim)
        # self.output_activation = nn.Tanh() # Optional

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
            prediction_flat = self.final_output_head(h_repr)
            # prediction_flat = self.output_activation(prediction_flat) # If using output activation
            all_outputs.append(prediction_flat)

        # Returns a list of *flattened* predicted HR image tensors
        # [pred_hr_flat_from_input_act, pred_hr_flat_from_h1_act, ..., pred_hr_flat_from_hN_act]
        return all_outputs

# 4. 实例化模型、损失函数和优化器
model = SR_MLPWithSharedOutputHead(input_dim, hidden_dim, output_dim, num_hidden_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练循环 (****** 修改部分 ******: Use sum of all intermediate losses for backprop)
losses_history = [[] for _ in range(num_hidden_layers + 1)] # Store avg MSE loss from each layer per epoch
print("--- 开始训练 (使用所有中间层Loss之和进行梯度更新) ---")
for epoch in range(num_epochs):
    model.train()
    epoch_batch_losses = [[] for _ in range(num_hidden_layers + 1)] # Store individual batch losses for averaging
    for batch_idx, (lr_data, hr_target) in enumerate(train_loader):
        lr_data, hr_target = lr_data.to(device), hr_target.to(device)
        hr_target_flat = hr_target.view(hr_target.size(0), -1) # (N, output_dim)

        # --- Forward Pass ---
        all_predictions_flat = model(lr_data) # List of flattened predictions

        # --- Loss Calculation & Recording ---
        # ******* MODIFICATION START *******
        # Calculate the loss for each layer's prediction and sum them up
        total_loss = torch.tensor(0.0, device=device, requires_grad=True) # Initialize total loss for this batch
        # The length of all_predictions_flat is num_hidden_layers + 1
        for i in range(num_hidden_layers + 1):
            # Calculate MSE loss between the i-th prediction and the target
            loss = criterion(all_predictions_flat[i], hr_target_flat)
            epoch_batch_losses[i].append(loss.item()) # Log individual loss value

            # Add this layer's loss to the total loss for backpropagation
            total_loss = total_loss + loss
        # ******* MODIFICATION END *******

        # --- Backpropagation (using the combined loss) ---
        optimizer.zero_grad()
        total_loss.backward() # Backpropagate the sum of losses
        optimizer.step()

        # --- Logging ---
        if batch_idx % 100 == 0:
            # Report the total combined loss used for backprop
            print(f'Train Epoch: {epoch+1} [{batch_idx * len(lr_data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Total Combined MSE Loss (for backprop): {total_loss.item():.6f}') # Log total loss

    # --- Epoch Summary ---
    avg_epoch_losses = [np.mean(batch_losses) if len(batch_losses) > 0 else 0 for batch_losses in epoch_batch_losses]
    print(f"\nEpoch {epoch+1} Average Individual MSE Losses (Shared Head, Combined Loss Backprop):")
    loss_str = " | ".join([f"L{i+1}_MSE: {l:.6f}" for i, l in enumerate(avg_epoch_losses)]) # Format individual MSEs
    print(f"[{loss_str}]\n")
    print(f"Epoch {epoch+1} Average Total Combined MSE Loss: {np.sum(avg_epoch_losses):.6f}\n") # Also print the sum of averages

    # Store the history of average individual MSE losses
    for i in range(num_hidden_layers + 1):
        losses_history[i].append(avg_epoch_losses[i])

print("--- 训练完成 ---")

# 6. 绘图并保存 Loss 历史
plt.figure(figsize=(12, 7)) # Slightly larger figure
for i in range(num_hidden_layers + 1):
    label = f'MSE from Rep. after Input Layer Act.' if i == 0 else f'MSE from Rep. after Hidden Layer {i} Act.'
    # All these MSEs contributed to the combined loss used for backprop
    plt.plot(range(1, num_epochs + 1), losses_history[i], label=label, marker='o', linestyle='-')

# Add a line for the sum of average losses (representing the total loss magnitude)
sum_avg_losses = [sum(epoch_losses) for epoch_losses in zip(*losses_history)]
plt.plot(range(1, num_epochs + 1), sum_avg_losses, label='Sum of Average MSEs (Total Loss Trend)', color='black', linestyle='--', linewidth=2)

plt.title('MNIST SR Training: Individual & Total MSE Loss History\n(Shared Head, Combined Intermediate Loss for Backprop)')
plt.xlabel('Epoch')
plt.ylabel('Average MSE Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend(fontsize='small', loc='upper right') # Adjust legend location if needed
plt.grid(True)
plt.yscale('log') # Log scale is often helpful for losses
loss_plot_path = os.path.join(results_dir, 'mnist_sr_loss_history_combined.png')
plt.savefig(loss_plot_path)
print(f"Loss history plot saved to: {loss_plot_path}")
plt.close()

# 7. 评估模型性能 (Still evaluate based on the final layer's output for consistency)
model.eval()
test_mse_loss_final_layer = 0 # Based on final layer output only for evaluation metric
with torch.no_grad():
    for lr_data, hr_target in test_loader:
        lr_data, hr_target = lr_data.to(device), hr_target.to(device)
        hr_target_flat = hr_target.view(hr_target.size(0), -1)

        all_outputs_flat = model(lr_data)
        final_output_flat = all_outputs_flat[-1] # Evaluate using the prediction from the last representation

        test_mse_loss_final_layer += criterion(final_output_flat, hr_target_flat).item()

# Average loss over number of batches
test_mse_loss_final_layer /= len(test_loader)
print(f'\nTest set: Average MSE loss (final layer output): {test_mse_loss_final_layer:.6f}\n')


# 8. 单个样本可视化 (No change needed here, visualization logic is independent of training method)
print("Visualizing single sample reconstruction...")
model.eval()

# --- 选择一个样本 ---
sample_idx = 5
print(f"Selected sample index from TEST SET: {sample_idx}")

# --- 获取模型输入 (LR) 和 归一化目标 (HR) ---
lr_sample_tensor, hr_sample_tensor_norm = test_dataset[sample_idx]

# --- 获取原始未归一化的 HR 图像用于可视化 ---
original_uint8_tensor = base_test_dataset.data[sample_idx]
original_pil_image = transforms.ToPILImage()(original_uint8_tensor)
hr_original_tensor = vis_transform(original_pil_image)

# --- 预处理用于模型的输入 ---
lr_input_batch = lr_sample_tensor.unsqueeze(0).to(device)

# --- 模型推理 ---
with torch.no_grad():
    all_sample_outputs_flat = model(lr_input_batch)

# --- 后处理预测结果 ---
predicted_hr_images = []
for output_flat in all_sample_outputs_flat:
    pred_img = output_flat.view(1, hr_size, hr_size).squeeze().cpu().numpy()
    predicted_hr_images.append(pred_img)

# --- 可视化 ---
num_layers_to_plot = num_hidden_layers + 1
num_total_plots = 2 + num_layers_to_plot
fig, axes = plt.subplots(1, num_total_plots, figsize=(3 * num_total_plots, 3.5))

def imshow_denormalize(ax, img_tensor, title):
    img = img_tensor.cpu().squeeze().numpy()
    mean_val = norm_mean[0]
    std_val = norm_std[0]
    img = img * std_val + mean_val
    img = np.clip(img, 0, 1)
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=9)
    ax.axis('off')

# 1. 绘制 LR 输入图像 (denormalized)
lr_display = F.interpolate(lr_sample_tensor.unsqueeze(0), size=(hr_size, hr_size), mode='bicubic', align_corners=False).squeeze(0)
imshow_denormalize(axes[0], lr_display, "Input LR\n(Upscaled Bicubic)")

# 2. 绘制 HR Ground Truth 图像 (original, unnormalized)
axes[1].imshow(hr_original_tensor.squeeze(), cmap='gray')
axes[1].set_title("Ground Truth HR", fontsize=9)
axes[1].axis('off')

# 3. 绘制每个表示层生成的 HR 预测图像 (denormalized)
for i in range(num_layers_to_plot):
    ax_pred = axes[i + 2]
    pred_img_numpy = predicted_hr_images[i]

    mean_val = norm_mean[0]
    std_val = norm_std[0]
    pred_img_denorm = pred_img_numpy * std_val + mean_val
    pred_img_denorm = np.clip(pred_img_denorm, 0, 1)

    layer_name = f'Input Act.' if i == 0 else f'Hidden {i} Act.'
    title = f"Pred. from {layer_name}"

    with torch.no_grad():
        prediction_tensor_flat = all_sample_outputs_flat[i].view(-1)
        target_tensor_flat = hr_sample_tensor_norm.view(-1).to(device)
        mse_val = F.mse_loss(prediction_tensor_flat, target_tensor_flat).item()

    title += f"\nMSE: {mse_val:.4f}"

    ax_pred.imshow(pred_img_denorm, cmap='gray')
    ax_pred.set_title(title, fontsize=9)
    ax_pred.axis('off')

plt.tight_layout(pad=0.5)

vis_plot_path = os.path.join(results_dir, f'sample_{sample_idx}_reconstruction_combined_loss.png')
plt.savefig(vis_plot_path)
print(f"Single sample visualization saved to: {vis_plot_path}")
plt.close()