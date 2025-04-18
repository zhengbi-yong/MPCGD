# 2025年4月17日
# 使用 MLP 进行 MNIST 超分辨率 (LR -> HR)，监控中间层 Loss (通过共享最终输出头计算MSE)
# 修改：使用所有中间层和最终层的 Loss 之和来更新梯度
# 新增：可视化单个样本在各层的重建结果 (加入 PSNR, SSIM, MAE 指标)
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
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure # 新增导入

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 创建结果文件夹 ---
now = datetime.datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
results_base_dir = "results"
results_dir = "mnist_sr_shared_head_combined_loss_metrics"
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
num_epochs = 30      # Reduced epochs for faster testing, increase as needed
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

# Transform for visualization (only ToTensor, no normalization)
vis_transform = transforms.ToTensor()

# --- Custom Dataset for SR ---
class MNIST_SR_Dataset(Dataset):
    def __init__(self, mnist_dataset, lr_transform, hr_transform, vis_transform):
        self.mnist_dataset = mnist_dataset
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.vis_transform = vis_transform # Store vis transform
        self.base_data = mnist_dataset.data
        self.base_targets = mnist_dataset.targets

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Use original uint8 tensor for consistency
        original_uint8_image = self.base_data[idx].unsqueeze(0) # Add channel dim for ToPILImage
        pil_image = transforms.ToPILImage()(original_uint8_image)

        lr_image = self.lr_transform(pil_image)
        hr_image_transformed = self.hr_transform(pil_image)
        hr_image_original_vis = self.vis_transform(pil_image) # Get original 0-1 HR for vis/metrics

        # Return LR, Normalized HR (for loss), Original 0-1 HR (for vis/metrics)
        return lr_image, hr_image_transformed, hr_image_original_vis

# --- Create Datasets and Loaders ---
base_train_dataset = datasets.MNIST(root='./data', train=True, download=True)
base_test_dataset = datasets.MNIST(root='./data', train=False, download=True)

# Pass vis_transform to dataset
train_dataset = MNIST_SR_Dataset(base_train_dataset, lr_transform, hr_transform, vis_transform)
test_dataset = MNIST_SR_Dataset(base_test_dataset, lr_transform, hr_transform, vis_transform)

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
        # Output activation (like Tanh or Sigmoid) might be beneficial for SR
        # if the target data is normalized to [-1, 1] or [0, 1] respectively.
        # Since we normalize with mean/std, a linear output is common.
        # self.output_activation = nn.Tanh() # Optional, depends on normalization

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten input LR image
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
            # if hasattr(self, 'output_activation'):
            #     prediction_flat = self.output_activation(prediction_flat)
            all_outputs.append(prediction_flat)

        # Returns a list of *flattened* predicted HR image tensors (normalized scale)
        # [pred_hr_flat_from_input_act, pred_hr_flat_from_h1_act, ..., pred_hr_flat_from_hN_act]
        return all_outputs

# 4. 实例化模型、损失函数和优化器
model = SR_MLPWithSharedOutputHead(input_dim, hidden_dim, output_dim, num_hidden_layers).to(device)
criterion = nn.MSELoss() # Loss calculated on normalized data
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练循环 (Use sum of all intermediate losses for backprop)
losses_history = [[] for _ in range(num_hidden_layers + 1)] # Store avg MSE loss from each layer per epoch
print("--- 开始训练 (使用所有中间层Loss之和进行梯度更新) ---")
for epoch in range(num_epochs):
    model.train()
    epoch_batch_losses = [[] for _ in range(num_hidden_layers + 1)] # Store individual batch losses for averaging
    for batch_idx, (lr_data, hr_target_norm, _) in enumerate(train_loader): # Ignore hr_original_vis during training
        lr_data, hr_target_norm = lr_data.to(device), hr_target_norm.to(device)
        hr_target_flat_norm = hr_target_norm.view(hr_target_norm.size(0), -1) # (N, output_dim)

        # --- Forward Pass ---
        all_predictions_flat_norm = model(lr_data) # List of flattened predictions (normalized scale)

        # --- Loss Calculation & Recording ---
        # Calculate the loss for each layer's prediction and sum them up
        # Use a clone for total_loss to ensure gradient calculation starts correctly
        total_loss = torch.tensor(0.0, device=device, requires_grad=False) # Initialize total loss for this batch

        for i in range(num_hidden_layers + 1):
            # Calculate MSE loss between the i-th prediction and the normalized target
            loss = criterion(all_predictions_flat_norm[i], hr_target_flat_norm)
            epoch_batch_losses[i].append(loss.item()) # Log individual loss value

            # Add this layer's loss to the total loss for backpropagation
            total_loss = total_loss + loss
            # Alternatively, create a list of losses and sum at the end:
            # layer_losses.append(loss)
        # total_loss = sum(layer_losses)

        # --- Backpropagation (using the combined loss) ---
        optimizer.zero_grad()
        # Need to enable gradients for total_loss if initialized with requires_grad=False
        # A simple way is to just sum the losses directly if they require grad.
        # Let's re-calculate total_loss ensuring it requires grad
        total_loss_for_grad = torch.tensor(0.0, device=device, requires_grad=True)
        for i in range(num_hidden_layers + 1):
             loss_for_grad = criterion(all_predictions_flat_norm[i], hr_target_flat_norm)
             total_loss_for_grad = total_loss_for_grad + loss_for_grad

        total_loss_for_grad.backward() # Backpropagate the sum of losses
        optimizer.step()

        # --- Logging ---
        if batch_idx % 100 == 0:
            # Report the total combined loss used for backprop
            print(f'Train Epoch: {epoch+1} [{batch_idx * len(lr_data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Total Combined MSE Loss (for backprop): {total_loss_for_grad.item():.6f}') # Log total loss

    # --- Epoch Summary ---
    avg_epoch_losses = [np.mean(batch_losses) if batch_losses else 0 for batch_losses in epoch_batch_losses]
    print(f"\nEpoch {epoch+1} Average Individual MSE Losses (Shared Head, Combined Loss Backprop):")
    loss_str = " | ".join([f"L{i+1}_MSE: {l:.6f}" for i, l in enumerate(avg_epoch_losses)]) # Format individual MSEs
    print(f"[{loss_str}]\n")
    print(f"Epoch {epoch+1} Average Total Combined MSE Loss: {np.sum(avg_epoch_losses):.6f}\n") # Also print the sum of averages

    # Store the history of average individual MSE losses
    for i in range(num_hidden_layers + 1):
        losses_history[i].append(avg_epoch_losses[i])

print("--- 训练完成 ---")

# 6. 绘图并保存 Loss 历史
plt.figure(figsize=(12, 7))
for i in range(num_hidden_layers + 1):
    label = f'MSE from Rep. after Input Layer Act.' if i == 0 else f'MSE from Rep. after Hidden Layer {i} Act.'
    plt.plot(range(1, num_epochs + 1), losses_history[i], label=label, marker='o', linestyle='-')

sum_avg_losses = [sum(epoch_losses) for epoch_losses in zip(*losses_history)]
plt.plot(range(1, num_epochs + 1), sum_avg_losses, label='Sum of Average MSEs (Total Loss Trend)', color='black', linestyle='--', linewidth=2)

plt.title('MNIST SR Training: Individual & Total MSE Loss History\n(Shared Head, Combined Intermediate Loss for Backprop)')
plt.xlabel('Epoch')
plt.ylabel('Average MSE Loss (Normalized Data)')
plt.xticks(range(1, num_epochs + 1))
plt.legend(fontsize='small', loc='upper right')
plt.grid(True)
plt.yscale('log')
loss_plot_path = os.path.join(results_dir, 'mnist_sr_loss_history_combined.png')
plt.savefig(loss_plot_path)
print(f"Loss history plot saved to: {loss_plot_path}")
plt.close()

# 7. 评估模型性能 (Evaluate based on final layer's output using normalized data)
model.eval()
test_mse_loss_final_layer_norm = 0
with torch.no_grad():
    for lr_data, hr_target_norm, _ in test_loader: # Ignore hr_original_vis
        lr_data, hr_target_norm = lr_data.to(device), hr_target_norm.to(device)
        hr_target_flat_norm = hr_target_norm.view(hr_target_norm.size(0), -1)

        all_outputs_flat_norm = model(lr_data)
        final_output_flat_norm = all_outputs_flat_norm[-1] # Evaluate using the prediction from the last representation

        test_mse_loss_final_layer_norm += criterion(final_output_flat_norm, hr_target_flat_norm).item()

# Average loss over number of batches
test_mse_loss_final_layer_norm /= len(test_loader)
print(f'\nTest set: Average MSE loss (final layer output, normalized data): {test_mse_loss_final_layer_norm:.6f}\n')


# 8. 单个样本可视化 (****** 修改部分 ******)
print("Visualizing single sample reconstruction with quality metrics...")
model.eval()

# --- 选择一个样本 ---
sample_idx = 5
print(f"Selected sample index from TEST SET: {sample_idx}")

# --- 获取模型输入 (LR), 归一化目标 (HR), 和 原始未归一化目标 (HR) ---
# Use the modified dataset __getitem__
lr_sample_tensor, _, hr_original_vis_tensor = test_dataset[sample_idx]
# lr_sample_tensor shape: (C, lr_h, lr_w)
# hr_original_vis_tensor shape: (C, hr_h, hr_w), range [0, 1]

# --- 预处理用于模型的输入 ---
lr_input_batch = lr_sample_tensor.unsqueeze(0).to(device) # (1, C, lr_h, lr_w)

# --- 模型推理 ---
with torch.no_grad():
    # Returns list of flattened, *normalized* predictions
    all_sample_outputs_flat_norm = model(lr_input_batch)

# --- 实例化指标计算器 ---
# Metrics are typically calculated on images in the [0, 1] range.
# data_range=1.0 assumes input tensors are scaled to [0, 1]
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
mae_metric = nn.L1Loss() # MAE (L1 Loss)

# --- 后处理预测结果 & 计算指标 ---
predicted_hr_images_denorm = [] # Store denormalized images for display
metrics_per_layer = [] # Store dict of metrics for each layer

# Prepare ground truth for metric calculation (needs batch and channel dim, on device)
# hr_original_vis_tensor is already [0, 1] range, shape (C, H, W)
gt_tensor_for_metrics = hr_original_vis_tensor.unsqueeze(0).to(device) # (1, C, H, W)

for i, output_flat_norm in enumerate(all_sample_outputs_flat_norm):
    # Reshape normalized prediction to image format (1, C, H, W)
    pred_img_norm = output_flat_norm.view(1, 1, hr_size, hr_size) # Assuming C=1 for MNIST

    # --- Denormalize prediction for visualization AND metrics ---
    # Clone to avoid modifying the original tensor if needed elsewhere
    pred_img_denorm_tensor = pred_img_norm.clone().cpu().squeeze() # (H, W)
    mean_val = norm_mean[0]
    std_val = norm_std[0]
    pred_img_denorm_tensor = pred_img_denorm_tensor * std_val + mean_val
    pred_img_denorm_tensor = torch.clamp(pred_img_denorm_tensor, 0, 1) # Clamp to [0, 1] range

    # Store denormalized numpy array for plotting
    predicted_hr_images_denorm.append(pred_img_denorm_tensor.numpy())

    # --- Calculate Metrics on Denormalized [0, 1] Images ---
    # Add batch and channel dimension back for torchmetrics
    pred_tensor_for_metrics = pred_img_denorm_tensor.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, H, W)

    # Ensure gt_tensor_for_metrics also has the channel dimension if missing (it should have it)
    if gt_tensor_for_metrics.dim() == 3: # If somehow shape is (1, H, W)
         gt_tensor_for_metrics_ = gt_tensor_for_metrics.unsqueeze(1) # Add channel dim -> (1, 1, H, W)
    else:
         gt_tensor_for_metrics_ = gt_tensor_for_metrics

    # Calculate metrics
    mse_val = F.mse_loss(pred_tensor_for_metrics, gt_tensor_for_metrics_).item()
    mae_val = mae_metric(pred_tensor_for_metrics, gt_tensor_for_metrics_).item()
    psnr_val = psnr_metric(pred_tensor_for_metrics, gt_tensor_for_metrics_).item()
    # SSIM needs slightly different input shape handling in some torchmetrics versions
    try:
        ssim_val = ssim_metric(pred_tensor_for_metrics, gt_tensor_for_metrics_).item()
    except Exception as e:
        print(f"Warning: SSIM calculation failed for layer {i}. Error: {e}")
        ssim_val = float('nan') # Assign NaN if SSIM fails


    metrics = {
        'MSE': mse_val,
        'MAE': mae_val,
        'PSNR': psnr_val,
        'SSIM': ssim_val
    }
    metrics_per_layer.append(metrics)

# --- 可视化 ---
num_layers_to_plot = num_hidden_layers + 1
num_total_plots = 2 + num_layers_to_plot
# Increase figure width slightly to accommodate metrics
fig, axes = plt.subplots(1, num_total_plots, figsize=(3.5 * num_total_plots, 4.5)) # Increased height

# Function to display image (already denormalized numpy)
def imshow_simple(ax, img_numpy, title):
    ax.imshow(img_numpy, cmap='gray', vmin=0, vmax=1) # Ensure range is 0-1
    ax.set_title(title, fontsize=8) # Smaller font for potentially long titles
    ax.axis('off')

# 1. 绘制 LR 输入图像 (denormalized and upscaled for comparison)
# Denormalize LR for display
lr_display_tensor = lr_sample_tensor.clone().cpu()
lr_display_tensor = lr_display_tensor * norm_std[0] + norm_mean[0]
lr_display_tensor = torch.clamp(lr_display_tensor, 0, 1)
# Upscale LR using bicubic interpolation
lr_display_upscaled = F.interpolate(lr_display_tensor.unsqueeze(0), size=(hr_size, hr_size), mode='bicubic', align_corners=False).squeeze()
imshow_simple(axes[0], lr_display_upscaled.numpy(), "Input LR\n(Upscaled Bicubic)")

# 2. 绘制 HR Ground Truth 图像 (original, unnormalized 0-1)
imshow_simple(axes[1], hr_original_vis_tensor.squeeze().numpy(), "Ground Truth HR\n(Original 0-1)")

# 3. 绘制每个表示层生成的 HR 预测图像 (denormalized) 并显示指标
for i in range(num_layers_to_plot):
    ax_pred = axes[i + 2]
    pred_img_denorm_numpy = predicted_hr_images_denorm[i] # Already denormalized numpy [0, 1]

    layer_name = f'Input Act.' if i == 0 else f'Hidden {i} Act.'
    metrics = metrics_per_layer[i]
    title = (f"Pred. from {layer_name}\n"
             f"MSE: {metrics['MSE']:.4f} MAE: {metrics['MAE']:.4f}\n"
             f"PSNR: {metrics['PSNR']:.2f} dB SSIM: {metrics['SSIM']:.4f}")

    imshow_simple(ax_pred, pred_img_denorm_numpy, title)

plt.tight_layout(pad=0.5, h_pad=1.5) # Adjust padding if needed

vis_plot_path = os.path.join(results_dir, f'sample_{sample_idx}_reconstruction_metrics_combined_loss.png')
plt.savefig(vis_plot_path)
print(f"Single sample visualization with metrics saved to: {vis_plot_path}")
plt.close()