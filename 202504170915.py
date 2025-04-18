# 2025年4月17日
# 使用 MLP 进行 MNIST 超分辨率 (LR -> HR)，监控中间层 Loss (通过共享最终输出头计算MSE)，仅使用最终 Loss 更新梯度
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
results_base_dir = "results_mnist_sr_shared_head" # 修改文件夹基础名
results_dir = os.path.join(results_base_dir, timestamp_str)
results_base_dir = "results"
results_dir = "mnist_sr_shared_head_final_loss"
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
hidden_dim = 256     # Hidden layer dimension (might need more capacity for SR)
num_hidden_layers = 4 # Number of hidden layers
num_epochs = 30       # More epochs might be needed for SR
batch_size = 128
learning_rate = 1e-3

# 确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 2. 数据准备 (MNIST SR)

# --- Transforms ---
# Normalize parameters (same as before for MNIST)
norm_mean = (0.1307,)
norm_std = (0.3081,)

# HR transform (Normalization)
hr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# LR transform (Resize to LR, then ToTensor, then Normalize)
lr_transform = transforms.Compose([
    transforms.Resize((lr_size, lr_size), interpolation=InterpolationMode.BICUBIC), # Downsample
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std) # Use same normalization as HR
])

# Transform for visualization (just ToTensor, no normalization)
vis_transform = transforms.ToTensor()

# --- Custom Dataset for SR ---
class MNIST_SR_Dataset(Dataset):
    def __init__(self, mnist_dataset, lr_transform, hr_transform):
        self.mnist_dataset = mnist_dataset
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        # Store the base data directly if possible (e.g., for PIL access)
        # Or rely on mnist_dataset structure
        self.base_data = mnist_dataset.data
        self.base_targets = mnist_dataset.targets

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Get the original image data (usually uint8 tensor)
        # We need the PIL image to apply transforms correctly
        pil_image = transforms.ToPILImage()(self.base_data[idx]) # Convert uint8 tensor to PIL

        # Create LR image by applying LR transform to the PIL image
        lr_image = self.lr_transform(pil_image)

        # Create HR target image by applying HR transform to the PIL image
        hr_image_transformed = self.hr_transform(pil_image) # Apply HR transform separately

        return lr_image, hr_image_transformed

# --- Create Datasets and Loaders ---
# Load the base MNIST dataset (provides original images)
base_train_dataset = datasets.MNIST(root='./data', train=True, download=True)
base_test_dataset = datasets.MNIST(root='./data', train=False, download=True) # Use test set for evaluation

# Wrap the base dataset with our SR dataset structure
train_dataset = MNIST_SR_Dataset(base_train_dataset, lr_transform, hr_transform)
test_dataset = MNIST_SR_Dataset(base_test_dataset, lr_transform, hr_transform) # Use base_test_dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# For visualization, we need the original unnormalized image from the *same* dataset used for testing
# We will get this directly from base_test_dataset later

# 3. 定义 MLP 模型 for SR
class SR_MLPWithSharedOutputHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(SR_MLPWithSharedOutputHead, self).__init__()
        self.activation = nn.ReLU()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        # The shared head maps hidden representations to the flattened HR image space
        self.final_output_head = nn.Linear(hidden_dim, output_dim)
        # Optional: Add Tanh activation to output layer if pixels are normalized to [-1, 1]
        # If normalized to [0, 1] or other ranges, might not need/want this.
        # self.output_activation = nn.Tanh() # Example if needed

    def forward(self, x):
        # Input x is expected to be the LR image batch (N, C, H_lr, W_lr)
        x = x.view(x.size(0), -1) # Flatten LR input: (N, input_dim)

        representations_to_evaluate = []
        h = self.activation(self.input_layer(x))
        representations_to_evaluate.append(h) # Representation after Input Layer Activation

        current_h = h
        for hidden_layer in self.hidden_layers:
            current_h = self.activation(hidden_layer(current_h))
            representations_to_evaluate.append(current_h) # Rep. after Hidden Layer 1, 2, ... N Activation

        all_outputs = []
        # Use the *shared* head to get predictions (flattened HR images) from each representation
        for h_repr in representations_to_evaluate:
            prediction_flat = self.final_output_head(h_repr)
            # Apply output activation if defined
            # prediction_flat = self.output_activation(prediction_flat)
            all_outputs.append(prediction_flat)

        # Returns a list of *flattened* predicted HR image tensors
        # [pred_hr_flat_from_input_act, pred_hr_flat_from_h1_act, ..., pred_hr_flat_from_hN_act]
        return all_outputs


# 4. 实例化模型、损失函数和优化器
model = SR_MLPWithSharedOutputHead(input_dim, hidden_dim, output_dim, num_hidden_layers).to(device)
criterion = nn.MSELoss() # Use Mean Squared Error for image reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练循环 (Focus on final loss for backprop)
losses_history = [[] for _ in range(num_hidden_layers + 1)] # Store avg MSE loss from each layer per epoch
print("--- 开始训练 ---")
for epoch in range(num_epochs):
    model.train()
    epoch_batch_losses = [[] for _ in range(num_hidden_layers + 1)] # Store individual batch losses for averaging
    for batch_idx, (lr_data, hr_target) in enumerate(train_loader):
        lr_data, hr_target = lr_data.to(device), hr_target.to(device)

        # Flatten the HR target image to match the model's output shape for loss calculation
        hr_target_flat = hr_target.view(hr_target.size(0), -1) # (N, output_dim)

        # --- Forward Pass ---
        # Get list of flattened predicted HR images from all specified layers
        all_predictions_flat = model(lr_data)

        # --- Loss Calculation & Recording ---
        final_loss_tensor = None
        # The length of all_predictions_flat is num_hidden_layers + 1
        for i in range(num_hidden_layers + 1):
            # Calculate MSE loss between the i-th prediction and the target
            loss = criterion(all_predictions_flat[i], hr_target_flat)
            epoch_batch_losses[i].append(loss.item()) # Log individual loss value

            # Store the loss from the *final* representation for backpropagation
            if i == num_hidden_layers:
                final_loss_tensor = loss

        # --- Backpropagation (using only final loss) ---
        optimizer.zero_grad()
        if final_loss_tensor is not None:
            final_loss_tensor.backward()
        else:
             raise ValueError("Final loss tensor was not calculated.")
        optimizer.step()

        # --- Logging ---
        if batch_idx % 100 == 0:
             print(f'Train Epoch: {epoch+1} [{batch_idx * len(lr_data)}/{len(train_loader.dataset)} '
                   f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                   f'Final Rep. MSE Loss (for backprop): {final_loss_tensor.item():.6f}')

    # --- Epoch Summary ---
    avg_epoch_losses = [np.mean(batch_losses) if len(batch_losses) > 0 else 0 for batch_losses in epoch_batch_losses]
    print(f"\nEpoch {epoch+1} Average MSE Losses (Shared Head, Final Loss Backprop):")
    loss_str = " | ".join([f"L{i+1}_MSE: {l:.6f}" for i, l in enumerate(avg_epoch_losses)]) # Format MSE
    print(f"[{loss_str}]\n")

    # Store the history of average individual MSE losses
    for i in range(num_hidden_layers + 1):
        losses_history[i].append(avg_epoch_losses[i])

print("--- 训练完成 ---")

# 6. 绘图并保存 Loss 历史
plt.figure(figsize=(12, 6))
for i in range(num_hidden_layers + 1):
    label = f'MSE from Rep. after Input Layer Act.' if i == 0 else f'MSE from Rep. after Hidden Layer {i} Act.'
    if i == num_hidden_layers: label += ' (Used for Backprop)'
    else: label += ' (Recorded Only, Shared Head)'
    plt.plot(range(1, num_epochs + 1), losses_history[i], label=label, marker='o', linestyle='-')
plt.title('MNIST SR Training MSE Loss History (Shared Output Head, Only Final Loss for Backprop)')
plt.xlabel('Epoch')
plt.ylabel('Average MSE Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend(fontsize='small')
plt.grid(True)
# MSE loss might not decrease as sharply as CE, linear scale might be okay, but log can highlight early changes
plt.yscale('log') # Or 'linear'
loss_plot_path = os.path.join(results_dir, 'mnist_sr_loss_history_shared_head.png')
plt.savefig(loss_plot_path)
print(f"Loss history plot saved to: {loss_plot_path}")
plt.close() # 关闭图形，释放内存

# 7. 评估模型性能
model.eval()
test_mse_loss = 0 # Based on final layer output
with torch.no_grad():
    for lr_data, hr_target in test_loader:
        lr_data, hr_target = lr_data.to(device), hr_target.to(device)
        hr_target_flat = hr_target.view(hr_target.size(0), -1)

        all_outputs_flat = model(lr_data)
        final_output_flat = all_outputs_flat[-1] # Evaluate using the prediction from the last representation

        test_mse_loss += criterion(final_output_flat, hr_target_flat).item()

# Average loss over number of batches
test_mse_loss /= len(test_loader)
print(f'\nTest set: Average MSE loss (final layer): {test_mse_loss:.6f}\n')
# Note: Lower MSE is better.


# 8. 单个样本可视化 (****** 修改部分 ******)
print("Visualizing single sample reconstruction...")
model.eval() # 确保模型在评估模式

# --- 选择一个样本 ---
sample_idx = 5 # 可以修改这个索引来查看不同的样本
print(f"Selected sample index from TEST SET: {sample_idx}")

# --- 获取模型输入 (LR) 和 归一化目标 (HR) ---
# 从 test_dataset 获取，确保了 LR 和 HR 对应
lr_sample_tensor, hr_sample_tensor_norm = test_dataset[sample_idx] # Get normalized versions for model input/comparison

# --- 获取原始未归一化的 HR 图像用于可视化 ---
# *** 修改: 从 base_test_dataset 获取原始图像 ***
# 1. 获取原始数据（通常是 uint8 tensor）
original_uint8_tensor = base_test_dataset.data[sample_idx]
# 2. 转换为 PIL Image
original_pil_image = transforms.ToPILImage()(original_uint8_tensor)
# 3. 应用可视化变换 (ToTensor) 得到 [0, 1] 范围的 FloatTensor
hr_original_tensor = vis_transform(original_pil_image)
# 现在 hr_original_tensor 是未归一化的、与 lr_sample_tensor 对应的 HR 图像
# --- 结束修改 ---

# --- 预处理用于模型的输入 ---
lr_input_batch = lr_sample_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

# --- 模型推理 ---
with torch.no_grad():
    # Get list of *flattened* predicted HR images
    all_sample_outputs_flat = model(lr_input_batch)

# --- 后处理预测结果 ---
predicted_hr_images = []
for output_flat in all_sample_outputs_flat:
    # Reshape the flattened output back to image dimensions (1, H_hr, W_hr)
    pred_img = output_flat.view(1, hr_size, hr_size).squeeze().cpu().numpy() # Remove channel=1 dim, move to CPU, convert to numpy
    predicted_hr_images.append(pred_img)

# --- 可视化 ---
num_layers_to_plot = num_hidden_layers + 1
# Create subplots: 1 row, 2 fixed images (LR, HR Ground Truth) + N+1 predicted images
num_total_plots = 2 + num_layers_to_plot
fig, axes = plt.subplots(1, num_total_plots, figsize=(3 * num_total_plots, 3.5))

# Function to handle displaying normalized images (approximate denormalization for vis)
# Note: Simple denormalization might not be perfect but is good for visualization
def imshow_denormalize(ax, img_tensor, title):
    img = img_tensor.cpu().squeeze().numpy() # Ensure tensor is on CPU, remove channel dim, convert to numpy
    # Denormalize: img * std + mean
    mean_val = norm_mean[0]
    std_val = norm_std[0]
    img = img * std_val + mean_val
    img = np.clip(img, 0, 1) # Clip values to be in valid range [0, 1] for imshow
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=9)
    ax.axis('off')

# 1. 绘制 LR 输入图像 (denormalized)
# 使用插值将 LR 图像放大以进行视觉比较
lr_display = F.interpolate(lr_sample_tensor.unsqueeze(0), size=(hr_size, hr_size), mode='bicubic', align_corners=False).squeeze(0)
imshow_denormalize(axes[0], lr_display, "Input LR\n(Upscaled Bicubic)")

# 2. 绘制 HR Ground Truth 图像 (original, unnormalized)
# *** 修改：直接使用我们加载的 hr_original_tensor ***
axes[1].imshow(hr_original_tensor.squeeze(), cmap='gray') # Display the unnormalized [0,1] tensor
axes[1].set_title("Ground Truth HR", fontsize=9)
axes[1].axis('off')
# --- 结束修改 ---

# 3. 绘制每个表示层生成的 HR 预测图像 (denormalized)
for i in range(num_layers_to_plot):
    ax_pred = axes[i + 2] # Start plotting predictions from the 3rd subplot
    pred_img_numpy = predicted_hr_images[i] # This is already numpy array

    # Denormalize the predicted image (which corresponds to normalized target)
    mean_val = norm_mean[0]
    std_val = norm_std[0]
    pred_img_denorm = pred_img_numpy * std_val + mean_val
    pred_img_denorm = np.clip(pred_img_denorm, 0, 1)

    layer_name = f'Input Act.' if i == 0 else f'Hidden {i} Act.'
    title = f"Pred. from {layer_name}"

    # Calculate MSE using the *flattened* tensors (model output vs normalized target)
    with torch.no_grad():
        # all_sample_outputs_flat[i] is the raw flattened output from the model for this layer
        # Shape [1, output_dim] on device
        prediction_tensor_flat = all_sample_outputs_flat[i].view(-1) # Ensure shape [output_dim]

        # hr_sample_tensor_norm is the normalized ground truth from test_dataset
        # Shape [1, hr_size, hr_size], needs flatten and device
        target_tensor_flat = hr_sample_tensor_norm.view(-1).to(device) # Ensure shape [output_dim] and on device

        # Calculate MSE between the two flattened tensors
        mse_val = F.mse_loss(prediction_tensor_flat, target_tensor_flat).item()

    title += f"\nMSE: {mse_val:.4f}" # Add the correctly calculated MSE to the title

    ax_pred.imshow(pred_img_denorm, cmap='gray')
    ax_pred.set_title(title, fontsize=9)
    ax_pred.axis('off')


plt.tight_layout(pad=0.5) # Adjust layout

# 保存可视化结果图
vis_plot_path = os.path.join(results_dir, f'sample_{sample_idx}_reconstruction_shared_head.png')
plt.savefig(vis_plot_path)
print(f"Single sample visualization saved to: {vis_plot_path}")
plt.close() # 关闭图形