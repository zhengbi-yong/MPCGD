# 2025年4月16日 22:41:00
# 验证多层感知机 (MLP) 中间层输出的有效性，逼近sin(x)函数
# 修改：仅使用最终层 loss 进行梯度更新，但记录所有层 loss

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import os # 导入 os 模块
import datetime # 导入 datetime 模块

# --- 创建结果文件夹 ---
# 获取当前时间并格式化为字符串
now = datetime.datetime.now()
# 注意：根据当前时间，年份应为 2025
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S") # 例如 2025-04-16_14-10-00

# 定义结果文件夹路径
results_base_dir = "results"
results_dir = os.path.join(results_base_dir, timestamp_str)

# 创建文件夹 (如果不存在)
os.makedirs(results_dir, exist_ok=True)
print(f"结果将保存在: {results_dir}")
# ---------------------

# 1. 超参数设置
input_dim = 1         # 输入维度 (x)
output_dim = 1        # 输出维度 (sin(x))
hidden_dim = 64       # 隐藏层维度
num_hidden_layers = 4 # ****** 隐藏层的数量 (总共 5 层 MLP = 1 输入 + 4 隐藏 + 1 输出) ******
num_epochs = 5000     # 训练轮数
learning_rate = 1e-3  # 学习率
num_samples = 500     # 训练样本数量

# 确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 2. 数据准备
# 生成 x 在 [-2π, 2π] 范围内的数据点
x_train_np = np.linspace(-2 * math.pi, 2 * math.pi, num_samples)
y_train_np = np.sin(x_train_np)

# 转换为 PyTorch Tensors
# 需要增加一个维度以匹配模型的输入/输出 (batch_size, features)
x_train = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)

# 3. 定义 MLP 模型 (与之前相同)
class MLPWithIntermediateOutput(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(MLPWithIntermediateOutput, self).__init__()
        self.num_hidden_layers = num_hidden_layers

        # 输入层
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh() # Tanh 激活函数比较适合逼近 sin

        # 动态创建隐藏层
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers -1): # 第一个隐藏层在input_layer后，所以这里是 n-1 个
             self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 为每个隐藏层和最终层创建输出头 (projection head)
        # 总共有 num_hidden_layers 个隐藏层 + 1 个最终输出层 = num_hidden_layers + 1 个输出
        self.output_heads = nn.ModuleList()
        for _ in range(num_hidden_layers + 1):
            self.output_heads.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        intermediate_outputs = [] # 存储中间层的输出（经过投影头之后）
        hidden_outputs_before_projection = [] # 存储投影前的隐藏层激活值

        # Input Layer
        h = self.activation(self.input_layer(x))
        hidden_outputs_before_projection.append(h)
        intermediate_outputs.append(self.output_heads[0](h)) # 第一个隐藏层的输出预测

        # Hidden Layers
        for i in range(self.num_hidden_layers -1):
             h = self.activation(self.hidden_layers[i](h))
             hidden_outputs_before_projection.append(h)
             intermediate_outputs.append(self.output_heads[i+1](h)) # 后续隐藏层的输出预测

        # 最后一个隐藏层的输出经过最后一个投影头得到最终预测
        final_output = self.output_heads[self.num_hidden_layers](h) # 使用最后一个隐藏层的激活值

        # 返回所有中间预测和最终预测
        # 确保返回顺序是从第一个隐藏层到最后一个（最终）输出
        all_outputs = intermediate_outputs + [final_output]
        return all_outputs


# 4. 实例化模型、损失函数和优化器 (与之前相同)
model = MLPWithIntermediateOutput(input_dim, hidden_dim, output_dim, num_hidden_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练循环 (****** 修改部分 ******)
# 存储每个输出层的 loss 历史记录
losses_history = [[] for _ in range(num_hidden_layers + 1)]

for epoch in range(num_epochs):
    model.train() # 设置为训练模式

    # 前向传播，获取所有输出
    all_predictions = model(x_train) # 包含 num_hidden_layers+1 个预测

    # 计算每个输出的 loss 用于记录
    current_epoch_losses = [] # 临时存储当前 epoch 的各个 loss 值
    final_loss_tensor = None # 用于存储最终层的 Tensor loss

    for i in range(num_hidden_layers + 1):
        # 计算当前层的预测与目标之间的 loss
        loss = criterion(all_predictions[i], y_train)
        # 记录标量 loss 值
        current_epoch_losses.append(loss.item())
        # 如果是最后一层 (final output), 保存其 Tensor loss 用于反向传播
        if i == num_hidden_layers:
            final_loss_tensor = loss

    # --- 修改点：只使用最终层的 loss 进行反向传播 ---
    optimizer.zero_grad()       # 清空梯度
    if final_loss_tensor is not None: # 确保 final_loss_tensor 已被赋值
        final_loss_tensor.backward() # **只对最终层的 loss 调用 backward()**
    else:
        # 理论上不应发生，加个错误处理
        raise ValueError("Final loss tensor was not calculated.")
    optimizer.step()            # 更新模型参数 (会更新所有参与计算图的参数)
    # ------------------------------------------

    # 记录当前 epoch 的所有 loss 值 (这部分不变)
    for i in range(num_hidden_layers + 1):
        losses_history[i].append(current_epoch_losses[i])

    # 打印训练信息 (使用最终层的 loss 作为代表打印，或仍打印所有 loss)
    if (epoch + 1) % 500 == 0:
        loss_str = " | ".join([f"L{i+1}: {l:.6f}" for i, l in enumerate(current_epoch_losses)])
        # 打印最终层的loss 或 所有loss，这里选择打印所有loss以便观察
        print(f'Epoch [{epoch+1}/{num_epochs}], Final Layer Loss (for backprop): {final_loss_tensor.item():.6f} | All Recorded Losses: [{loss_str}]')


# 6. 绘图并保存 Loss 历史 (与之前相同)
plt.figure(figsize=(12, 6))
for i in range(num_hidden_layers + 1):
    label = f'Hidden Layer {i+1} Output Loss (Recorded)' if i < num_hidden_layers else 'Final Output Loss (Used for Backprop)'
    plt.plot(losses_history[i], label=label)

plt.title('Training Loss History (Only Final Loss for Backprop)') # 更新标题
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.grid(True)
plt.yscale('log')

loss_plot_path = os.path.join(results_dir, 'loss_history_final_backprop.png') # 修改文件名
plt.savefig(loss_plot_path)
print(f"Loss history plot saved to: {loss_plot_path}")

# 7. 可视化逼近效果并保存 (与之前相同)
model.eval() # 设置为评估模式
with torch.no_grad():
    x_test_np = np.linspace(-2.5 * math.pi, 2.5 * math.pi, 1000)
    y_test_np = np.sin(x_test_np)
    x_test = torch.tensor(x_test_np, dtype=torch.float32).unsqueeze(1)
    all_preds_test = model(x_test)
    final_pred_test = all_preds_test[-1]

plt.figure(figsize=(12, 7))
plt.plot(x_test_np, y_test_np, label='True sin(x)', linestyle='--', linewidth=2, color='black')

for i in range(num_hidden_layers):
    plt.plot(x_test_np, all_preds_test[i].numpy(), label=f'Hidden Layer {i+1} Approx. (Not used in backprop)', alpha=0.6, linestyle=':')

plt.plot(x_test_np, final_pred_test.numpy(), label='MLP Final Approximation (Used in backprop)', color='red', linewidth=1.5)

plt.title('Function Approximation: sin(x) (Only Final Loss for Backprop)') # 更新标题
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.grid(True)
plt.ylim(-1.5, 1.5)

approx_plot_path = os.path.join(results_dir, 'function_approximation_final_backprop.png') # 修改文件名
plt.savefig(approx_plot_path)
print(f"Function approximation plot saved to: {approx_plot_path}")