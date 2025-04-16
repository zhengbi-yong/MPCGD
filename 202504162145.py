# 2024年4月16日 21:45:00
# 验证多层感知机 (MLP) 中间层输出的有效性，逼近sin(x)函数


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
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S") # 例如 2025-04-16_13-58-00

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

# 3. 定义 MLP 模型
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

        # 为了方便理解，也可以只返回用于计算loss的投影输出
        # return intermediate_outputs + [final_output]
        # 如果需要访问原始隐藏层激活值，可以像这样返回：
        return all_outputs #, hidden_outputs_before_projection


# 4. 实例化模型、损失函数和优化器
model = MLPWithIntermediateOutput(input_dim, hidden_dim, output_dim, num_hidden_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练循环
# 存储每个输出层的 loss 历史记录
# 总共有 num_hidden_layers + 1 个输出需要计算 loss
losses_history = [[] for _ in range(num_hidden_layers + 1)]

for epoch in range(num_epochs):
    model.train() # 设置为训练模式

    # 前向传播，获取所有输出
    all_predictions = model(x_train) # 包含 num_hidden_layers+1 个预测

    # 计算每个输出的 loss 并累加总 loss
    total_loss = 0
    current_epoch_losses = [] # 临时存储当前 epoch 的各个 loss 值

    for i in range(num_hidden_layers + 1):
        loss = criterion(all_predictions[i], y_train)
        current_epoch_losses.append(loss.item()) # 记录标量 loss 值
        total_loss += loss # 累加 tensor loss 用于反向传播

    # 反向传播和优化
    optimizer.zero_grad()       # 清空梯度
    total_loss.backward()       # 反向传播计算梯度
    optimizer.step()            # 更新模型参数

    # 记录当前 epoch 的所有 loss 值
    for i in range(num_hidden_layers + 1):
        losses_history[i].append(current_epoch_losses[i])

    # 打印训练信息
    if (epoch + 1) % 500 == 0:
        loss_str = " | ".join([f"L{i+1}: {l:.6f}" for i, l in enumerate(current_epoch_losses)])
        print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item():.6f} | Individual Losses: [{loss_str}]')


# 6. 绘图并保存 Loss 历史
plt.figure(figsize=(12, 6))
for i in range(num_hidden_layers + 1):
    label = f'Hidden Layer {i+1} Output Loss' if i < num_hidden_layers else 'Final Output Loss'
    plt.plot(losses_history[i], label=label)

plt.title('Training Loss History for Intermediate and Final Outputs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.grid(True)
plt.yscale('log') # 使用对数刻度可能更容易观察 loss 下降

# --- 保存 Loss 历史图 ---
loss_plot_path = os.path.join(results_dir, 'loss_history.png')
plt.savefig(loss_plot_path)
print(f"Loss history plot saved to: {loss_plot_path}")
# -------------------------

# 7. 可视化逼近效果并保存
model.eval() # 设置为评估模式
with torch.no_grad(): # 关闭梯度计算
    # 为了更平滑的曲线，可以生成更多的测试点
    x_test_np = np.linspace(-2.5 * math.pi, 2.5 * math.pi, 1000)
    y_test_np = np.sin(x_test_np)
    x_test = torch.tensor(x_test_np, dtype=torch.float32).unsqueeze(1)
    # 获取所有层的预测输出
    all_preds_test = model(x_test) # all_preds_test 是一个列表，包含 num_hidden_layers + 1 个 tensor
    final_pred_test = all_preds_test[-1] # 最后一个是最终预测

plt.figure(figsize=(12, 7)) # 稍微调大图形尺寸以容纳更多图例
plt.plot(x_test_np, y_test_np, label='True sin(x)', linestyle='--', linewidth=2, color='black')

# --- 绘制中间层的逼近效果 ---
# all_preds_test 包含 num_hidden_layers+1 个输出，索引 0 到 num_hidden_layers-1 是中间层输出
for i in range(num_hidden_layers):
    plt.plot(x_test_np, all_preds_test[i].numpy(), label=f'Hidden Layer {i+1} Approx.', alpha=0.6, linestyle=':')

# --- 绘制最终层的逼近效果 ---
plt.plot(x_test_np, final_pred_test.numpy(), label='MLP Final Approximation', color='red', linewidth=1.5)


plt.title('Function Approximation: sin(x) by All Layers') # 更新标题
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right') # 调整图例位置避免遮挡
plt.grid(True)
plt.ylim(-1.5, 1.5) # 设置 y 轴范围更聚焦

# --- 保存包含所有层逼近效果的函数逼近图 ---
approx_plot_path = os.path.join(results_dir, 'function_approximation_all_layers.png') # 修改文件名
plt.savefig(approx_plot_path)
print(f"Function approximation plot (all layers) saved to: {approx_plot_path}")
# ------------------------------------------