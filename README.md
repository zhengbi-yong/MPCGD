# 环境安装

* Windows下载并安装miniconda
```
https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
```
* 创建虚拟环境并激活
```
conda create -y -n mpcgd python=3.10
conda activate mpcgd
```
* 安装pytorch
```
https://pytorch.org
```
! 在上述页面找到合适的pytorch版本，要先在命令行中或者PowerShell中运行`nvidia-smi`命令查看CUDA Version，然后在上述页面中选择低于CUDA Version的pytorch版本进行安装。
* 安装matplotlib
```
pip install matplotlib
```

* 在合适的目录下克隆本仓库
```
git clone https://github.com/zhengbi-yong/MPCGD.git
```
* 用合适的代码编辑器(例如VS code)打开项目文件夹
* 运行逼近sin(x)函数的程序
```
python 202504162145.py
```
* 观察results中生成的结果

# 文件与实验的对应表
202504162145.py —— 用MLP逼近sin(x)，每层loss都参与梯度更新，相当于多目标优化。
202504162241.py —— 用MLP逼近sin(x)，只有最后的loss参与梯度更新。