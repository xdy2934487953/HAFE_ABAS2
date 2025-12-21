# 依赖管理指南

## 📦 环境设置

### 1. 安装所有依赖

```bash
# 安装所有必需的Python包
pip install -r requirements.txt
```

### 2. 检查安装状态

```bash
# 运行依赖检查脚本
python check_dependencies.py
```

## 📋 依赖说明

### 必需依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.0.1 | PyTorch深度学习框架 |
| torch-geometric | 2.3.1 | 图神经网络库 |
| transformers | 4.30.0 | Hugging Face NLP模型 |
| stanza | 1.5.0 | 高级NLP处理 |
| numpy | 1.24.3 | 数值计算 |
| scipy | >=1.10.0 | 科学计算 |
| pandas | >=1.5.0 | 数据处理 |
| scikit-learn | 1.3.0 | 机器学习算法 |
| matplotlib | >=3.6.0 | 数据可视化 |
| seaborn | >=0.12.0 | 统计图表 |
| lxml | 4.9.2 | XML/HTML处理 |
| tqdm | 4.65.0 | 进度条显示 |

### 可选依赖

| 包名 | 用途 |
|------|------|
| torchtext | 文本处理扩展 |
| torchvision | 图像处理 |
| torchaudio | 音频处理 |

## 🔧 常见问题解决

### CUDA相关问题

如果您的系统有NVIDIA GPU：

```bash
# 安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 然后安装其他依赖
pip install -r requirements.txt
```

### 版本冲突

如果遇到版本冲突：

```bash
# 创建新的虚拟环境
python -m venv absa_env
absa_env\Scripts\activate  # Windows
# source absa_env/bin/activate  # Linux/Mac

# 在新环境中安装依赖
pip install -r requirements.txt
```

### 国内网络问题

如果pip下载速度慢：

```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 或使用阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 🚀 快速开始

### 1. 环境检查

```bash
# 检查所有依赖
python check_dependencies.py
```

### 2. 运行测试

```bash
# 测试DIB模块
python -c "from src.disentangled_information_bottleneck import test_dib_module; test_dib_module()"

# 测试简化版模型
python test_simplified_quick.py
```

### 3. 开始训练

```bash
# 训练简化版Causal-HAFE
python train_simplified.py --model simplified_causal_hafe --dataset semeval2014

# 训练原版Causal-HAFE
python train_causal.py --model causal_hafe --dataset semeval2014
```

## 📊 系统要求

### 最低要求
- **Python**: 3.8+
- **RAM**: 8GB+
- **磁盘**: 10GB+

### 推荐配置
- **Python**: 3.9-3.11
- **RAM**: 16GB+
- **GPU**: NVIDIA GPU (可选，但推荐用于训练)

## 🔍 依赖检查输出说明

运行 `python check_dependencies.py` 后的输出示例：

```
系统信息:
--------------------
Python版本: 3.12.7
CUDA可用: True
CUDA版本: 11.8
GPU数量: 1
GPU 0: NVIDIA RTX 3080 (12.0 GB)

============================================================
ABSA项目依赖检查
============================================================
检查必需依赖:
----------------------------------------
[OK] torch 2.0.1
[OK] torch-geometric 2.3.1
[OK] transformers 4.30.0
[OK] stanza 1.5.0
[OK] numpy 1.24.3
[OK] scipy 1.10.0
[OK] pandas 1.5.0
[OK] scikit-learn 1.3.0
[OK] matplotlib 3.6.0
[OK] seaborn 0.12.0
[OK] lxml 4.9.2
[OK] tqdm 4.65.0

必需依赖: 12/12 个包正常

[SUCCESS] 所有必需依赖已正确安装！

您可以运行以下命令开始训练:
python train_simplified.py --model simplified_causal_hafe --dataset semeval2014
```

## 📝 手动安装特定包

如果自动安装失败，可以手动安装：

```bash
# PyTorch (CPU版本)
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric
pip install torch-geometric==2.3.1

# 其他包
pip install transformers==4.30.0 stanza==1.5.0 numpy==1.24.3
pip install scipy pandas scikit-learn matplotlib seaborn lxml tqdm
```

## 🎯 验证安装

安装完成后，运行以下命令验证：

```bash
python -c "
import torch
import torch_geometric
from transformers import BertTokenizer
import stanza
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
print('All dependencies imported successfully!')
"
```

## 💡 提示

1. **虚拟环境**: 建议使用conda或venv创建虚拟环境
2. **版本锁定**: requirements.txt中的版本是经过测试的推荐版本
3. **更新依赖**: 定期检查并更新到最新稳定版本
4. **兼容性**: 如果遇到兼容性问题，可以适当调整版本号

如果遇到任何依赖相关的问题，请运行 `python check_dependencies.py` 查看详细状态，然后参考上述解决方案。
