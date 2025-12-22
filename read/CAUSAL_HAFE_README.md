# Causal-HAFE: 基于因果解耦与虚假相关消除的公平性ABSA

## 概述

Causal-HAFE是原HAFE-ABSA框架的因果推断增强版本,通过以下三个核心模块解决方面级情感分析中的频率偏差问题:

1. **Deconfounded GAT (去混淆图注意力层)**: 基于后门调整消除上下文混淆因子
2. **DIB (解耦信息瓶颈)**: 将特征分解为因果部分(Z_c)和虚假部分(Z_s)
3. **TIE (总间接效应)**: 反事实推理消除先验偏差

## 新增文件

```
src/
├── deconfounded_gat.py              # 去混淆图注意力层
├── disentangled_information_bottleneck.py  # DIB模块
├── counterfactual_inference.py      # 反事实推理
├── causal_hafe.py                   # Causal-HAFE主模型
train_causal.py                      # Causal-HAFE训练脚本
```

## 快速开始

### 基础训练

```bash
# 训练Causal-HAFE模型 (SemEval-2014)
python train_causal.py --dataset semeval2014 --model causal_hafe --epochs 50

# 使用TIE推理
python train_causal.py --dataset semeval2014 --model causal_hafe --epochs 50 --use_tie_inference
```

### 训练Baseline对比

```bash
# 训练消融基线 (无因果模块)
python train_causal.py --dataset semeval2014 --model baseline --epochs 50
```

## 参数说明

### 模型架构参数

- `--causal_dim`: 因果表示维度 (默认: 128)
- `--spurious_dim`: 虚假表示维度 (默认: 64)
- `--num_confounders`: 混淆因子原型数量 (默认: 5)
- `--gat_heads`: GAT注意力头数 (默认: 1)

### DIB损失权重

- `--lambda_indep`: 解耦约束权重 (默认: 0.1)
- `--lambda_bias`: 偏差拟合权重 (默认: 0.5)
- `--lambda_ib`: 信息瓶颈权重 (默认: 0.01)

### 其他参数

- `--num_frequency_buckets`: 频率分桶数 (默认: 5)
- `--use_tie_inference`: 最终评估时使用TIE反事实推理

## 训练示例

### 1. 标准Causal-HAFE训练

```bash
python train_causal.py \
    --dataset semeval2014 \
    --model causal_hafe \
    --causal_dim 128 \
    --spurious_dim 64 \
    --num_confounders 5 \
    --lambda_indep 0.1 \
    --lambda_bias 0.5 \
    --lambda_ib 0.01 \
    --epochs 50 \
    --lr 0.001
```

### 2. 使用TIE推理的完整训练

```bash
python train_causal.py \
    --dataset semeval2014 \
    --model causal_hafe \
    --epochs 50 \
    --use_tie_inference
```

### 3. 消融实验 - Baseline

```bash
python train_causal.py \
    --dataset semeval2014 \
    --model baseline \
    --epochs 50
```

### 4. SemEval-2016数据集

```bash
python train_causal.py \
    --dataset semeval2016 \
    --model causal_hafe \
    --epochs 50 \
    --use_tie_inference
```

## 模型架构详解

### 1. 前向传播流程

```
Input (BERT features)
    ↓
[F3模块] 特征增强
    ↓
[DIB模块] 特征解耦 → Z_c (因果) + Z_s (虚假)
    ↓
[Deconfounded GAT] 去混淆图传播 (在Z_c上)
    ↓
[分类器] 情感预测
```

### 2. 损失函数

```
L_total = L_task + λ1·L_indep + λ2·L_bias + λ3·L_IB

其中:
- L_task: 情感分类损失
- L_indep: I(Z_c; Z_s) 最小化互信息
- L_bias: 强制Z_s预测频率分桶
- L_IB: I(Z_c; X) 信息瓶颈
```

### 3. 反事实推理 (TIE)

```
TIE = Logits(A, R) - Logits(A, ∅)

- Logits(A, R): 完整输入的预测
- Logits(A, ∅): 屏蔽上下文后的预测
```

## 评估指标

### 准确性指标

- Accuracy: 整体准确率
- Macro-F1: 宏平均F1
- Micro-F1: 微平均F1

### 公平性指标

- **Variance**: Per-aspect F1的方差
- **Gap**: 最大F1与最小F1的差距
- **Gini Coefficient**: 基尼系数
- **DP-Aspect**: 高频vs低频aspect的性能差异
  - 高频: Top 25%
  - 低频: Bottom 25%

## 预期结果

根据文献,Causal-HAFE预期在以下方面优于基线:

1. **总体性能**: Macro-F1与RoBERTa/DualGCN持平或略高
2. **低频aspect**: Tail分组F1提升10-15%
3. **公平性**: DP-Aspect显著降低 (更公平)
4. **鲁棒性**: 在对抗数据集ARTS上表现更稳定

## 模块测试

每个模块都可以独立测试:

```bash
# 测试DIB模块
python src/disentangled_information_bottleneck.py

# 测试反事实推理
python src/counterfactual_inference.py

# 测试完整模型
python src/causal_hafe.py
```

## 与原HAFE的对比

| 特性 | 原HAFE | Causal-HAFE |
|------|--------|-------------|
| 图传播 | Type-Aware GCN | Deconfounded GAT |
| 特征表示 | 单一表示 | 解耦表示(Z_c + Z_s) |
| 训练目标 | 单任务损失 | 多任务损失(DIB) |
| 推理方式 | 直接预测 | TIE反事实推理 |
| 公平性 | 基础公平性增强 | 显式因果去偏 |

## 故障排除

### 内存不足

如果遇到内存问题,尝试:
- 减少`causal_dim`和`spurious_dim`
- 减少`num_confounders`
- 减少`gat_heads`

### 训练不稳定

如果损失震荡:
- 降低学习率 `--lr 0.0001`
- 调整DIB权重 (减小`lambda_indep`和`lambda_ib`)
- 增加dropout `--dropout 0.5`

### F3预处理缓存

F3模块会缓存预处理结果到`checkpoints/`目录。如果需要重新预处理:
```bash
rm -rf checkpoints/f3_*
```

## 引用

如果使用本代码,请引用:

```
基于因果解耦与虚假相关消除的公平性异构图神经网络情感分析
Causal-HAFE: Causal Heterogeneous Aspect-Frequency Enhanced ABSA
```

## 联系与反馈

如有问题或建议,请通过GitHub Issues反馈。
