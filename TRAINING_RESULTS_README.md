# 训练结果保存系统

## 🎯 功能概述

为Causal-HAFE和简化版Causal-HAFE添加了完整的训练结果保存和管理功能，包括：

- 📊 **自动日志记录**：训练过程、评估指标、模型参数
- 💾 **智能模型保存**：最佳模型、定期checkpoint、最新状态
- 📈 **可视化图表**：训练曲线、性能对比、收敛分析
- 📋 **实验管理**：多实验对比、汇总报告、配置管理
- 🔍 **详细分析**：收敛状态、性能趋势、超参数影响

## 🚀 快速开始

### 1. 训练时自动保存

运行训练脚本时会自动创建实验日志：

```bash
# 简化版Causal-HAFE
python train_simplified.py --model simplified_causal_hafe --dataset semeval2014

# 原版Causal-HAFE
python train_causal.py --model causal_hafe --dataset semeval2014
```

训练完成后会显示：
```
实验日志目录: ./experiments/simplified_causal_hafe_semeval2014_lr0.001_h256_20241220_143000
```

## 📁 实验目录结构

每个实验自动创建完整的目录结构：

```
experiments/
└── model_dataset_params_timestamp/
    ├── checkpoints/           # 模型checkpoint
    │   ├── latest.pt         # 最新模型
    │   ├── best.pt          # 最佳模型
    │   └── epoch_10.pt      # 定期保存
    ├── logs/                 # 训练日志
    │   ├── train_log.csv    # 训练过程日志
    │   └── eval_log.csv     # 评估指标日志
    ├── plots/                # 可视化图表
    │   └── training_curves.png
    ├── configs/              # 配置文件
    │   └── experiment_config.json
    └── experiment_report.json # 实验总结报告
```

## 📊 日志内容

### 训练日志 (train_log.csv)
| 字段 | 说明 |
|------|------|
| epoch | 训练轮数 |
| train_loss | 训练损失 |
| task_loss | 任务损失 |
| indep_loss | 解耦损失 |
| bias_loss | 偏差拟合损失 |
| ib_loss | 信息瓶颈损失 |
| lr | 学习率 |
| grad_norm | 梯度范数 |
| time_elapsed | 每轮耗时(秒) |

### 评估日志 (eval_log.csv)
| 字段 | 说明 |
|------|------|
| epoch | 评估轮数 |
| accuracy | 准确率 |
| macro_f1 | 宏平均F1 |
| micro_f1 | 微平均F1 |
| gini | 基尼系数(公平性) |
| dp_aspect | 方面级差异(公平性) |
| high_freq_f1 | 高频方面F1 |
| low_freq_f1 | 低频方面F1 |

## 🛠️ 实验管理工具

### 列出所有实验
```bash
python experiment_manager.py --action list
```

### 分析单个实验
```bash
python experiment_manager.py --action analyze --experiments ./experiments/实验名
```

### 对比多个实验
```bash
python experiment_manager.py --action compare \
    --experiments ./experiments/exp1 ./experiments/exp2 ./experiments/exp3 \
    --output comparison_report.json
```

### 生成对比图表
```bash
python experiment_manager.py --action plot \
    --experiments ./experiments/exp1 ./experiments/exp2 \
    --metrics macro_f1 accuracy gini \
    --output comparison_plot.png
```

### 创建汇总表格
```bash
python experiment_manager.py --action table \
    --experiments ./experiments/exp1 ./experiments/exp2 \
    --output experiments_summary.csv
```

## 📈 可视化功能

### 自动生成的图表
训练完成后自动生成：
- **训练损失曲线**：总损失、任务损失、解耦损失等
- **评估指标曲线**：准确率、F1分数、公平性指标
- **学习率和梯度**：学习率变化、梯度范数监控

### 对比图表
使用experiment_manager.py生成：
- **多实验性能对比**
- **不同指标的趋势图**
- **超参数影响分析**

## 📋 实验报告

每个实验生成详细报告 (experiment_report.json)：

```json
{
  "experiment_name": "simplified_causal_hafe_semeval2014_lr0.001_h256",
  "timestamp": "20241220_143000",
  "config": {...},
  "best_metrics": {
    "epoch": 25,
    "macro_f1": 0.7421,
    "accuracy": 0.8115,
    "gini": 0.2341
  },
  "final_metrics": {...},
  "training_summary": {
    "avg_total_loss": 0.4231,
    "final_total_loss": 0.1123,
    "loss_convergence": "converged"
  }
}
```

## 🔧 高级用法

### 自定义实验名称
```python
from utils import ExperimentLogger

logger = ExperimentLogger(
    experiment_name="my_custom_experiment",
    save_dir="./my_experiments",
    config={"custom_param": "value"}
)
```

### 手动记录训练步骤
```python
logger.log_train_step(
    epoch=epoch,
    loss_dict={'total': 0.5, 'task': 0.3},
    lr=0.001,
    time_elapsed=45.2
)

logger.log_eval_step(epoch, metrics)
```

### 保存模型checkpoint
```python
logger.save_model(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    is_best=True
)
```

### 生成完整报告
```python
# 生成可视化
logger.plot_training_curves()

# 生成总结报告
report = logger.generate_report()

# 打印总结
logger.print_summary()
```

## 📊 实验对比分析

### 性能对比表格
```
实验汇总表格:
====================================================================================================
Experiment                          Model                  Dataset      LR    Best_Macro_F1  Best_Accuracy
----------------------------------------------------------------------------------------------------
simplified_causal_hafe_semeval2014  simplified_causal_hafe  semeval2014  0.001      0.7421         0.8115
causal_hafe_semeval2014             causal_hafe            semeval2014  0.0001     0.6832         0.7543
baseline_semeval2014                baseline               semeval2014  0.001      0.6987         0.7721
```

### 收敛分析
- **converged**: 损失在最后10轮中标准差 < 1%
- **converging**: 损失在最后10轮中标准差 < 5%
- **not_converged**: 损失仍未稳定

## 🎯 最佳实践

### 1. 实验命名
- 使用描述性名称：`{model}_{dataset}_{key_params}_{timestamp}`
- 包含重要超参数：学习率、隐藏维度等

### 2. 定期清理
```bash
# 只保留最佳实验
find ./experiments -name "*best.pt" -exec dirname {} \; | sort -u

# 删除旧实验
find ./experiments -mtime +30 -type d -exec rm -rf {} \;
```

### 3. 批量分析
```bash
# 分析所有实验
python experiment_manager.py --action analyze

# 生成完整对比报告
python experiment_manager.py --action table --output all_experiments.csv
python experiment_manager.py --action plot --output all_comparison.png
```

## 🔍 故障排除

### 常见问题

**Q: 实验目录没有创建？**
A: 检查写入权限，确保`./experiments`目录存在且可写。

**Q: 日志文件损坏？**
A: 删除损坏的实验目录，重新运行训练。

**Q: 可视化图表为空？**
A: 确保安装了matplotlib和seaborn：
```bash
pip install matplotlib seaborn
```

**Q: 内存不足？**
A: 减少`--eval_every`参数，或在分析时只加载必要的实验。

## 📚 API参考

### ExperimentLogger
```python
class ExperimentLogger:
    def __init__(experiment_name, save_dir="./experiments", config=None)
    def log_train_step(epoch, loss_dict, lr=None, grad_norm=None, time_elapsed=None)
    def log_eval_step(epoch, metrics)
    def save_model(model, optimizer=None, scheduler=None, epoch=None, is_best=False)
    def load_model(model, checkpoint_path, optimizer=None, scheduler=None)
    def plot_training_curves(save_plots=True)
    def generate_report()
    def print_summary()
```

### ABSAResultsManager
```python
class ABSAResultsManager:
    def __init__(results_dir="./experiments")
    def load_experiment(exp_dir)
    def compare_experiments(exp_names, metrics=['macro_f1', 'accuracy'])
    def generate_comparison_report(exp_names, output_file=None)
```

这个训练结果保存系统让您能够：
- 🔄 **重现实验**：完整的配置和checkpoint
- 📊 **分析性能**：详细的指标追踪和可视化
- 🔍 **对比实验**：系统性的多实验性能对比
- 📈 **优化参数**：基于历史数据的超参数调优
