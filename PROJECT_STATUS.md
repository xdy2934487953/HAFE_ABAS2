# Causal-HAFE 项目实现状态

**日期**: 2025-12-18
**状态**: 代码实现完成，等待测试

---

## 已完成的工作 ✅

### 1. 核心模块实现

#### ✅ 模块一：去混淆图注意力层
- **文件**: `src/deconfounded_gat.py`
- **功能**: 基于后门调整的Deconfounded GAT
- **关键类**: `DeconfoundedGATConv`, `TypeAwareDeconfoundedGAT`
- **原理**: α_ij^causal = Σ_k P(c_k) · Attention(h_i, h_j | c_k)

#### ✅ 模块二：解耦信息瓶颈 (DIB)
- **文件**: `src/disentangled_information_bottleneck.py`
- **功能**: 特征分解为因果部分(Z_c)和虚假部分(Z_s)
- **关键类**:
  - `DisentangledEncoder`: 编码器
  - `MutualInformationEstimator`: I(Z_c; Z_s)估计
  - `FrequencyDiscriminator`: 频率判别器
  - `DIBModule`: 完整DIB模块
- **损失**: L_indep + L_bias + L_IB

#### ✅ 模块三：反事实推理 (TIE)
- **文件**: `src/counterfactual_inference.py`
- **功能**: 基于总间接效应的反事实推理
- **关键类**:
  - `CounterfactualInference`: 基础TIE
  - `AdaptiveCounterfactualInference`: 自适应TIE
  - `EnsembleCounterfactualInference`: 集成TIE
- **公式**: TIE = Logits(A, R) - Logits(A, ∅)

#### ✅ 主模型整合
- **文件**: `src/causal_hafe.py`
- **功能**: 整合三大模块的完整Causal-HAFE模型
- **关键类**:
  - `CausalHAFE_Model`: 完整模型
  - `CausalHAFE_Baseline`: 消融基线
- **流程**: F3 → DIB → Deconfounded GAT → 分类器

### 2. 数据处理增强

#### ✅ 频率分桶功能
- **修改文件**: `src/data_loader.py`
- **新增方法**:
  - `compute_frequency_buckets()`: 计算频率分桶
  - `get_aspect_key()`: 获取aspect标识符
- **用途**: 为DIB模块提供频率标签

#### ✅ 公平性评估指标
- **文件**: `src/evaluator.py` (已有，已确认包含所需指标)
- **指标**:
  - Variance: Per-aspect F1方差
  - Gap: 最大-最小F1差距
  - Gini: 基尼系数
  - DP-Aspect: 高频vs低频性能差异

### 3. 训练脚本

#### ✅ Causal-HAFE训练脚本
- **文件**: `train_causal.py`
- **功能**:
  - Causal-HAFE完整训练流程
  - DIB多任务损失
  - TIE推理评估
  - 频率分桶集成
- **用法**: `python train_causal.py --dataset semeval2014 --model causal_hafe`

### 4. 文档

#### ✅ 使用文档
- **文件**: `CAUSAL_HAFE_README.md`
- **内容**: 完整的使用说明、参数解释、训练示例

#### ✅ 实验脚本
- **文件**: `run_causal_experiments.sh`
- **功能**: 一键运行所有对比实验

---

## 下一步计划 📋

### 立即任务（明天）

1. **测试框架可运行性** ⏰ 优先级：最高
   ```bash
   # 快速测试（10分钟）
   python train_causal.py --dataset semeval2014 --model causal_hafe --epochs 10 --eval_every 2
   ```
   **检查点**:
   - [ ] 能否正常启动
   - [ ] DIB损失是否正常计算
   - [ ] 是否有CUDA/内存问题
   - [ ] 公平性指标是否输出

2. **修复可能的Bug** ⏰ 取决于测试结果
   - 导入路径问题
   - 维度不匹配
   - 设备分配问题

3. **完整训练** ⏰ 测试通过后
   ```bash
   # 完整训练（1-2小时）
   python train_causal.py --dataset semeval2014 --model causal_hafe --epochs 50 --use_tie_inference
   ```

### 可选任务（时间充裕时）

4. **添加ARTS数据集支持** ⏰ 可选
   - 下载ARTS: `git clone https://github.com/zhijing-jin/ARTS_TestSet.git data/ARTS`
   - 实现ARTS数据加载器
   - 测试鲁棒性提升

5. **添加SemEval-2014 Laptop支持** ⏰ 可选
   - 修改data_loader添加Laptop解析
   - 跨领域验证

6. **长尾分割评估** ⏰ 可选
   - 自动分割Head/Medium/Tail
   - 分组公平性评估

---

## 技术细节备忘

### 模型参数配置

**推荐配置**:
```python
causal_dim = 128          # 因果表示维度
spurious_dim = 64         # 虚假表示维度
num_confounders = 5       # 混淆因子原型数
num_frequency_buckets = 5 # 频率分桶数
lambda_indep = 0.1        # 解耦约束权重
lambda_bias = 0.5         # 偏差拟合权重
lambda_ib = 0.01          # 信息瓶颈权重
```

**如果内存不足**:
```python
causal_dim = 64
spurious_dim = 32
num_confounders = 3
gat_heads = 1
```

### 关键文件位置

```
src/
├── deconfounded_gat.py                    # 新增
├── disentangled_information_bottleneck.py # 新增
├── counterfactual_inference.py            # 新增
├── causal_hafe.py                         # 新增
├── data_loader.py                         # 已修改（添加频率分桶）
├── evaluator.py                           # 未修改（已包含公平性指标）
├── hafe_absa.py                           # 原有（保留）
├── type_aware_gcn.py                      # 原有（保留）
└── fairPHM.py                             # 原有（F3模块）

train_causal.py                            # 新增
CAUSAL_HAFE_README.md                      # 新增
run_causal_experiments.sh                  # 新增
```

### 预期结果

根据文献（文献更新.md）：

1. **总体性能**: Macro-F1与RoBERTa/DualGCN持平或略高
2. **低频aspect**: Tail分组F1提升 **10-15%**
3. **公平性**: DP-Aspect显著降低
4. **鲁棒性**: ARTS数据集性能下降更小

---

## 已知问题和注意事项 ⚠️

1. **F3缓存**: 首次运行会预处理F3模块（约5-10分钟）
2. **设备自动检测**: 代码会自动选择CUDA/MPS/CPU
3. **边类型**: 需要确保graph_builder生成了edge_types
4. **内存使用**: Deconfounded GAT比标准GCN内存占用更大

---

## 快速命令参考

```bash
# 基础测试
python train_causal.py --dataset semeval2014 --model causal_hafe --epochs 10

# 完整训练
python train_causal.py --dataset semeval2014 --model causal_hafe --epochs 50

# 使用TIE推理
python train_causal.py --dataset semeval2014 --model causal_hafe --epochs 50 --use_tie_inference

# Baseline对比
python train_causal.py --dataset semeval2014 --model baseline --epochs 50

# 运行所有实验
bash run_causal_experiments.sh
```

---

## 联系方式

如有问题，在终端中继续对话即可：
```bash
claude code
# 然后说："继续Causal-HAFE项目"
```

---

**最后更新**: 2025-12-18
**状态**: ✅ 代码实现完成，⏳ 等待测试验证
