"""
Causal-HAFE: 基于因果解耦与虚假相关消除的公平性异构图神经网络情感分析

整合三个核心模块:
1. Deconfounded GAT: 基于后门调整的去混淆图注意力层
2. DIB: 解耦信息瓶颈 (特征分解为因果部分和虚假部分)
3. Counterfactual Inference: 基于TIE的反事实推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import sys
sys.path.append('.')

from deconfounded_gat import DeconfoundedGATConv, TypeAwareDeconfoundedGAT
from disentangled_information_bottleneck import (
    DIBModule, compute_frequency_buckets
)
from counterfactual_inference import (
    CounterfactualInference,
    AdaptiveCounterfactualInference,
    EnsembleCounterfactualInference
)
from fairPHM import EnhancedUnsupervisedF3Module


class CausalHAFE_Model(nn.Module):
    """
    Causal-HAFE完整模型

    架构流程:
    1. F3特征增强 (来自原始HAFE)
    2. DIB解耦编码 (分解为Z_c和Z_s)
    3. Deconfounded GAT图传播 (在Z_c上进行)
    4. 情感分类 (基于Z_c)
    """

    def __init__(self, input_dim, hidden_dim, num_classes=3,
                 device='cuda', dataset_name='absa',
                 # DIB参数
                 causal_dim=128, spurious_dim=64, num_frequency_buckets=5,
                 # Deconfounded GAT参数
                 num_confounders=5, num_edge_types=4, gat_heads=1,
                 # 损失权重 (降低以防止梯度爆炸)
                 lambda_indep=0.01, lambda_bias=0.1, lambda_ib=0.001,
                 dropout=0.3):
        """
        Args:
            input_dim: BERT特征维度 (768)
            hidden_dim: 隐藏层维度
            num_classes: 情感类别数 (3: pos/neg/neu)
            causal_dim: 因果表示维度
            spurious_dim: 虚假表示维度
            num_frequency_buckets: 频率分桶数
            num_confounders: 混淆因子原型数
            num_edge_types: 边类型数 (4)
            gat_heads: GAT注意力头数
            lambda_indep: 解耦约束权重
            lambda_bias: 偏差拟合权重
            lambda_ib: 信息瓶颈权重
        """
        super(CausalHAFE_Model, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.causal_dim = causal_dim
        self.dataset_name = dataset_name

        # 损失权重
        self.lambda_indep = lambda_indep
        self.lambda_bias = lambda_bias
        self.lambda_ib = lambda_ib

        # ===== 模块1: F3特征增强 (来自原始HAFE) =====
        self.f3_module = EnhancedUnsupervisedF3Module(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=device,
            delta=20,
            use_lambda_weighting=False,
            method='info_theory_only',
            adaptive_threshold=True,
            use_fast_mi=True
        )

        # ===== 模块2: 解耦信息瓶颈 (DIB) =====
        self.dib_module = DIBModule(
            input_dim=input_dim,
            causal_dim=causal_dim,
            spurious_dim=spurious_dim,
            num_frequency_buckets=num_frequency_buckets,
            dropout=dropout
        )

        # ===== 模块3: 去混淆图注意力层 =====
        # 第一层GAT: input_dim -> causal_dim
        self.gat1 = DeconfoundedGATConv(
            in_channels=causal_dim,
            out_channels=causal_dim,
            num_edge_types=num_edge_types,
            num_confounders=num_confounders,
            heads=gat_heads,
            dropout=dropout
        )

        # 第二层GAT: causal_dim -> causal_dim
        self.gat2 = DeconfoundedGATConv(
            in_channels=causal_dim * gat_heads,
            out_channels=causal_dim,
            num_edge_types=num_edge_types,
            num_confounders=num_confounders,
            heads=1,  # 最后一层使用单头
            dropout=dropout
        )

        # ===== 情感分类器 (仅使用Z_c) =====
        self.classifier = nn.Linear(causal_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

        # 预处理标志
        self.f3_preprocessed = False
        self.confounders_initialized = False

    def _compute_graph_hash(self, num_nodes, num_edges):
        """计算图的哈希值用于缓存"""
        hash_str = f"{self.dataset_name}_{num_nodes}_{num_edges}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]

    def preprocess_f3(self, all_graphs):
        """F3预处理 (与原始HAFE相同)"""
        if self.f3_preprocessed:
            return

        print("\n开始Causal-HAFE F3预处理...")

        # 合并所有图
        all_features = []
        all_edges = []
        node_offset = 0

        for graph in all_graphs:
            features = graph['features']
            edge_index = graph['edge_index']

            all_features.append(features)

            if edge_index.shape[1] > 0:
                adjusted_edges = edge_index + node_offset
                all_edges.append(adjusted_edges)

            node_offset += features.shape[0]

        combined_features = torch.cat(all_features, dim=0).to(self.device)

        if len(all_edges) > 0:
            combined_edge_index = torch.cat(all_edges, dim=1).to(self.device)
        else:
            combined_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        num_nodes = combined_features.shape[0]
        num_edges = combined_edge_index.shape[1]

        print(f"合并图: {num_nodes} 个节点, {num_edges} 条边")

        graph_hash = self._compute_graph_hash(num_nodes, num_edges)
        cache_name = f"{self.dataset_name}_{graph_hash}"

        print(f"缓存标识: {cache_name}")

        # 转换为EdgeIndexWrapper (与原HAFE相同)
        if num_edges > 0:
            class EdgeIndexWrapper:
                def __init__(self, edge_index):
                    self._edge_index = edge_index
                    self.storage = type('obj', (object,), {
                        '_row': edge_index[0],
                        '_col': edge_index[1]
                    })()

                def __getitem__(self, idx):
                    return self._edge_index[idx]

                @property
                def shape(self):
                    return self._edge_index.shape

                def cpu(self):
                    return EdgeIndexWrapper(self._edge_index.cpu())

                def to(self, device):
                    """添加to()方法支持设备转换"""
                    return EdgeIndexWrapper(self._edge_index.to(device))

            wrapped_edge_index = EdgeIndexWrapper(combined_edge_index)
        else:
            wrapped_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        # F3异质邻居检测
        self.f3_module.preprocess_hetero_neighbors(
            edge_index=wrapped_edge_index,
            node_features=combined_features,
            dataset_name=cache_name,
            sensitive_attrs=None
        )

        # 训练F3估计器
        self.f3_module.train_f3_estimator(
            node_features=combined_features,
            dataname=cache_name,
            epochs=500,
            lr=0.001
        )

        self.f3_preprocessed = True
        print("Causal-HAFE F3预处理完成!\n")

    def initialize_confounders(self, all_graphs):
        """
        初始化混淆因子原型 (基于所有训练数据)

        应在训练开始前调用一次
        """
        if self.confounders_initialized:
            return

        print("\n初始化Causal-HAFE混淆因子原型...")

        # 收集所有节点特征（DIB编码后的因果表示）
        all_causal_features = []

        with torch.no_grad():
            for graph in all_graphs:
                features = graph['features'].to(self.device)  # 确保在正确设备上

                # 先通过F3增强
                if self.f3_preprocessed:
                    features = self.f3_module.enhance_features(features)

                # 再通过DIB编码得到因果表示
                z_c_all, _ = self.dib_module(features)  # 现在features已经在CUDA上

                all_causal_features.append(z_c_all.cpu())  # 收集时转到CPU节省显存

        # 合并所有因果特征
        combined_features = torch.cat(all_causal_features, dim=0)

        print(f"收集到 {combined_features.shape[0]} 个节点的因果特征")

        # 对每一层GAT初始化混淆因子（转回CUDA）
        self.gat1.initialize_confounders(combined_features.to(self.device))
        self.gat2.initialize_confounders(combined_features.to(self.device))

        self.confounders_initialized = True
        print("混淆因子原型初始化完成!\n")

    def forward(self, features, edge_index, aspect_indices,
                edge_types=None, frequency_labels=None,
                return_dib_losses=False):
        """
        前向传播

        Args:
            features: [num_nodes, input_dim]
            edge_index: [2, num_edges]
            aspect_indices: [num_aspects]
            edge_types: [num_edges] 边类型
            frequency_labels: [num_aspects] 频率分桶标签 (训练时需要)
            return_dib_losses: 是否返回DIB损失

        Returns:
            logits: [num_aspects, num_classes] 情感分类logits
            dib_losses: dict (如果return_dib_losses=True)
        """
        # 1. F3特征增强
        if self.f3_preprocessed:
            enhanced_features = self.f3_module.enhance_features(features)
        else:
            enhanced_features = features

        # 2. DIB解耦编码
        # 仅对方面词节点进行解耦 (减少计算量)
        aspect_features = enhanced_features[aspect_indices]  # [num_aspects, input_dim]

        z_c_aspects, z_s_aspects = self.dib_module(aspect_features)
        # z_c_aspects: [num_aspects, causal_dim]
        # z_s_aspects: [num_aspects, spurious_dim]

        # 对所有节点进行解耦 (用于图传播)
        z_c_all, z_s_all = self.dib_module(enhanced_features)
        # z_c_all: [num_nodes, causal_dim]

        # 3. Deconfounded GAT图传播 (仅在因果表示上进行)
        h = self.gat1(z_c_all, edge_index, edge_types)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gat2(h, edge_index, edge_types)
        # h: [num_nodes, causal_dim]

        # 4. 提取方面词节点的embedding
        aspect_embeddings = h[aspect_indices]  # [num_aspects, causal_dim]

        # 5. 情感分类 (仅使用因果表示)
        logits = self.classifier(aspect_embeddings)

        # 6. 计算DIB损失 (训练时)
        if return_dib_losses and frequency_labels is not None:
            dib_losses = self.dib_module.compute_dib_losses(
                aspect_features, z_c_aspects, z_s_aspects, frequency_labels
            )
            return logits, dib_losses
        else:
            return logits

    def compute_total_loss(self, logits, labels, dib_losses):
        """
        计算总损失

        L_total = L_task + λ1*L_indep + λ2*L_bias + λ3*L_IB

        Args:
            logits: [num_aspects, num_classes]
            labels: [num_aspects]
            dib_losses: dict包含 'indep', 'bias', 'ib'

        Returns:
            total_loss: 标量总损失
            loss_dict: 各项损失的字典
        """
        # 任务损失 (情感分类)
        loss_task = F.cross_entropy(logits, labels)

        # DIB损失
        loss_indep = dib_losses['indep']
        loss_bias = dib_losses['bias']
        loss_ib = dib_losses['ib']

        # 总损失
        total_loss = (
            loss_task +
            self.lambda_indep * loss_indep +
            self.lambda_bias * loss_bias +
            self.lambda_ib * loss_ib
        )

        loss_dict = {
            'total': total_loss.item(),
            'task': loss_task.item(),
            'indep': loss_indep.item(),
            'bias': loss_bias.item(),
            'ib': loss_ib.item()
        }

        return total_loss, loss_dict

    def predict_with_tie(self, features, edge_index, aspect_indices,
                        edge_types=None, tie_mode='basic'):
        """
        使用反事实推理进行预测 (推理阶段)

        Args:
            features: [num_nodes, input_dim]
            edge_index: [2, num_edges]
            aspect_indices: [num_aspects]
            edge_types: [num_edges]
            tie_mode: 'basic', 'adaptive', 或 'ensemble'

        Returns:
            predictions: [num_aspects]
            logits: [num_aspects, num_classes]
        """
        self.eval()

        with torch.no_grad():
            if tie_mode == 'basic':
                inferencer = CounterfactualInference(self, mask_strategy='zero')
                predictions, logits = inferencer.predict_with_tie(
                    features, edge_index, aspect_indices, edge_types
                )

            elif tie_mode == 'adaptive':
                inferencer = AdaptiveCounterfactualInference(
                    self, mask_strategy='zero', confidence_threshold=0.8
                )
                predictions, logits, _ = inferencer.predict_with_adaptive_tie(
                    features, edge_index, aspect_indices, edge_types
                )

            elif tie_mode == 'ensemble':
                inferencer = EnsembleCounterfactualInference(
                    self, mask_strategies=['zero', 'mean']
                )
                predictions, logits = inferencer.predict_with_ensemble_tie(
                    features, edge_index, aspect_indices, edge_types
                )

            else:
                # 默认: 不使用TIE,直接预测
                logits = self.forward(features, edge_index, aspect_indices, edge_types)
                predictions = logits.argmax(dim=1)

        return predictions, logits


class CausalHAFE_Baseline(nn.Module):
    """
    Causal-HAFE消融基线
    去除因果模块,仅保留异构图结构
    """

    def __init__(self, input_dim, hidden_dim, num_classes=3,
                 num_edge_types=4, dropout=0.3):
        super(CausalHAFE_Baseline, self).__init__()

        from type_aware_gcn import TypeAwareGCNConv

        self.gcn1 = TypeAwareGCNConv(input_dim, hidden_dim, num_edge_types)
        self.gcn2 = TypeAwareGCNConv(hidden_dim, hidden_dim, num_edge_types)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, edge_index, aspect_indices, edge_types=None):
        h = self.gcn1(features, edge_index, edge_types)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gcn2(h, edge_index, edge_types)

        aspect_embeddings = h[aspect_indices]
        logits = self.classifier(aspect_embeddings)

        return logits


def test_causal_hafe():
    """测试Causal-HAFE模型"""
    print("Testing Causal-HAFE Model...\n")

    # 参数
    batch_size = 2
    num_nodes = 10
    num_aspects = 2
    input_dim = 768
    hidden_dim = 128
    causal_dim = 64
    num_classes = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = CausalHAFE_Model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        causal_dim=causal_dim,
        num_classes=num_classes,
        device=device,
        dataset_name='test'
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 模拟数据
    features = torch.randn(num_nodes, input_dim).to(device)
    edge_index = torch.randint(0, num_nodes, (2, 20)).to(device)
    aspect_indices = torch.tensor([0, 5]).to(device)
    edge_types = torch.randint(0, 4, (20,)).to(device)
    labels = torch.randint(0, num_classes, (num_aspects,)).to(device)
    frequency_labels = torch.randint(0, 5, (num_aspects,)).to(device)

    # 测试前向传播
    print("\n1. Testing forward pass...")
    logits, dib_losses = model(
        features, edge_index, aspect_indices, edge_types,
        frequency_labels, return_dib_losses=True
    )

    print(f"   Logits shape: {logits.shape}")
    print(f"   DIB losses: {dib_losses}")

    # 测试损失计算
    print("\n2. Testing loss computation...")
    total_loss, loss_dict = model.compute_total_loss(logits, labels, dib_losses)
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Loss breakdown: {loss_dict}")

    # 测试TIE预测
    print("\n3. Testing TIE prediction...")
    predictions, tie_logits = model.predict_with_tie(
        features, edge_index, aspect_indices, edge_types, tie_mode='basic'
    )
    print(f"   Predictions: {predictions}")
    print(f"   TIE logits shape: {tie_logits.shape}")

    print("\nCausal-HAFE Model test passed!")


if __name__ == "__main__":
    test_causal_hafe()
