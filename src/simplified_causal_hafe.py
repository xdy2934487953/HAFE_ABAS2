"""
简化版Causal-HAFE: 性能优化版ABSA模型

核心改进:
1. 简化GAT架构 - 使用标准GAT替代复杂DeconfoundedGAT
2. 联合分类 - 同时使用Z_c和Z_s进行情感分类
3. 增强特征 - 改进的图结构和情感词识别
4. 优化训练 - 更合适的超参数和训练策略

架构流程:
1. 增强BERT特征提取 (带情感增强)
2. DIB解耦编码 (分离因果和虚假表示)
3. 简化GAT图传播 (标准多头注意力)
4. 联合情感分类 (Z_c + Z_s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree

from disentangled_information_bottleneck import DIBModule


class EnhancedFeatureExtractor(nn.Module):
    """
    增强的特征提取器
    1. BERT特征 + 情感词增强
    2. 位置编码
    3. 节点重要性权重
    """

    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()

        # 情感增强层
        self.sentiment_enhancer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 位置编码
        self.position_encoder = nn.Linear(1, hidden_dim)

        # 节点类型编码 (aspect, opinion, context)
        self.node_type_encoder = nn.Embedding(3, hidden_dim)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, features, node_types=None, positions=None):
        """
        Args:
            features: [num_nodes, input_dim] BERT特征
            node_types: [num_nodes] 节点类型 (0=aspect, 1=opinion, 2=context)
            positions: [num_nodes] 位置索引
        """
        # 情感增强
        enhanced = self.sentiment_enhancer(features)

        # 节点类型编码
        if node_types is not None:
            type_emb = self.node_type_encoder(node_types)
            enhanced = enhanced + type_emb

        # 位置编码
        if positions is not None:
            pos_emb = self.position_encoder(positions.unsqueeze(-1))
            enhanced = enhanced + pos_emb

        # 输出投影
        output = self.output_proj(enhanced)
        output = self.dropout(output)

        return output


class SimplifiedGATConv(MessagePassing):
    """
    简化的GAT层
    移除复杂的混淆因子，专注标准多头注意力
    """

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3, edge_dim=4):
        super().__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        # 多头注意力
        self.conv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,  # 边类型特征
            concat=True
        )

        # 边类型嵌入
        self.edge_type_emb = nn.Embedding(edge_dim, edge_dim)

        # 残差连接
        if in_channels != out_channels * heads:
            self.residual = nn.Linear(in_channels, out_channels * heads)
        else:
            self.residual = nn.Identity()

        self.layer_norm = nn.LayerNorm(out_channels * heads)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_types=None):
        # 边类型编码
        if edge_types is not None:
            edge_attr = self.edge_type_emb(edge_types)
        else:
            edge_attr = None

        # GAT前向传播
        out = self.conv(x, edge_index, edge_attr)

        # 残差连接
        residual = self.residual(x)
        out = out + residual

        # LayerNorm + Dropout
        out = self.layer_norm(out)
        out = self.dropout_layer(out)

        return out


class JointClassifier(nn.Module):
    """
    联合分类器
    同时使用因果表示(Z_c)和虚假表示(Z_s)进行分类
    """

    def __init__(self, causal_dim, spurious_dim, hidden_dim=128, num_classes=3):
        super().__init__()

        total_dim = causal_dim + spurious_dim

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z_c, z_s):
        """
        Args:
            z_c: [batch_size, causal_dim]
            z_s: [batch_size, spurious_dim]

        Returns:
            logits: [batch_size, num_classes]
        """
        # 拼接因果和虚假表示
        joint_features = torch.cat([z_c, z_s], dim=1)
        logits = self.classifier(joint_features)
        return logits


class SimplifiedCausalHAFE(nn.Module):
    """
    简化版Causal-HAFE模型
    性能优化版，保持因果解耦核心思想
    """

    def __init__(self, input_dim=768, hidden_dim=256,
                 causal_dim=128, spurious_dim=64, num_classes=3,
                 device='cuda', dataset_name='absa',
                 # GAT参数
                 gat_heads=4, num_edge_types=4,
                 # DIB参数
                 num_frequency_buckets=5,
                 # 损失权重 (优化后的权重)
                 lambda_indep=0.1, lambda_bias=0.5, lambda_ib=0.01,
                 dropout=0.3):
        """
        Args:
            input_dim: BERT特征维度 (768)
            hidden_dim: 隐藏层维度
            causal_dim: 因果表示维度
            spurious_dim: 虚假表示维度
            num_classes: 情感类别数 (3: pos/neg/neu)
            gat_heads: GAT注意力头数
            num_edge_types: 边类型数
            lambda_indep: 解耦约束权重
            lambda_bias: 偏差拟合权重
            lambda_ib: 信息瓶颈权重
        """
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.causal_dim = causal_dim
        self.spurious_dim = spurious_dim
        self.dataset_name = dataset_name

        # 损失权重
        self.lambda_indep = lambda_indep
        self.lambda_bias = lambda_bias
        self.lambda_ib = lambda_ib

        # ===== 模块1: 增强特征提取 =====
        self.feature_extractor = EnhancedFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )

        # ===== 模块2: DIB解耦编码 =====
        self.dib_module = DIBModule(
            input_dim=hidden_dim,  # 使用增强后的特征
            causal_dim=causal_dim,
            spurious_dim=spurious_dim,
            num_frequency_buckets=num_frequency_buckets,
            dropout=dropout
        )

        # ===== 模块3: 简化GAT图传播 =====
        # 第一层GAT: hidden_dim -> hidden_dim
        self.gat1 = SimplifiedGATConv(
            in_channels=causal_dim,  # 只对因果表示做图传播
            out_channels=causal_dim,
            heads=gat_heads,
            dropout=dropout,
            edge_dim=num_edge_types
        )

        # 第二层GAT: causal_dim * heads -> causal_dim
        self.gat2 = SimplifiedGATConv(
            in_channels=causal_dim * gat_heads,
            out_channels=causal_dim,
            heads=1,  # 最后一层单头
            dropout=dropout,
            edge_dim=num_edge_types
        )

        # ===== 模块4: 联合情感分类器 =====
        self.classifier = JointClassifier(
            causal_dim=causal_dim,
            spurious_dim=spurious_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, features, edge_index, aspect_indices,
                edge_types=None, node_types=None, positions=None,
                frequency_labels=None, return_dib_losses=False):
        """
        前向传播

        Args:
            features: [num_nodes, input_dim]
            edge_index: [2, num_edges]
            aspect_indices: [num_aspects]
            edge_types: [num_edges] 边类型
            node_types: [num_nodes] 节点类型 (aspect/opinion/context)
            positions: [num_nodes] 位置编码
            frequency_labels: [num_aspects] 频率分桶标签
            return_dib_losses: 是否返回DIB损失

        Returns:
            logits: [num_aspects, num_classes] 情感分类logits
            dib_losses: dict (如果return_dib_losses=True)
        """
        # 1. 增强特征提取
        enhanced_features = self.feature_extractor(
            features, node_types, positions
        )

        # 2. DIB解耦编码
        # 仅对方面词节点进行解耦
        aspect_features = enhanced_features[aspect_indices]
        z_c_aspects, z_s_aspects = self.dib_module(aspect_features)

        # 对所有节点进行解耦 (用于图传播)
        z_c_all, z_s_all = self.dib_module(enhanced_features)

        # 3. 简化GAT图传播 (仅在因果表示上进行)
        h = self.gat1(z_c_all, edge_index, edge_types)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gat2(h, edge_index, edge_types)
        # h: [num_nodes, causal_dim]

        # 4. 提取方面词节点的因果embedding
        aspect_causal_embeddings = h[aspect_indices]  # [num_aspects, causal_dim]

        # 5. 联合情感分类 (使用Z_c和Z_s)
        logits = self.classifier(aspect_causal_embeddings, z_s_aspects)

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
        """
        # 任务损失
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


class SimplifiedHAFE_Baseline(nn.Module):
    """
    简化版HAFE基线
    移除因果模块，仅保留增强的图结构
    """

    def __init__(self, input_dim=768, hidden_dim=256, num_classes=3,
                 gat_heads=4, num_edge_types=4, dropout=0.3):
        super().__init__()

        # 特征提取
        self.feature_extractor = EnhancedFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )

        # 两层简化GAT
        self.gat1 = SimplifiedGATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            dropout=dropout,
            edge_dim=num_edge_types
        )

        self.gat2 = SimplifiedGATConv(
            in_channels=hidden_dim * gat_heads,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout,
            edge_dim=num_edge_types
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, edge_index, aspect_indices,
                edge_types=None, node_types=None, positions=None):
        # 特征提取
        h = self.feature_extractor(features, node_types, positions)

        # GAT传播
        h = self.gat1(h, edge_index, edge_types)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gat2(h, edge_index, edge_types)

        # 提取aspect嵌入
        aspect_embeddings = h[aspect_indices]
        logits = self.classifier(aspect_embeddings)

        return logits


def test_simplified_model():
    """测试简化版模型"""
    print("Testing Simplified Causal-HAFE Model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 参数
    batch_size = 2
    num_nodes = 10
    num_aspects = 2
    input_dim = 768
    hidden_dim = 256
    causal_dim = 128
    spurious_dim = 64
    num_classes = 3

    # 创建模型
    model = SimplifiedCausalHAFE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        causal_dim=causal_dim,
        spurious_dim=spurious_dim,
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
    node_types = torch.randint(0, 3, (num_nodes,)).to(device)
    positions = torch.arange(num_nodes, dtype=torch.float).to(device)
    labels = torch.randint(0, num_classes, (num_aspects,)).to(device)
    frequency_labels = torch.randint(0, 5, (num_aspects,)).to(device)

    # 测试前向传播
    print("\n1. Testing forward pass...")
    logits, dib_losses = model(
        features, edge_index, aspect_indices, edge_types,
        node_types, positions, frequency_labels, return_dib_losses=True
    )

    print(f"   Logits shape: {logits.shape}")
    print(f"   DIB losses: {dib_losses}")

    # 测试损失计算
    print("\n2. Testing loss computation...")
    total_loss, loss_dict = model.compute_total_loss(logits, labels, dib_losses)
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Loss breakdown: {loss_dict}")

    print("\nSimplified Causal-HAFE Model test passed!")


if __name__ == "__main__":
    test_simplified_model()
