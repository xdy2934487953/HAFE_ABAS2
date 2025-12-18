"""
去混淆图注意力层 (Deconfounded GAT)
实现基于后门调整的因果图注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from sklearn.cluster import KMeans
import numpy as np


class DeconfoundedGATConv(MessagePassing):
    """
    去混淆图注意力卷积层

    通过后门调整消除混淆因子的影响:
    α_ij^causal ≈ Σ_k P(c_k) · Attention(h_i, h_j | c_k)

    其中 c_k 是通过聚类得到的上下文混淆因子原型
    """

    def __init__(self, in_channels, out_channels, num_edge_types=4,
                 num_confounders=5, heads=1, dropout=0.3, bias=True, **kwargs):
        """
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            num_edge_types: 边类型数量
            num_confounders: 混淆因子原型数量 (K)
            heads: 注意力头数
            dropout: dropout比例
        """
        super(DeconfoundedGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        self.num_confounders = num_confounders
        self.heads = heads
        self.dropout = dropout

        # 为每种边类型创建独立的变换矩阵
        self.weight_matrices = nn.ModuleList([
            nn.Linear(in_channels, out_channels * heads, bias=False)
            for _ in range(num_edge_types)
        ])

        # 为每个混淆因子原型创建注意力向量
        # 形状: [num_confounders, heads, 2 * out_channels]
        self.att_vectors = nn.Parameter(
            torch.Tensor(num_confounders, heads, 2 * out_channels)
        )

        # 混淆因子原型的先验概率 P(c_k)
        self.confounder_probs = nn.Parameter(
            torch.ones(num_confounders) / num_confounders
        )

        # 边类型重要性权重
        self.edge_importance = nn.Parameter(torch.ones(num_edge_types))

        # 混淆因子原型嵌入 (通过聚类初始化)
        self.confounder_prototypes = None  # 在preprocess时初始化

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels * heads))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        for weight in self.weight_matrices:
            nn.init.xavier_uniform_(weight.weight)

        nn.init.xavier_uniform_(self.att_vectors)

        # 初始化边类型重要性
        with torch.no_grad():
            self.edge_importance[0] = 2.0  # OPINION
            self.edge_importance[1] = 1.5  # SYNTAX_CORE
            self.edge_importance[2] = 1.0  # COREF
            self.edge_importance[3] = 0.5  # OTHER

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def initialize_confounders(self, node_features):
        """
        通过K-means聚类初始化混淆因子原型

        Args:
            node_features: [num_nodes, in_channels] 所有节点特征
        """
        print(f"\n初始化混淆因子原型 (K={self.num_confounders})...")

        # 转换为numpy进行聚类
        features_np = node_features.detach().cpu().numpy()

        # K-means聚类
        kmeans = KMeans(n_clusters=self.num_confounders, random_state=42, n_init=10)
        kmeans.fit(features_np)

        # 聚类中心作为混淆因子原型
        prototypes = torch.from_numpy(kmeans.cluster_centers_).float()
        self.confounder_prototypes = prototypes.to(node_features.device)

        # 更新先验概率为每个簇的样本比例
        labels = kmeans.labels_
        cluster_counts = np.bincount(labels, minlength=self.num_confounders)
        probs = cluster_counts / len(labels)

        with torch.no_grad():
            self.confounder_probs.copy_(torch.from_numpy(probs).float())

        print(f"混淆因子原型已初始化")
        print(f"簇大小分布: {cluster_counts}")
        print(f"先验概率 P(c_k): {probs}")

    def forward(self, x, edge_index, edge_types=None):
        """
        前向传播

        Args:
            x: [num_nodes, in_channels] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_types: [num_edges] 边类型

        Returns:
            out: [num_nodes, out_channels * heads] 输出特征
        """
        if edge_types is None:
            edge_types = torch.full((edge_index.shape[1],), 3,
                                   dtype=torch.long, device=x.device)

        # 检查混淆因子是否已初始化
        if self.confounder_prototypes is None:
            self.initialize_confounders(x)

        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_types = torch.full((x.size(0),), -1,
                                    dtype=torch.long, device=x.device)
        edge_types = torch.cat([edge_types, self_loop_types])

        # 按边类型分别处理并聚合
        out = torch.zeros(x.size(0), self.out_channels * self.heads,
                         device=x.device, dtype=x.dtype)

        for edge_type in range(self.num_edge_types):
            mask = (edge_types == edge_type)

            if mask.sum() == 0:
                continue

            type_edge_index = edge_index[:, mask]

            # 该类型的特征变换
            x_transformed = self.weight_matrices[edge_type](x)

            # 去混淆的消息传递
            out_type = self.propagate(
                type_edge_index,
                x=x_transformed,
                size=None
            )

            # 加权累加
            out += out_type * self.edge_importance[edge_type]

        # 处理自环 (简单的线性变换)
        self_loop_mask = (edge_types == -1)
        if self_loop_mask.sum() > 0:
            out += self.weight_matrices[0](x)  # 使用第一个权重矩阵

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i, x_j, edge_index_i, size_i):
        """
        计算去混淆的注意力权重并传递消息

        通过后门调整消除混淆:
        α_ij = Σ_k P(c_k) · exp(a_k^T [h_i || h_j])
        """
        # x_i, x_j: [num_edges, out_channels * heads]
        num_edges = x_i.size(0)

        # 重塑为多头格式: [num_edges, heads, out_channels]
        x_i = x_i.view(num_edges, self.heads, self.out_channels)
        x_j = x_j.view(num_edges, self.heads, self.out_channels)

        # 拼接源节点和目标节点特征: [num_edges, heads, 2*out_channels]
        x_cat = torch.cat([x_i, x_j], dim=-1)

        # 对每个混淆因子原型计算注意力
        # att_vectors: [num_confounders, heads, 2*out_channels]
        # x_cat: [num_edges, heads, 2*out_channels]
        # alpha_per_confounder: [num_edges, heads, num_confounders]

        alpha_per_confounder = torch.einsum(
            'ehd,khd->ehk',  # e=edges, h=heads, d=dims, k=confounders
            x_cat,
            self.att_vectors
        )

        # LeakyReLU激活
        alpha_per_confounder = F.leaky_relu(alpha_per_confounder, negative_slope=0.2)

        # 按混淆因子的先验概率加权求和
        # confounder_probs: [num_confounders]
        # alpha_causal: [num_edges, heads]
        alpha_causal = torch.einsum(
            'ehk,k->eh',
            alpha_per_confounder,
            F.softmax(self.confounder_probs, dim=0)  # 确保概率归一化
        )

        # Softmax归一化 (在目标节点的所有邻居上)
        alpha_causal = softmax(alpha_causal, edge_index_i, num_nodes=size_i)

        # Dropout
        alpha_causal = F.dropout(alpha_causal, p=self.dropout, training=self.training)

        # 应用注意力权重
        # alpha_causal: [num_edges, heads] -> [num_edges, heads, 1]
        # x_j: [num_edges, heads, out_channels]
        # out: [num_edges, heads, out_channels]
        out = x_j * alpha_causal.unsqueeze(-1)

        # 展平多头维度: [num_edges, heads * out_channels]
        out = out.view(num_edges, self.heads * self.out_channels)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'num_confounders={self.num_confounders})')


class TypeAwareDeconfoundedGAT(nn.Module):
    """
    结合类型感知和去混淆的GAT模块
    用于替换原始的TypeAwareGCN
    """

    def __init__(self, in_channels, out_channels, num_edge_types=4,
                 num_confounders=5, heads=1, dropout=0.3):
        super(TypeAwareDeconfoundedGAT, self).__init__()

        self.conv = DeconfoundedGATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_edge_types=num_edge_types,
            num_confounders=num_confounders,
            heads=heads,
            dropout=dropout
        )

    def forward(self, x, edge_index, edge_types=None):
        return self.conv(x, edge_index, edge_types)
