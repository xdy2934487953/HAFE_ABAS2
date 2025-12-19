"""
解耦信息瓶颈模块 (Disentangled Information Bottleneck, DIB)

将节点表示解耦为:
- Z_c: 因果语义表示 (用于情感分类)
- Z_s: 虚假频率表示 (用于捕获频率偏差)

优化目标:
1. L_task: 使用 Z_c 预测情感标签
2. L_indep: 最小化 I(Z_c; Z_s) 确保解耦
3. L_bias: 强制 Z_s 预测方面词频率阶
4. L_IB: 最小化 I(Z_c; X) 去除冗余噪声
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DisentangledEncoder(nn.Module):
    """
    解耦编码器
    将输入特征分解为因果部分和虚假部分
    """

    def __init__(self, input_dim, causal_dim, spurious_dim, dropout=0.3):
        """
        Args:
            input_dim: 输入特征维度
            causal_dim: 因果表示维度
            spurious_dim: 虚假表示维度
        """
        super(DisentangledEncoder, self).__init__()

        self.input_dim = input_dim
        self.causal_dim = causal_dim
        self.spurious_dim = spurious_dim

        # 因果分支编码器
        self.causal_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, causal_dim)
        )

        # 虚假分支编码器
        self.spurious_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, spurious_dim)
        )

        # 权重初始化 (Xavier初始化提高稳定性)
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] 输入特征

        Returns:
            z_c: [batch_size, causal_dim] 因果表示
            z_s: [batch_size, spurious_dim] 虚假表示
        """
        z_c = self.causal_encoder(x)
        z_s = self.spurious_encoder(x)

        return z_c, z_s


class MutualInformationEstimator(nn.Module):
    """
    互信息估计器
    使用对比学习方法估计 I(Z_c; Z_s)
    """

    def __init__(self, causal_dim, spurious_dim, hidden_dim=128):
        super(MutualInformationEstimator, self).__init__()

        # 统计网络 T(z_c, z_s)
        self.statistics_network = nn.Sequential(
            nn.Linear(causal_dim + spurious_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z_c, z_s):
        """
        估计互信息 I(Z_c; Z_s)

        使用 MINE (Mutual Information Neural Estimation) 方法:
        I(X;Y) ≈ E[T(x,y)] - log(E[e^T(x,y')])

        Args:
            z_c: [batch_size, causal_dim]
            z_s: [batch_size, spurious_dim]

        Returns:
            mi_estimate: 标量互信息估计 (用于最小化互信息)
        """
        batch_size = z_c.size(0)

        # 正样本: 真实的配对 (z_c, z_s)
        joint = torch.cat([z_c, z_s], dim=1)
        t_joint = self.statistics_network(joint)

        # 负样本: 打乱z_s生成边缘分布
        z_s_shuffled = z_s[torch.randperm(batch_size)]
        marginal = torch.cat([z_c, z_s_shuffled], dim=1)
        t_marginal = self.statistics_network(marginal)

        # MINE下界 (使用log-sum-exp技巧提高数值稳定性)
        # I(X;Y) ≈ E[T(x,y)] - log(E[exp(T(x,y'))])
        t_max = t_marginal.max().detach()
        exp_term = torch.exp(t_marginal - t_max).mean()
        mi_lower_bound = t_joint.mean() - (torch.log(exp_term + 1e-8) + t_max)

        # 修复：对于解耦约束，我们希望最小化I(Z_c; Z_s)
        # 但MINE的下界可能为负（当无相关性时）或正（当有相关性时）
        # 我们应该始终返回正值用于最小化：max(0, mi_lower_bound) 或使用exp
        mi_estimate = torch.exp(mi_lower_bound)  # 转换为正值范围[0, inf)

        return mi_estimate


class FrequencyDiscriminator(nn.Module):
    """
    频率判别器
    强制 Z_s 捕获方面词的频率信息
    """

    def __init__(self, spurious_dim, num_frequency_buckets=5, hidden_dim=64):
        """
        Args:
            spurious_dim: 虚假表示维度
            num_frequency_buckets: 频率分桶数量 (如: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
        """
        super(FrequencyDiscriminator, self).__init__()

        self.num_buckets = num_frequency_buckets

        self.classifier = nn.Sequential(
            nn.Linear(spurious_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_frequency_buckets)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z_s):
        """
        预测频率分桶

        Args:
            z_s: [batch_size, spurious_dim]

        Returns:
            logits: [batch_size, num_frequency_buckets]
        """
        return self.classifier(z_s)


class InformationBottleneckRegularizer(nn.Module):
    """
    信息瓶颈正则化器
    最小化 I(Z_c; X) 以去除冗余噪声
    """

    def __init__(self, causal_dim, input_dim, hidden_dim=128):
        super(InformationBottleneckRegularizer, self).__init__()

        # 使用变分下界估计 I(Z_c; X)
        self.statistics_network = nn.Sequential(
            nn.Linear(causal_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z_c, x):
        """
        估计 I(Z_c; X)

        Args:
            z_c: [batch_size, causal_dim]
            x: [batch_size, input_dim]

        Returns:
            ib_estimate: 信息瓶颈项
        """
        batch_size = z_c.size(0)

        # 正样本
        joint = torch.cat([z_c, x], dim=1)
        t_joint = self.statistics_network(joint)

        # 负样本
        x_shuffled = x[torch.randperm(batch_size)]
        marginal = torch.cat([z_c, x_shuffled], dim=1)
        t_marginal = self.statistics_network(marginal)

        # 互信息下界 (使用log-sum-exp技巧)
        t_max = t_marginal.max().detach()
        exp_term = torch.exp(t_marginal - t_max).mean()
        ib_lower_bound = t_joint.mean() - (torch.log(exp_term + 1e-8) + t_max)

        # 修复：对于信息瓶颈，我们希望最小化I(Z_c; X)
        # 转换为正值用于最小化
        ib_estimate = torch.exp(ib_lower_bound)

        return ib_estimate


class DIBModule(nn.Module):
    """
    完整的解耦信息瓶颈模块
    """

    def __init__(self, input_dim, causal_dim, spurious_dim,
                 num_frequency_buckets=5, dropout=0.3):
        """
        Args:
            input_dim: 输入特征维度
            causal_dim: 因果表示维度
            spurious_dim: 虚假表示维度
            num_frequency_buckets: 频率分桶数量
        """
        super(DIBModule, self).__init__()

        self.input_dim = input_dim
        self.causal_dim = causal_dim
        self.spurious_dim = spurious_dim

        # 解耦编码器
        self.encoder = DisentangledEncoder(
            input_dim, causal_dim, spurious_dim, dropout
        )

        # 互信息估计器 (用于解耦约束)
        self.mi_estimator = MutualInformationEstimator(
            causal_dim, spurious_dim
        )

        # 频率判别器 (用于偏差拟合)
        self.freq_discriminator = FrequencyDiscriminator(
            spurious_dim, num_frequency_buckets
        )

        # 信息瓶颈正则化器
        self.ib_regularizer = InformationBottleneckRegularizer(
            causal_dim, input_dim
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            z_c: [batch_size, causal_dim] 因果表示
            z_s: [batch_size, spurious_dim] 虚假表示
        """
        z_c, z_s = self.encoder(x)
        return z_c, z_s

    def compute_dib_losses(self, x, z_c, z_s, frequency_labels):
        """
        计算DIB的各项损失

        Args:
            x: [batch_size, input_dim] 原始输入
            z_c: [batch_size, causal_dim] 因果表示
            z_s: [batch_size, spurious_dim] 虚假表示
            frequency_labels: [batch_size] 频率分桶标签

        Returns:
            losses: dict包含各项损失
        """
        losses = {}

        # 1. 解耦约束: 最小化 I(Z_c; Z_s)
        mi_estimate = self.mi_estimator(z_c, z_s)
        losses['indep'] = mi_estimate

        # 2. 偏差拟合: 强制 Z_s 预测频率
        freq_logits = self.freq_discriminator(z_s)
        losses['bias'] = F.cross_entropy(freq_logits, frequency_labels)

        # 3. 信息瓶颈: 最小化 I(Z_c; X)
        ib_estimate = self.ib_regularizer(z_c, x)
        losses['ib'] = ib_estimate

        return losses


def compute_frequency_buckets(aspect_frequencies, num_buckets=5):
    """
    将方面词频率转换为分桶标签

    Args:
        aspect_frequencies: [batch_size] 方面词的频率计数
        num_buckets: 分桶数量

    Returns:
        bucket_labels: [batch_size] 分桶标签 (0 到 num_buckets-1)
    """
    # 计算百分位数作为分桶边界
    percentiles = np.linspace(0, 100, num_buckets + 1)
    boundaries = np.percentile(aspect_frequencies, percentiles)

    # 分配到桶
    bucket_labels = np.digitize(aspect_frequencies, boundaries[1:-1])
    bucket_labels = np.clip(bucket_labels, 0, num_buckets - 1)

    return torch.from_numpy(bucket_labels).long()


def test_dib_module():
    """测试DIB模块"""
    print("Testing DIB Module...")

    batch_size = 32
    input_dim = 768
    causal_dim = 128
    spurious_dim = 64
    num_buckets = 5

    # 创建模块
    dib = DIBModule(input_dim, causal_dim, spurious_dim, num_buckets)

    # 模拟输入
    x = torch.randn(batch_size, input_dim)
    frequencies = np.random.randint(1, 100, size=batch_size)
    freq_labels = compute_frequency_buckets(frequencies, num_buckets)

    # 前向传播
    z_c, z_s = dib(x)

    print(f"Input shape: {x.shape}")
    print(f"Causal representation (Z_c) shape: {z_c.shape}")
    print(f"Spurious representation (Z_s) shape: {z_s.shape}")

    # 计算损失
    losses = dib.compute_dib_losses(x, z_c, z_s, freq_labels)

    print("\nDIB Losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    print("\nDIB Module test passed!")


if __name__ == "__main__":
    test_dib_module()
