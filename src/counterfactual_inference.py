"""
反事实推理模块 (Counterfactual Inference with Total Indirect Effect)

在推理阶段通过反事实推理消除频率偏差:
TIE = Logits(A, R) - Logits(A, ∅)

其中:
- Logits(A, R): 完整输入下的预测 (包含偏差)
- Logits(A, ∅): 屏蔽上下文后的预测 (仅凭方面词的自然直接效应 NDE)

通过减去NDE,消除模型因高频词而产生的盲目预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualInference:
    """
    反事实推理器
    实现TIE (Total Indirect Effect) 计算
    """

    def __init__(self, model, mask_strategy='zero'):
        """
        Args:
            model: ABSA模型
            mask_strategy: 上下文屏蔽策略
                - 'zero': 用零向量替换上下文节点
                - 'mean': 用平均特征替换
                - 'noise': 用噪声替换
        """
        self.model = model
        self.mask_strategy = mask_strategy

    def mask_context(self, features, aspect_indices):
        """
        屏蔽上下文,仅保留方面词节点

        Args:
            features: [num_nodes, feature_dim] 节点特征
            aspect_indices: [num_aspects] 方面词节点索引

        Returns:
            masked_features: [num_nodes, feature_dim] 屏蔽后的特征
        """
        masked_features = features.clone()

        # 创建非方面词节点的掩码
        num_nodes = features.size(0)
        context_mask = torch.ones(num_nodes, dtype=torch.bool, device=features.device)
        context_mask[aspect_indices] = False

        # 根据策略屏蔽上下文
        if self.mask_strategy == 'zero':
            # 用零向量替换
            masked_features[context_mask] = 0.0

        elif self.mask_strategy == 'mean':
            # 用所有节点的平均特征替换
            mean_features = features.mean(dim=0, keepdim=True)
            masked_features[context_mask] = mean_features

        elif self.mask_strategy == 'noise':
            # 用高斯噪声替换
            noise = torch.randn_like(features[context_mask])
            masked_features[context_mask] = noise

        return masked_features

    def compute_tie(self, features, edge_index, aspect_indices, edge_types=None):
        """
        计算总间接效应 (TIE)

        Args:
            features: [num_nodes, feature_dim]
            edge_index: [2, num_edges]
            aspect_indices: [num_aspects]
            edge_types: [num_edges]

        Returns:
            tie_logits: [num_aspects, num_classes] TIE调整后的预测
            factual_logits: [num_aspects, num_classes] 原始预测
            nde_logits: [num_aspects, num_classes] NDE预测
        """
        # 1. 事实预测 Logits(A, R) - 完整输入
        factual_logits = self.model(features, edge_index, aspect_indices, edge_types)

        # 2. 反事实预测 Logits(A, ∅) - 屏蔽上下文
        masked_features = self.mask_context(features, aspect_indices)
        nde_logits = self.model(masked_features, edge_index, aspect_indices, edge_types)

        # 3. 计算TIE = Factual - NDE
        tie_logits = factual_logits - nde_logits

        return tie_logits, factual_logits, nde_logits

    def predict_with_tie(self, features, edge_index, aspect_indices, edge_types=None):
        """
        使用TIE进行预测

        Args:
            features: [num_nodes, feature_dim]
            edge_index: [2, num_edges]
            aspect_indices: [num_aspects]
            edge_types: [num_edges]

        Returns:
            predictions: [num_aspects] 预测的类别
            tie_logits: [num_aspects, num_classes] TIE logits
        """
        tie_logits, _, _ = self.compute_tie(
            features, edge_index, aspect_indices, edge_types
        )

        predictions = tie_logits.argmax(dim=1)

        return predictions, tie_logits


class AdaptiveCounterfactualInference:
    """
    自适应反事实推理器
    根据模型的置信度动态调整TIE的权重
    """

    def __init__(self, model, mask_strategy='zero', confidence_threshold=0.8):
        """
        Args:
            model: ABSA模型
            mask_strategy: 上下文屏蔽策略
            confidence_threshold: 置信度阈值
                - 高置信度 (>threshold): 使用较小的TIE权重
                - 低置信度 (<threshold): 使用较大的TIE权重
        """
        self.base_inferencer = CounterfactualInference(model, mask_strategy)
        self.confidence_threshold = confidence_threshold

    def compute_adaptive_tie(self, features, edge_index, aspect_indices, edge_types=None):
        """
        计算自适应TIE

        TIE_adaptive = α * Logits(A, R) + (1 - α) * TIE
        其中 α 根据置信度动态调整

        Args:
            features: [num_nodes, feature_dim]
            edge_index: [2, num_edges]
            aspect_indices: [num_aspects]
            edge_types: [num_edges]

        Returns:
            adaptive_logits: [num_aspects, num_classes] 自适应TIE logits
            alpha_weights: [num_aspects] 每个样本的权重
        """
        # 计算基础TIE
        tie_logits, factual_logits, nde_logits = self.base_inferencer.compute_tie(
            features, edge_index, aspect_indices, edge_types
        )

        # 计算置信度 (使用softmax概率的最大值)
        factual_probs = F.softmax(factual_logits, dim=1)
        confidence = factual_probs.max(dim=1)[0]  # [num_aspects]

        # 计算自适应权重
        # 高置信度 -> α接近1 (更相信原始预测)
        # 低置信度 -> α接近0 (更相信TIE调整)
        alpha = torch.sigmoid(
            (confidence - self.confidence_threshold) * 10
        )  # [num_aspects]

        # 自适应组合
        alpha = alpha.unsqueeze(1)  # [num_aspects, 1]
        adaptive_logits = alpha * factual_logits + (1 - alpha) * tie_logits

        return adaptive_logits, alpha.squeeze(1)

    def predict_with_adaptive_tie(self, features, edge_index, aspect_indices, edge_types=None):
        """
        使用自适应TIE进行预测

        Args:
            features: [num_nodes, feature_dim]
            edge_index: [2, num_edges]
            aspect_indices: [num_aspects]
            edge_types: [num_edges]

        Returns:
            predictions: [num_aspects] 预测的类别
            adaptive_logits: [num_aspects, num_classes]
            alpha_weights: [num_aspects] 权重
        """
        adaptive_logits, alpha_weights = self.compute_adaptive_tie(
            features, edge_index, aspect_indices, edge_types
        )

        predictions = adaptive_logits.argmax(dim=1)

        return predictions, adaptive_logits, alpha_weights


class EnsembleCounterfactualInference:
    """
    集成反事实推理器
    使用多种屏蔽策略并集成结果
    """

    def __init__(self, model, mask_strategies=['zero', 'mean', 'noise']):
        """
        Args:
            model: ABSA模型
            mask_strategies: 多种屏蔽策略的列表
        """
        self.inferencers = [
            CounterfactualInference(model, strategy)
            for strategy in mask_strategies
        ]

    def compute_ensemble_tie(self, features, edge_index, aspect_indices, edge_types=None):
        """
        计算集成TIE (对多种策略的TIE求平均)

        Args:
            features: [num_nodes, feature_dim]
            edge_index: [2, num_edges]
            aspect_indices: [num_aspects]
            edge_types: [num_edges]

        Returns:
            ensemble_logits: [num_aspects, num_classes] 集成后的logits
        """
        all_tie_logits = []

        for inferencer in self.inferencers:
            tie_logits, _, _ = inferencer.compute_tie(
                features, edge_index, aspect_indices, edge_types
            )
            all_tie_logits.append(tie_logits)

        # 对所有策略的TIE求平均
        ensemble_logits = torch.stack(all_tie_logits).mean(dim=0)

        return ensemble_logits

    def predict_with_ensemble_tie(self, features, edge_index, aspect_indices, edge_types=None):
        """
        使用集成TIE进行预测

        Args:
            features: [num_nodes, feature_dim]
            edge_index: [2, num_edges]
            aspect_indices: [num_aspects]
            edge_types: [num_edges]

        Returns:
            predictions: [num_aspects] 预测的类别
            ensemble_logits: [num_aspects, num_classes]
        """
        ensemble_logits = self.compute_ensemble_tie(
            features, edge_index, aspect_indices, edge_types
        )

        predictions = ensemble_logits.argmax(dim=1)

        return predictions, ensemble_logits


def test_counterfactual_inference():
    """测试反事实推理模块"""
    print("Testing Counterfactual Inference Module...")

    # 创建一个简单的测试模型
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 3)

        def forward(self, features, edge_index, aspect_indices, edge_types=None):
            aspect_features = features[aspect_indices]
            return self.linear(aspect_features)

    model = DummyModel()

    # 测试数据
    num_nodes = 10
    num_aspects = 2
    feature_dim = 768

    features = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    aspect_indices = torch.tensor([0, 5])
    edge_types = torch.randint(0, 4, (20,))

    # 测试基础反事实推理
    print("\n1. Testing Basic Counterfactual Inference:")
    cf_inferencer = CounterfactualInference(model, mask_strategy='zero')
    tie_logits, factual_logits, nde_logits = cf_inferencer.compute_tie(
        features, edge_index, aspect_indices, edge_types
    )

    print(f"   Factual logits shape: {factual_logits.shape}")
    print(f"   NDE logits shape: {nde_logits.shape}")
    print(f"   TIE logits shape: {tie_logits.shape}")
    print(f"   Sample TIE logits: {tie_logits[0]}")

    predictions, _ = cf_inferencer.predict_with_tie(
        features, edge_index, aspect_indices, edge_types
    )
    print(f"   Predictions: {predictions}")

    # 测试自适应反事实推理
    print("\n2. Testing Adaptive Counterfactual Inference:")
    adaptive_inferencer = AdaptiveCounterfactualInference(
        model, confidence_threshold=0.8
    )
    predictions, adaptive_logits, alpha_weights = adaptive_inferencer.predict_with_adaptive_tie(
        features, edge_index, aspect_indices, edge_types
    )

    print(f"   Adaptive logits shape: {adaptive_logits.shape}")
    print(f"   Alpha weights: {alpha_weights}")
    print(f"   Predictions: {predictions}")

    # 测试集成反事实推理
    print("\n3. Testing Ensemble Counterfactual Inference:")
    ensemble_inferencer = EnsembleCounterfactualInference(
        model, mask_strategies=['zero', 'mean']
    )
    predictions, ensemble_logits = ensemble_inferencer.predict_with_ensemble_tie(
        features, edge_index, aspect_indices, edge_types
    )

    print(f"   Ensemble logits shape: {ensemble_logits.shape}")
    print(f"   Predictions: {predictions}")

    print("\nCounterfactual Inference Module test passed!")


if __name__ == "__main__":
    test_counterfactual_inference()
