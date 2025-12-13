import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import grad
from torch.autograd import Function
from sklearn.cluster import KMeans
import torch.distributed as dist
import os
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GraphConv, global_mean_pool
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from utils import fair_metric
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from scipy.special import expit
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# 自动检测设备：如果是 Mac M芯片用 mps，如果是 NVIDIA 用 cuda，否则用 cpu
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class LogWriter:
    def __init__(self, logdir='./logs'):
        self.writer = SummaryWriter(logdir)

    def record(self, loss_item: dict, step: int):
        for key, value in loss_item.items():
            self.writer.add_scalar(key, value, step)

class InformationTheoryProcessor:
    """信息理论计算处理器"""
    
    def __init__(self, bins=15, device='cuda', use_fast_approximation=True):
        self.bins = bins
        self.device = device
        self.use_fast_approximation = use_fast_approximation
    
    def compute_mutual_information(self, x1, x2):
        """
        互信息计算 - 向量化实现
        
        """
        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(1)
        if len(x2.shape) == 1:
            x2 = x2.unsqueeze(1)
        
        # 确保在同一设备上
        x1 = x1.to(self.device).float()
        x2 = x2.to(self.device).float()
        
        # 处理维度不匹配
        if x1.shape[1] != x2.shape[1]:
            min_dim = min(x1.shape[1], x2.shape[1])
            x1 = x1[:, :min_dim]
            x2 = x2[:, :min_dim]
        
        if self.use_fast_approximation:
            return self._compute_mi_fast_vectorized(x1, x2)
        else:
            return self._compute_mi_accurate_batch(x1, x2)
    
    def _compute_mi_fast_vectorized(self, x1, x2):
        """
        快速向量化互信息计算
        
        基于理论：对于高斯分布，互信息 ≈ -0.5 * log(1 - ρ²)
        """
        # 批量标准化 一次性处理所有样本
        x1_mean = x1.mean(dim=1, keepdim=True)
        x1_std = x1.std(dim=1, keepdim=True) + 1e-8
        x1_norm = (x1 - x1_mean) / x1_std
        
        x2_mean = x2.mean(dim=1, keepdim=True)
        x2_std = x2.std(dim=1, keepdim=True) + 1e-8
        x2_norm = (x2 - x2_mean) / x2_std
        
        # 向量化相关系数计算
        correlations = torch.sum(x1_norm * x2_norm, dim=1) / x1.shape[1]
        
        # 向量化互信息转换
        correlations_abs = torch.abs(correlations)
        mutual_info_approx = -0.5 * torch.log(1 - correlations_abs.pow(2) + 1e-8)
        
        # 归一化到[0,1]范围
        mutual_info_normalized = torch.sigmoid(mutual_info_approx)
        
        return mutual_info_normalized
    
    def _compute_mi_accurate_batch(self, x1, x2):
        """准确互信息计算"""
        batch_size = x1.shape[0]
        
        # 批量离散化
        x1_discrete = self._batch_discretize(x1)
        x2_discrete = self._batch_discretize(x2)
        
        # 并行互信息计算
        mutual_infos = []
        for i in range(batch_size):
            mi = self._compute_mi_from_discrete(x1_discrete[i], x2_discrete[i])
            mutual_infos.append(mi)
        
        return torch.tensor(mutual_infos, device=self.device, dtype=torch.float32)
    
    def _batch_discretize(self, features):
        """批量特征离散化"""
        batch_size, feature_dim = features.shape
        
        if feature_dim == 1:
            # 标量情况的快速处理
            return torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        
        #  向量化分位数计算
        discretized_batch = []
        for i in range(batch_size):
            feat = features[i]
            quantiles = torch.quantile(feat, torch.linspace(0, 1, self.bins + 1, device=self.device))
            discretized = torch.bucketize(feat, quantiles[1:-1])
            discretized_batch.append(discretized)
        
        return torch.stack(discretized_batch)
    
    def _compute_mi_from_discrete(self, x_discrete, y_discrete):
        """从离散化数据计算互信息"""
        if len(x_discrete) == 0 or len(y_discrete) == 0:
            return 0.0
        
        # 计算联合直方图
        joint_hist = torch.zeros(self.bins, self.bins, device=self.device)
        
        x_indices = torch.clamp(x_discrete, 0, self.bins - 1)
        y_indices = torch.clamp(y_discrete, 0, self.bins - 1)
        
        # 快速直方图计算
        for x_val, y_val in zip(x_indices, y_indices):
            joint_hist[x_val, y_val] += 1
        
        joint_hist = joint_hist / (joint_hist.sum() + 1e-10)
        
        # 计算边际分布
        marginal_x = joint_hist.sum(dim=1) + 1e-10
        marginal_y = joint_hist.sum(dim=0) + 1e-10
        
        # 计算互信息
        joint_hist = joint_hist + 1e-10
        marginal_product = marginal_x.unsqueeze(1) * marginal_y.unsqueeze(0)
        mi = torch.sum(joint_hist * torch.log(joint_hist / marginal_product))
        
        return mi.item()
    
    def compute_entropy(self, features):
        """优化版熵计算"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        features = features.to(self.device).float()
        
        if self.use_fast_approximation:
            # 快速熵近似：基于方差
            # 对于高斯分布，熵 ≈ 0.5 * log(2πeσ²)
            variances = torch.var(features, dim=1) + 1e-8
            entropy_approx = 0.5 * torch.log(2 * torch.pi * torch.e * variances)
            return torch.sigmoid(entropy_approx)  # 归一化
        else:
            # 准确但较慢的熵计算
            return self._compute_entropy_accurate(features)
    
    def _compute_entropy_accurate(self, features):
        """准确的熵计算"""
        batch_size = features.shape[0]
        entropies = []
        
        for i in range(batch_size):
            feature_vec = features[i]
            
            if len(feature_vec) < 2:
                entropies.append(0.0)
                continue
            
            hist = torch.histc(feature_vec, bins=self.bins, 
                             min=feature_vec.min(), max=feature_vec.max())
            
            prob_dist = hist / hist.sum()
            prob_dist = prob_dist[prob_dist > 0]
            
            if len(prob_dist) > 0:
                entropy_val = -torch.sum(prob_dist * torch.log2(prob_dist + 1e-10))
                entropies.append(entropy_val.item())
            else:
                entropies.append(0.0)
        
        return torch.tensor(entropies, device=self.device, dtype=torch.float32)

class InfoTheoryHeteroDetector(nn.Module):
    """异质邻居检测器"""
    
    def __init__(self, input_dim, hidden_dim, device, bins=15, use_fast_mi=True):
        super(InfoTheoryHeteroDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.bins = bins
        self.use_fast_mi = use_fast_mi
        
        # 信息理论处理器
        self.info_processor = InformationTheoryProcessor(
            bins=bins, device=device, use_fast_approximation=use_fast_mi)
        
        # 信息融合网络 - 用于综合多个信息理论指标
        self.fusion_net = nn.Sequential(
            nn.Linear(3, 32),  # 减少输入维度（去除对比学习特征）
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)
        
        # 缓存
        self.training_stats = {
            'info_scores': [],
            'fusion_weights': []
        }
        
        # 性能提示
        if use_fast_mi:
            print(f"快速互信息计算模式")
        else:
            print(f"使用精确互信息计算模式")
    
    def _get_edge_indices(self, edge_index):
        """处理不同类型的edge_index，返回row, col"""
        # 情况1：SparseTensor格式（你原来的代码）
        if hasattr(edge_index, 'storage') and hasattr(edge_index.storage, '_row'):
            row, col = edge_index.storage._row, edge_index.storage._col
            if hasattr(row, 'cpu'):
                row, col = row.cpu(), col.cpu()
        # 情况2：普通tensor格式 [2, num_edges]
        elif isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            if hasattr(edge_index, 'cpu'):
                edge_index = edge_index.cpu()
            row, col = edge_index[0], edge_index[1]
        # 情况3：已经是tuple
        elif isinstance(edge_index, (tuple, list)):
            row, col = edge_index[0], edge_index[1]
        else:
            raise TypeError(f"不支持的edge_index类型: {type(edge_index)}")
    
        return row, col
    
    def _get_num_edges(self, edge_index):
        """获取边的数量"""
        if hasattr(edge_index, 'storage'):  # SparseTensor
            return len(edge_index.storage._row)
        else:  # 普通tensor
            return edge_index.shape[1]
        
    def detect_heterogeneous_edges(self, features, edge_index):
        """主检测方法：基于信息理论"""
        print("信息理论异质邻居检测...")
        
        # Step 1: 计算信息理论指标
        print("  计算信息理论指标...")
        info_metrics = self._compute_information_metrics(features, edge_index)
        
        # Step 2: 信息理论指标融合
        print(" 融合信息理论指标...")
        fusion_results = self._info_theory_fusion(info_metrics)
        
        # Step 3: 确定异质边
        hetero_mask, final_threshold = self._determine_heterogeneous_edges(
            fusion_results['combined_scores'])
        
        # 统计信息
        self._update_training_stats(info_metrics, fusion_results)
        
        detailed_scores = {
            'info_scores': info_metrics,
            'fusion_scores': fusion_results,
            'final_threshold': final_threshold,
            'hetero_ratio': hetero_mask.float().mean().item()
        }
        
        training_info = {
            'avg_mutual_info': info_metrics['mutual_info'].mean().item(),
            'avg_conditional_entropy': info_metrics['conditional_entropy'].mean().item(),
            'avg_degree_entropy': info_metrics['degree_entropy'].mean().item()
        }
        
        print(f"检测完成！异质边比例: {detailed_scores['hetero_ratio']*100:.1f}%")
        
        return hetero_mask, detailed_scores, training_info
    
    def _compute_information_metrics(self, features, edge_index):
        """计算信息理论指标"""
        row, col = self._get_edge_indices(edge_index)
        
        # 特征互信息
        feature_mi = self.info_processor.compute_mutual_information(
            features[row], features[col])
        
        # 特征熵
        feature_entropy_row = self.info_processor.compute_entropy(features[row])
        feature_entropy_col = self.info_processor.compute_entropy(features[col])
        conditional_entropy = (feature_entropy_row + feature_entropy_col) / 2 - feature_mi
        
        # 计算度数相关的信息理论指标
        degrees = torch.bincount(row, minlength=features.shape[0]).float()
        norm_degrees = degrees / degrees.max()
        degree_entropy = self.info_processor.compute_entropy(
            torch.stack([norm_degrees[row], norm_degrees[col]], dim=1))
        
        return {
            'mutual_info': feature_mi,
            'conditional_entropy': conditional_entropy,
            'degree_entropy': degree_entropy
        }
    
    def _info_theory_fusion(self, info_metrics):
        """信息理论指标融合"""
        
        # 构建融合输入：互信息、条件熵、度数熵
        fusion_input = torch.stack([
            info_metrics['mutual_info'],
            info_metrics['conditional_entropy'],
            info_metrics['degree_entropy']
        ], dim=1)
        
        # 通过网络学习融合权重
        fusion_weights = self.fusion_net(fusion_input).squeeze()
        
        # 基于信息理论的异质性得分
        # 低互信息 + 高条件熵 + 高度数熵 = 更异质
        info_heterogeneity_score = (
            (1 - info_metrics['mutual_info']) * 0.5 +  # 低互信息权重
            info_metrics['conditional_entropy'] * 0.3 +  # 高条件熵权重
            info_metrics['degree_entropy'] * 0.2         # 度数多样性权重
        )
        
        # 融合得分：结合学习到的权重和启发式得分
        combined_scores = fusion_weights * info_heterogeneity_score + \
                         (1 - fusion_weights) * (1 - info_metrics['mutual_info'])
        
        return {
            'info_heterogeneity': info_heterogeneity_score,
            'fusion_weights': fusion_weights,
            'combined_scores': combined_scores
        }
    
    def _determine_heterogeneous_edges(self, combined_scores):
        """确定异质边：自适应阈值"""
        
        # 多种阈值策略
        threshold_q25 = torch.quantile(combined_scores, 0.25)
        threshold_q30 = torch.quantile(combined_scores, 0.30)
        
        mean_score = combined_scores.mean()
        std_score = combined_scores.std()
        threshold_stat = mean_score + 0.5 * std_score  # 注意：这里改为加号，因为高分表示更异质
        
        # 梯度变化点检测
        sorted_scores, _ = torch.sort(combined_scores, descending=True)  # 降序排序
        gradients = sorted_scores[:-1] - sorted_scores[1:]
        max_gradient_idx = torch.argmax(gradients)
        threshold_gradient = sorted_scores[max_gradient_idx]
        
        # 综合阈值
        final_threshold = torch.median(torch.stack([
            threshold_q25, threshold_q30, threshold_stat, threshold_gradient
        ]))
        
        hetero_mask = combined_scores > final_threshold  # 改为大于阈值
        
        # 异质边比例调整
        hetero_ratio = hetero_mask.float().mean()
        if hetero_ratio < 0.05:
            final_threshold = torch.quantile(combined_scores, 0.95)
            hetero_mask = combined_scores > final_threshold
        elif hetero_ratio > 0.5:
            final_threshold = torch.quantile(combined_scores, 0.75)
            hetero_mask = combined_scores > final_threshold
        
        return hetero_mask, final_threshold
    
    def _update_training_stats(self, info_metrics, fusion_results):
        """更新训练统计信息"""
        self.training_stats['info_scores'].append({
            'avg_mi': info_metrics['mutual_info'].mean().item(),
            'avg_ce': info_metrics['conditional_entropy'].mean().item(),
            'avg_de': info_metrics['degree_entropy'].mean().item()
        })
        self.training_stats['fusion_weights'].append(
            fusion_results['fusion_weights'].mean().item())
    
    def get_detection_summary(self):
        """获取检测过程的详细总结"""
        if not self.training_stats['info_scores']:
            return "尚未进行检测"
        
        latest_scores = self.training_stats['info_scores'][-1]
        latest_weight = self.training_stats['fusion_weights'][-1]
        
        summary = f"""
异质邻居检测总结
{'='*50}
信息理论指标:
   - 平均互信息: {latest_scores['avg_mi']:.4f}
   - 平均条件熵: {latest_scores['avg_ce']:.4f}
   - 平均度数熵: {latest_scores['avg_de']:.4f}

融合策略:
   - 平均融合权重: {latest_weight:.3f}
   - 检测原理: 基于互信息最小化和熵最大化
"""
        return summary

class EnhancedUnsupervisedF3Module(nn.Module):
    """增强的无监督F3模块 -使用信息理论检测异质邻居"""
    
    def __init__(self, input_dim, hidden_dim, device, delta, use_lambda_weighting,
                 similarity_threshold=0.6, method='info_theory_only', 
                 adaptive_threshold=True, use_fast_mi=True):
        super(EnhancedUnsupervisedF3Module, self).__init__()
        
        print("初始化纯信息理论无监督F3模块...")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.delta = delta
        self.use_lambda_weighting = use_lambda_weighting
        self.method = method
        self.adaptive_threshold = adaptive_threshold
        self.use_fast_mi = use_fast_mi
        
        # 信息理论异质邻居检测器
        self.hetero_detector = InfoTheoryHeteroDetector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=device,
            use_fast_mi=use_fast_mi
        )
        
        # F3估计器保持不变
        self.f3_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        ).to(device)
        
        self.gating_network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(device)

        # 缓存和状态
        self.hetero_adj_sparse = None
        self.hetero_features = None
        self.lambda_weights = None
        self.is_preprocessed = False
        self.detection_summary = None
    
    def _get_edge_indices(self, edge_index):
        """处理不同类型的edge_index，返回row, col"""
        # 情况1：SparseTensor格式（你原来的代码）
        if hasattr(edge_index, 'storage') and hasattr(edge_index.storage, '_row'):
            row, col = edge_index.storage._row, edge_index.storage._col
            if hasattr(row, 'cpu'):
                row, col = row.cpu(), col.cpu()
        # 情况2：普通tensor格式 [2, num_edges]
        elif isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            if hasattr(edge_index, 'cpu'):
                edge_index = edge_index.cpu()
            row, col = edge_index[0], edge_index[1]
        # 情况3：已经是tuple
        elif isinstance(edge_index, (tuple, list)):
            row, col = edge_index[0], edge_index[1]
        else:
            raise TypeError(f"不支持的edge_index类型: {type(edge_index)}")
        return row, col
    
    def preprocess_hetero_neighbors(self, edge_index, node_features, dataset_name, 
                                  sensitive_attrs=None):
        """使用信息理论方法进行异质邻居预处理"""
        num_nodes = node_features.shape[0]
        cache_file = f"{dataset_name}_info_theory_hetero_sparse.pt"
        
        if hasattr(node_features, 'cpu'):
            node_features = node_features.cpu()
        if hasattr(edge_index, 'cpu'):
            edge_index = edge_index.cpu()
        
        print("信息理论异质邻居检测...")
        
        try:
            if dataset_name != "custom" and os.path.exists(cache_file):
                # 改进后的代码
                cached_data = torch.load(cache_file)
                cached_adj = cached_data['hetero_adj_sparse']

                # 验证缓存是否匹配当前图的大小
                if cached_adj.shape[0] == num_nodes:
                    self.hetero_adj_sparse = cached_adj
                    print(f"加载缓存的信息论异质邻居矩阵: {cache_file}")
                else:
                    print(f"警告：缓存大小不匹配！")
                    print(f"  缓存: {cached_adj.shape[0]} 节点")
                    print(f"  当前: {num_nodes} 节点")
                    print(f"  删除旧缓存并重新计算...")
                    import os
                    os.remove(cache_file)
                    raise FileNotFoundError("缓存失效，重新计算")
                self.detection_summary = cached_data.get('detection_summary', None)
                print(f"加载缓存的信息理论异质邻居矩阵: {cache_file}")
            else:
                raise FileNotFoundError("重新计算")
        except:
            print("重新计算信息理论异质邻居矩阵...")
            
            hetero_mask, detailed_scores, training_info = self.hetero_detector.detect_heterogeneous_edges(
                node_features.to(self.device), edge_index.to(self.device))
            
            row, col = self._get_edge_indices(edge_index)
            
            hetero_row = row[hetero_mask.cpu()]
            hetero_col = col[hetero_mask.cpu()]
            
            if len(hetero_row) > 0:
                hetero_indices = torch.stack([hetero_row, hetero_col])
                hetero_values = torch.ones(len(hetero_row), dtype=torch.float32)
                
                self.hetero_adj_sparse = torch.sparse_coo_tensor(
                    hetero_indices, hetero_values, (num_nodes, num_nodes)
                ).coalesce()
            else:
                hetero_indices = torch.zeros((2, 0), dtype=torch.long)
                hetero_values = torch.zeros(0, dtype=torch.float32)
                self.hetero_adj_sparse = torch.sparse_coo_tensor(
                    hetero_indices, hetero_values, (num_nodes, num_nodes)
                )
            
            self.detection_summary = {
                'detailed_scores': detailed_scores,
                'training_info': training_info,
                'detector_summary': self.hetero_detector.get_detection_summary()
            }
            
            if dataset_name != "custom":
                cache_data = {
                    'hetero_adj_sparse': self.hetero_adj_sparse,
                    'detection_summary': self.detection_summary
                }
                torch.save(cache_data, cache_file)
                print(f"保存信息理论异质邻居矩阵: {cache_file}")
        
        self._compute_hetero_features_sparse(node_features)
        
        if self.use_lambda_weighting:
            self._compute_lambda_weights_sparse(edge_index)
        
        self.is_preprocessed = True
        
        if self.detection_summary and 'detector_summary' in self.detection_summary:
            print(self.detection_summary['detector_summary'])
        
        print("信息理论F3预处理完成")
    
    def _compute_hetero_features_sparse(self, node_features):
        """计算异质邻居特征"""
        hetero_degrees = torch.sparse.sum(self.hetero_adj_sparse, dim=1).to_dense()
        hetero_degrees_safe = torch.clamp(hetero_degrees, min=1.0)
        
        if hasattr(node_features, 'cpu'):
            features_tensor = node_features.cpu().float()
        else:
            features_tensor = node_features.float()
            
        hetero_sum = torch.sparse.mm(self.hetero_adj_sparse, features_tensor)
        hetero_features = hetero_sum / hetero_degrees_safe.unsqueeze(1)
        
        no_hetero_mask = (hetero_degrees == 0)
        hetero_features[no_hetero_mask] = 0
        
        self.hetero_features = hetero_features.to(self.device)
        
        print(f"异质邻居特征统计:")
        print(f"   - 平均异质邻居数: {hetero_degrees.mean():.2f}")
        print(f"   - 无异质邻居节点数: {no_hetero_mask.sum()}")
        print(f"   - 异质邻居特征维度: {self.hetero_features.shape}")
    
    def _compute_lambda_weights_sparse(self, edge_index):
        """计算动态权重lambda"""
        row, col = self._get_edge_indices(edge_index)
        
        num_nodes = self.hetero_features.shape[0]
        
        unique_nodes, total_degrees = torch.unique(row, return_counts=True)
        total_degree_tensor = torch.zeros(num_nodes)
        total_degree_tensor[unique_nodes] = total_degrees.float()
        
        hetero_degrees = torch.sparse.sum(self.hetero_adj_sparse, dim=1).to_dense()
        homo_degrees = total_degree_tensor - hetero_degrees
        
        lambda_weights = homo_degrees - hetero_degrees + 1
        self.lambda_weights = lambda_weights.reshape(-1, 1).to(self.device, dtype=torch.float)
        
        print(f"Lambda权重统计: 均值={self.lambda_weights.mean():.2f}, 标准差={self.lambda_weights.std():.2f}")
    
    def train_f3_estimator(self, node_features,dataname, epochs=500, lr=0.01, test_ratio=0.1):
        """训练F3估计器"""

        if dataname == 'german':
            lr= 0.001
            self.is_preprocessed = True
        if not self.is_preprocessed:
            raise RuntimeError("请先调用 preprocess_hetero_neighbors() 进行预处理")
        
        node_features = node_features.to(self.device)
        
        valid_mask = torch.any(self.hetero_features != 0, dim=1)
        valid_features = node_features[valid_mask]
        valid_hetero_features = self.hetero_features[valid_mask]
        
        if len(valid_features) == 0:
            print("没有有效的异质邻居数据用于训练F3估计器")
            return
        
        indices = np.arange(len(valid_features))
        train_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=42)
        
        X_train = valid_features[train_idx]
        X_test = valid_features[test_idx]
        y_train = valid_hetero_features[train_idx]
        y_test = valid_hetero_features[test_idx]
        
        optimizer = torch.optim.Adam(list(self.f3_estimator.parameters()) + list(self.gating_network.parameters()),lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        
        best_val_loss = float('inf')
        best_state = None
        
        print(f"开始训练信息理论F3估计器 - 训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
        
        for epoch in range(epochs):
            self.f3_estimator.train()
            optimizer.zero_grad()
            
            pred_train = self.f3_estimator(X_train)
            train_loss = criterion(pred_train, y_train)
            
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            
            self.f3_estimator.eval()
            with torch.no_grad():
                pred_test = self.f3_estimator(X_test)
                val_loss = criterion(pred_test, y_test)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.f3_estimator.state_dict().copy()
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        if best_state is not None:
            self.f3_estimator.load_state_dict(best_state)
        
        print(f"信息理论F3估计器训练完成 - 最佳验证损失: {best_val_loss:.6f}")
    
    def enhance_features(self, node_features):
        """使用门控机制增强特征"""
        if not self.is_preprocessed:
            raise RuntimeError("请先调用 preprocess_hetero_neighbors() 进行预处理")

        node_features = node_features.to(self.device)

        self.f3_estimator.eval()
        self.gating_network.eval()
        with torch.no_grad():
            estimated_hetero = self.f3_estimator(node_features)

            # 为每个节点计算自适应的融合比例 delta
            adaptive_delta = self.gating_network(node_features)

            # 使用学习到的delta进行增强
            enhancement = adaptive_delta * estimated_hetero
            enhanced_features = node_features + enhancement

        return enhanced_features
    
    def get_detection_analysis(self):
        """获取详细的检测分析报告"""
        if not self.detection_summary:
            return "尚未进行异质邻居检测"
        
        detailed_scores = self.detection_summary['detailed_scores']
        training_info = self.detection_summary['training_info']
        
        analysis = f"""
信息理论异质邻居检测分析报告
{'='*60}

检测结果统计:
   - 异质边比例: {detailed_scores['hetero_ratio']*100:.1f}%
   - 最终阈值: {detailed_scores['final_threshold']:.4f}

信息理论指标:
   - 平均互信息: {training_info['avg_mutual_info']:.4f}
   - 平均条件熵: {training_info['avg_conditional_entropy']:.4f}
   - 平均度数熵: {training_info['avg_degree_entropy']:.4f}

融合权重分析:
   - 信息异质性得分范围: [{detailed_scores['fusion_scores']['info_heterogeneity'].min():.3f}, {detailed_scores['fusion_scores']['info_heterogeneity'].max():.3f}]
   - 平均融合权重: {detailed_scores['fusion_scores']['fusion_weights'].mean():.3f}
"""
        return analysis

    def clear_cache(self):
        """清理内存缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class AdversarialGenerator(nn.Module):
    """对抗样本生成器"""
    def __init__(self, input_dim, hidden_dim=128):
        super(AdversarialGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
    def forward(self, x, epsilon=0.1):
        """生成对抗扰动"""
        perturbation = self.network(x)
        perturbation = epsilon * perturbation
        return x + perturbation

class CounterfactualGenerator(nn.Module):
    """反事实样本生成器"""  
    def __init__(self, input_dim, latent_dim=64, hidden_dim=128):
        super(CounterfactualGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        """编码特征到潜在空间"""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar
    
    def decode(self, z, target_sens):
        """解码潜在表示和目标敏感属性到反事实特征"""
        z_with_target = torch.cat([z, target_sens.unsqueeze(-1)], dim=-1)
        return self.decoder(z_with_target)
    
    def forward(self, x, current_sens, target_sens):
        """生成反事实样本"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        counterfactual_x = self.decode(z, target_sens)
        
        recon_loss = F.mse_loss(self.decode(z, current_sens), x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        return counterfactual_x, recon_loss, kl_loss

class EnhancedSAP(nn.Module):
    """增强版SAP模块：集成对抗训练和反事实生成"""
    def __init__(self, in_dim, hid_dim, out_dim, encoder, layer_num, 
                 adversarial_weight=0.4, counterfactual_weight=0.1, 
                 device='cuda'):
        super(EnhancedSAP, self).__init__()
        
        self.variant_infer = nn.Sequential(
            nn.Linear(in_features=2*hid_dim, out_features=1),
            nn.Sigmoid()
        )
        self.sens_infer_backbone = ConstructModel(in_dim, hid_dim, encoder=encoder, layer_num=layer_num)
        self.sens_infer_classifier = nn.Sequential(
            nn.Linear(in_features=hid_dim+1, out_features=out_dim),
            nn.Softmax(dim=1)
        )
        
        self.adversarial_generator = AdversarialGenerator(hid_dim)
        self.counterfactual_generator = CounterfactualGenerator(hid_dim)
        
        self.discriminator = nn.Sequential(
            nn.Linear(hid_dim, hid_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid_dim//2, 1),
            nn.Sigmoid()
        )
        
        self.adversarial_weight = adversarial_weight
        self.counterfactual_weight = counterfactual_weight
        self.device = device
        
        self.prev_perf = None
        
        self.setup_optimizers()
        
        for m in self.modules():
            self.weights_init(m)
    
    def setup_optimizers(self):
        """设置各组件的优化器"""
        self.optimizer_main = torch.optim.Adam(
            list(self.variant_infer.parameters()) + 
            list(self.sens_infer_backbone.parameters()) + 
            list(self.sens_infer_classifier.parameters()),
            lr=0.001, weight_decay=1e-5
        )
        
        self.optimizer_adv_gen = torch.optim.Adam(
            self.adversarial_generator.parameters(), lr=0.001
        )
        
        self.optimizer_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.001
        )
        
        self.optimizer_cf = torch.optim.Adam(
            self.counterfactual_generator.parameters(), lr=0.001
        )

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, emb_cat, features, edge_index, labels, sens_attr=None, training=True, train_idx=None):
        """增强前向传播"""
        edge_weight_variant = self.variant_infer(emb_cat)
        edge_weight_variant = edge_weight_variant.squeeze()
        
        if hasattr(edge_index, 'storage'):
            edge_index_copy = edge_index.clone().fill_value(1., dtype=None)
            edge_index_copy.storage.set_value_(edge_index_copy.storage.value() * edge_weight_variant.to(edge_index_copy.device()))
            h = self.sens_infer_backbone(features, edge_index_copy)
        else:
            edge_weight = edge_weight_variant.to(edge_index.device)
            h = self.sens_infer_backbone(features, edge_index, edge_weight)
        
        sens_attr_partition = self.sens_infer_classifier(torch.cat([h, labels.unsqueeze(1)], dim=1))
        
        if not training or sens_attr is None or train_idx is None:
            return sens_attr_partition, 1-edge_weight_variant
        
        enhancement_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        try:
            h_train = h[train_idx]
            labels_train = labels[train_idx]
            sens_attr_train = sens_attr[train_idx]
            
            if self.adversarial_weight > 0:
                noise = 0.01 * torch.randn_like(h_train) #简单的高斯噪声  系数控制
                h_adv = h_train + noise
                #h_adv = self.adversarial_generator(h_train)  # 使用生成器生成对抗噪声
                adv_loss = F.mse_loss(h_adv, h_train.detach()) 
                enhancement_loss = enhancement_loss + self.adversarial_weight * adv_loss
            
            if self.counterfactual_weight > 0:
                sens_attr_float = sens_attr_train.float()
                regularization = torch.var(h_train.mean(dim=1) * sens_attr_float)
                enhancement_loss = enhancement_loss + self.counterfactual_weight * regularization
                
        except Exception as e:
            print(f"简化增强训练失败: {e}")
            enhancement_loss = torch.tensor(0.0, device=self.device)
        
        return sens_attr_partition, 1-edge_weight_variant, enhancement_loss

    def get_feature_analysis(self, emb_cat, features, edge_index, labels, sens_attr, train_idx=None):
        """获取特征分析结果"""
        try:
            self.eval()
            with torch.no_grad():
                edge_weight_variant = self.variant_infer(emb_cat)
                edge_weight_variant = edge_weight_variant.squeeze()
                
                if hasattr(edge_index, 'storage'):
                    edge_index_copy = edge_index.clone().fill_value(1., dtype=None)
                    edge_index_copy.storage.set_value_(edge_index_copy.storage.value() * edge_weight_variant.to(edge_index_copy.device()))
                    h = self.sens_infer_backbone(features, edge_index_copy)
                else:
                    edge_weight = edge_weight_variant.to(edge_index.device)
                    h = self.sens_infer_backbone(features, edge_index, edge_weight)
                
                if train_idx is not None:
                    h_analysis = h[train_idx]
                    sens_attr_analysis = sens_attr[train_idx]
                else:
                    h_analysis = h
                    sens_attr_analysis = sens_attr
                
                sens_0_mask = (sens_attr_analysis == 0)
                sens_1_mask = (sens_attr_analysis == 1)
                
                if sens_0_mask.sum() > 0 and sens_1_mask.sum() > 0:
                    h_sens_0 = h_analysis[sens_0_mask].mean(dim=0)
                    h_sens_1 = h_analysis[sens_1_mask].mean(dim=0)
                    feature_diff = torch.norm(h_sens_1 - h_sens_0).item()
                else:
                    feature_diff = 0.0
                
                return {
                    'feature_difference': feature_diff,
                    'sens_0_count': sens_0_mask.sum().item(),
                    'sens_1_count': sens_1_mask.sum().item(),
                    'edge_weights': edge_weight_variant
                }
        except Exception as e:
            print(f"特征分析出错: {e}")
            return {
                'feature_difference': 0.0,
                'sens_0_count': 0,
                'sens_1_count': 0,
                'edge_weights': None
            }

    def update_weights(self, adversarial_weight=None, counterfactual_weight=None):
        """动态调整权重"""
        if adversarial_weight is not None:
            self.adversarial_weight = adversarial_weight
        if counterfactual_weight is not None:
            self.counterfactual_weight = counterfactual_weight

class FairPHM(nn.Module):
    def __init__(self, args):
        super(FairPHM, self).__init__()
        self.in_dim = args.in_dim
        self.hid_dim = args.hid_dim
        self.out_dim = args.out_dim
        self.args = args
        print(f" 初始化 FairPHM 模块")
        
        # 添加投影头，用于公平性增强中的对比学习
        self.projection_head = nn.Sequential(
            nn.Linear(args.hid_dim, args.hid_dim),
            nn.ReLU(),
            nn.Linear(args.hid_dim, args.hid_dim)
        ).to(args.device)
        
        self.beta_contrastive = getattr(args, 'beta_contrastive', 0.5)
        
        f3_config = getattr(args, 'f3_config', {})
        print(f"初始化信息理论F3模块，设备: {args.device}")
        
        dataset_name = getattr(args, 'dataset', 'unknown')
        use_fast_mi = f3_config.get('use_fast_mi', True) #生效
               
        self.f3_module = EnhancedUnsupervisedF3Module(
            input_dim=args.in_dim,
            hidden_dim=f3_config.get('hidden_dim', args.hid_dim),
            device=args.device,
            delta=f3_config.get('delta',10),
            use_lambda_weighting=f3_config.get('use_lambda_weighting', False),
            similarity_threshold=f3_config.get('similarity_threshold', 0.6),
            method='info_theory_only',
            adaptive_threshold=f3_config.get('adaptive_threshold', True),
            use_fast_mi=use_fast_mi
        )
        
        # F3预处理完成标志和缓存
        self.f3_preprocessed = False
        self.cached_enhanced_features = None
        
        gnn_backbone = ConstructModel(args.in_dim, args.hid_dim, args.encoder, args.layer_num)
        self.gnn_backbone = gnn_backbone.to(args.device)
        self.classifier = nn.Linear(args.hid_dim, args.out_dim).to(args.device)
        
        #将投影头的参数加入优化器（用于公平性对比学习）
        self.optimizer_infer = torch.optim.Adam(
            list(self.gnn_backbone.parameters()) +
            list(self.classifier.parameters()) +
            list(self.projection_head.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
        
        for m in self.modules():
            self.weights_init(m)

        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.criterion_irm = IRMLoss()
        self.criterion_env = nn.BCEWithLogitsLoss(reduction='none')
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


    def info_nce_loss(self, features, labels, temperature=0.07):
        """InfoNCE对比损失 - 用于公平性增强"""
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask.fill_diagonal_(0)
        
        exp_sim = torch.exp(similarity_matrix)
        neg_sum = torch.sum(exp_sim * (1 - mask), dim=1, keepdim=True)
        denominator = neg_sum + torch.diag(exp_sim).unsqueeze(1)
        
        log_prob = similarity_matrix - torch.log(denominator.clamp(min=1e-9))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        loss = -mean_log_prob_pos.mean()
        return loss

    def _monitor_memory(self, stage=""):
        """内存监控"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"{stage} GPU内存 - 已用: {allocated:.2f}GB, 缓存: {reserved:.2f}GB")

    def preprocess_f3(self, data):
        """信息理论F3模块预处理"""
        if not self.f3_preprocessed:
            print("信息理论F3预处理...")
            self._monitor_memory("F3预处理前")
            
            self.f3_module.preprocess_hetero_neighbors(
                edge_index=data.edge_index,
                node_features=data.features,
                dataset_name=getattr(self.args, 'dataset', 'custom'),
                sensitive_attrs=None
            )
            
            self._monitor_memory("异质邻居预处理后")
            
            self.f3_module.train_f3_estimator(
                node_features=data.features,
                dataname=getattr(self.args, 'dataset', 'custom'),
                epochs=getattr(self.args, 'f3_epochs', 50)
            )
            
            print("预计算增强特征...")
            self.cached_enhanced_features = self.f3_module.enhance_features(data.features)
            
            self.f3_preprocessed = True
            self._monitor_memory("F3预处理完成后")
            
            # 打印检测分析
            analysis = self.f3_module.get_detection_analysis()
            if analysis != "尚未进行异质邻居检测":
                print(analysis)
            
            self.f3_module.clear_cache()
            print("信息理论F3预处理完成")

    def train_model(self, data, **kwargs):
        # 首先进行F3预处理
        self.preprocess_f3(data)
        
        best_loss = 100
        best_result = 0
        pbar = kwargs.get('pbar', None)
        
        part_mat_list, edge_weight_inv_list = [], []

        if hasattr(self.args, 'seed_dir'):
            writer = LogWriter(self.args.seed_dir)
        
        print("开始分区阶段...")
        for i in range(self.args.partition_times):
            part_mat, edge_weight_inv = self.sens_partition(data, writer if 'writer' in locals() else None)
            part_mat_list.append(part_mat[data.idx_train])
            edge_weight_inv_list.append(edge_weight_inv)
            self._monitor_memory(f"分区{i+1}完成")

        print("开始主训练循环...")
        for epoch in range(self.args.epochs):
            loss_log_list = []

            self.gnn_backbone.train()
            self.classifier.train()
            self.projection_head.train()  # 设置投影头为训练模式
            self.optimizer_infer.zero_grad()

            enhanced_features = self.cached_enhanced_features

            for i, edge_weight_inv in enumerate(edge_weight_inv_list):
                edge_weight_inv = edge_weight_inv.squeeze()
                
                if hasattr(data.edge_index, 'storage'):
                    edge_weight_copy = data.edge_index.clone()
                    edge_weight_copy = edge_weight_copy.fill_value(1., dtype=None)
                    edge_weight_copy.storage.set_value_(edge_weight_copy.storage.value() * edge_weight_inv.to(edge_weight_copy.device()))
                    
                    emb = self.gnn_backbone(enhanced_features, edge_weight_copy)
                else:
                    edge_weight = edge_weight_inv.to(data.edge_index.device)
                    emb = self.gnn_backbone(enhanced_features, data.edge_index, edge_weight)
                output = self.classifier(emb)

                loss_cls = self.criterion_cls(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())
                
                group_assign = part_mat_list[i].argmax(dim=1)
                for j in range(part_mat_list[i].shape[-1]):
                    select_idx = torch.where(group_assign == j)[0]
                    if len(select_idx) == 0:
                        continue

                    sub_logits = output[data.idx_train][select_idx]
                    sub_labels = data.labels[data.idx_train][select_idx]

                    loss_log = self.criterion_cls(sub_logits, sub_labels.unsqueeze(1).float())
                    loss_log_list.append(loss_log.view(-1))

            # 损失计算与组合（包含公平性对比学习）
            # 1. 计算不变损失
            loss_log_cat = torch.cat(loss_log_list, dim=0)
            Var, Mean = torch.var_mean(loss_log_cat)
            loss_inv = Var + self.args.alpha * Mean

            # 2. 计算对比损失（用于公平性增强）
            with torch.no_grad():
                emb_base = self.gnn_backbone(enhanced_features, data.edge_index)
            z = self.projection_head(emb_base[data.idx_train])
            loss_con = self.info_nce_loss(z, data.labels[data.idx_train])

            # 3. 组合总损失
            loss_train = loss_inv + self.beta_contrastive * loss_con

            loss_train.backward()
            self.optimizer_infer.step()

            self.gnn_backbone.eval()
            self.classifier.eval()
            self.projection_head.eval()
            with torch.no_grad():
                emb_val = self.gnn_backbone(self.cached_enhanced_features, data.edge_index)
                output_val = self.classifier(emb_val)

            loss_cls_val = self.criterion_cls(output_val[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float())
            pred = (output_val.squeeze() > 0).type_as(data.labels)
            
            auc_val = roc_auc_score(data.labels[data.idx_val].cpu(), output_val[data.idx_val].cpu())
            f1_val = f1_score(data.labels[data.idx_val].cpu(), pred[data.idx_val].cpu())
            acc_val = accuracy_score(data.labels[data.idx_val].cpu(), pred[data.idx_val].cpu())
            
            parity_val, equality_val = fair_metric(pred[data.idx_val].cpu().numpy(),
                                                   data.labels[data.idx_val].cpu().numpy(),
                                                   data.sens[data.idx_val].cpu().numpy())
            
            if self.args.dataset in ['pokec_z', 'pokec_n']:
                if loss_cls_val.item() < best_loss:
                    best_loss = loss_cls_val.item()
                    torch.save(self.state_dict(), f'./weights/FairINV_InfoTheoryF3_{self.args.encoder}.pt')
            else:
                if auc_val-parity_val-equality_val > best_result:
                    best_result = auc_val-parity_val-equality_val
                    torch.save(self.state_dict(), f'./weights/FairINV_InfoTheoryF3_{self.args.encoder}.pt')

            if 'writer' in locals():
                # 更新日志记录（包含对比损失）
                writer.record(loss_item={'train/loss_inv': loss_inv.item(), 
                                         'train/loss_con': loss_con.item(),
                                         'train/loss_all': loss_train.item()}, step=epoch)
                writer.record(loss_item={'val/auc': auc_val, 'val/f1': f1_val, 'val/acc': acc_val,
                                         'val/dp': parity_val, 'val/eo': equality_val}, step=epoch)

            if pbar is not None:
                # 更新进度条显示
                pbar.set_postfix({'loss_train': "{:.4f}".format(loss_train.item()), 
                                  'loss_inv': "{:.4f}".format(loss_inv.item()),
                                  'loss_con': "{:.4f}".format(loss_con.item())})
                pbar.update(1)

            if epoch % 100 == 0:
                self.f3_module.clear_cache()

        if pbar is not None:
            pbar.close()

    def sens_partition(self, data, writer):
        ref_backbone, ref_classifier = self.train_ref_model(data, writer)

        partition_module = EnhancedSAP(
            in_dim=self.in_dim, 
            hid_dim=self.hid_dim, 
            out_dim=self.args.env_num, 
            encoder=self.args.encoder, 
            layer_num=self.args.layer_num,
            adversarial_weight=getattr(self.args, 'adversarial_weight', 0.4),
            counterfactual_weight=getattr(self.args, 'counterfactual_weight', 0.1),
            device=self.args.device
        ).to(self.args.device)
        
        emb = ref_backbone(self.cached_enhanced_features, data.edge_index)
        
        logits = ref_classifier(emb)
        
        scale = torch.tensor(1.).to(self.args.device).requires_grad_()
        error = self.criterion_env(logits[data.idx_train]*scale,
                                   data.labels[data.idx_train].unsqueeze(1).float())
        
        # 处理不同类型的edge_index来构建emb_cat
        if hasattr(data.edge_index, 'storage'):  # SparseTensor类型
            row, col = data.edge_index.storage._row, data.edge_index.storage._col
        else:  # 普通tensor类型
            row, col = data.edge_index[0], data.edge_index[1]
        
        emb_cat = torch.cat([emb[row], emb[col]], dim=1)
        
        sens_attr = getattr(data, 'sens', None)
        if sens_attr is not None:
            sens_attr_train = sens_attr[data.idx_train]
        else:
            sens_attr_train = None

        print("开始训练增强SAP模块...")
        for epoch in range(500):
            loss_penalty_list = []
            partition_module.train()
            
            if sens_attr_train is not None:
                part_mat, edge_weight_inv, enhancement_loss = partition_module(
                    emb_cat.detach(), 
                    self.cached_enhanced_features, 
                    data.edge_index, 
                    data.labels,
                    sens_attr,
                    training=True,
                    train_idx=data.idx_train
                )
            else:
                part_mat, edge_weight_inv = partition_module(
                    emb_cat.detach(), 
                    self.cached_enhanced_features, 
                    data.edge_index, 
                    data.labels,
                    training=False
                )
                enhancement_loss = torch.tensor(0.0, device=self.args.device)
            
            for env_idx in range(self.args.env_num):
                loss_weight = part_mat[:, env_idx]
                penalty_grad = grad((error.squeeze(1) * loss_weight[data.idx_train]).mean(), 
                                  [scale], create_graph=True)[0].pow(2).mean()
                loss_penalty_list.append(penalty_grad)

            risk_final = -torch.stack(loss_penalty_list).sum()
            
            if isinstance(enhancement_loss, torch.Tensor) and enhancement_loss.item() != 0:
                total_loss = risk_final + 0.1 * enhancement_loss
            else:
                total_loss = risk_final
            
            total_loss.backward(retain_graph=True)
            
            partition_module.optimizer_main.step()
            partition_module.optimizer_main.zero_grad()
                
            if writer is not None and epoch % 50 == 0:
                log_dict = {'enhanced_sap/risk_final': risk_final.item()}
                if isinstance(enhancement_loss, torch.Tensor) and enhancement_loss.item() != 0:
                    log_dict['enhanced_sap/enhancement_loss'] = enhancement_loss.item()
                writer.record(loss_item=log_dict, step=epoch)
            
            if epoch % 10 == 0 and epoch > 0:
                current_perf = -risk_final.item()
                if hasattr(partition_module, 'prev_perf') and partition_module.prev_perf is not None:
                    if current_perf > partition_module.prev_perf:
                        partition_module.update_weights(
                            adversarial_weight=min(0.3, partition_module.adversarial_weight * 1.1),
                            counterfactual_weight=min(0.2, partition_module.counterfactual_weight * 1.1)
                        )
                    else:
                        partition_module.update_weights(
                            adversarial_weight=max(0.05, partition_module.adversarial_weight * 0.9),
                            counterfactual_weight=max(0.05, partition_module.counterfactual_weight * 0.9)
                        )
                partition_module.prev_perf = current_perf

        with torch.no_grad():
            if sens_attr_train is not None:
                result = partition_module(
                    emb_cat.detach(), 
                    self.cached_enhanced_features, 
                    data.edge_index, 
                    data.labels,
                    sens_attr,
                    training=False,
                    train_idx=data.idx_train
                )
                soft_split_final, edge_weight_inv = result
            else:
                result = partition_module(
                    emb_cat.detach(), 
                    self.cached_enhanced_features, 
                    data.edge_index, 
                    data.labels,
                    training=False
                )
                soft_split_final, edge_weight_inv = result
        
        print("增强SAP模块训练完成")
        
        if sens_attr_train is not None:
            try:
                feature_analysis = partition_module.get_feature_analysis(
                    emb_cat.detach(), 
                    self.cached_enhanced_features, 
                    data.edge_index, 
                    data.labels, 
                    sens_attr,
                    train_idx=data.idx_train
                )
                print("增强SAP训练统计:")
                print(f"  - 对抗权重: {partition_module.adversarial_weight:.3f}")
                print(f"  - 反事实权重: {partition_module.counterfactual_weight:.3f}")
                print(f"  - 特征差异: {feature_analysis['feature_difference']:.4f}")
                print(f"  - 敏感属性分布: {feature_analysis['sens_0_count']} vs {feature_analysis['sens_1_count']}")
            except Exception as e:
                print(f"统计信息输出失败: {e}")
        
        return soft_split_final, edge_weight_inv

    def train_ref_model(self, data, writer=None):
        """训练参考模型"""
        ref_backbone = ConstructModel(self.args.in_dim, self.args.hid_dim, self.args.encoder,
                                       self.args.layer_num).to(self.args.device)
        ref_classifier = nn.Linear(self.args.hid_dim, self.args.out_dim).to(self.args.device)
        optimizer_part = torch.optim.Adam(list(ref_backbone.parameters()) + list(ref_classifier.parameters()),
                                          lr=self.args.lr, weight_decay=self.args.weight_decay)

        print("开始训练参考模型...")
        for epoch in range(500):
            ref_backbone.train()
            ref_classifier.train()
            optimizer_part.zero_grad()

            emb = ref_backbone(data.features, data.edge_index)
            output = ref_classifier(emb)
            loss_train = self.criterion_cls(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float())

            loss_train.backward()
            optimizer_part.step()
            if writer is not None and epoch % 50 == 0:
                writer.record(loss_item={'pre-train/cls_loss': loss_train.item()}, step=epoch)
            ref_backbone.eval()
            ref_classifier.eval()

        print("参考模型训练完成")
        return ref_backbone, ref_classifier

    def forward(self, x, edge_index):
        if self.f3_preprocessed and self.cached_enhanced_features is not None:
            enhanced_x = self.cached_enhanced_features
        else:
            enhanced_x = x
            
        emb = self.gnn_backbone(enhanced_x, edge_index)
        output = self.classifier(emb)
        return output

class IRMLoss(_Loss):
    def __init__(self):
        super(IRMLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, updated_split_all):
        env_penalty = []
        for updated_split_each in updated_split_all:
            per_penalty = []
            env_num = updated_split_each.shape[-1]
            group_assign = updated_split_each.argmax(dim=1)

            for env in range(env_num):
                select_idx = torch.where(group_assign == env)[0]
                
                if len(select_idx) == 0: 
                    continue
                    
                sub_logits = logits[select_idx]
                sub_label = labels[select_idx]
                scale_dummy = torch.tensor(1.).to(device).requires_grad_()
                loss_env = self.loss(sub_logits * scale_dummy, sub_label)
                loss_grad = grad(loss_env, [scale_dummy], create_graph=True)[0]
                per_penalty.append(torch.sum(loss_grad ** 2))
            
            if len(per_penalty) > 0:
                env_penalty.append(torch.stack(per_penalty).mean())
        
        if len(env_penalty) == 0: 
            return torch.tensor(0.0).to(logits.device)
            
        loss_penalty = torch.stack(env_penalty).mean()
        return loss_penalty

class ConstructModel(nn.Module):
    def __init__(self, in_dim, hid_dim, encoder, layer_num):
        super(ConstructModel, self).__init__()
        self.encoder = encoder
        
        if encoder == 'gcn':
            self.model = nn.ModuleList()
            self.model.append(GCNConv(in_dim, hid_dim))
            for i in range(layer_num - 1):
                self.model.append(GCNConv(hid_dim, hid_dim))
        elif encoder == 'gin':
            self.model = GIN(nfeat=in_dim, nhid=hid_dim, dropout=0.5)
        elif encoder == 'sage':
            self.model = SAGE(nfeat=in_dim, nhid=hid_dim, dropout=0.5)
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        if self.encoder == 'gcn':
            for i, layer in enumerate(self.model):
                h = layer(h, edge_index, edge_weight=edge_weight)
                if i < len(self.model) - 1:
                    h = F.relu(h)
        elif self.encoder in ['gin', 'sage']:
            try:
                h = self.model(x, edge_index, edge_weight)
            except:
                h = self.model(x, edge_index)
        return h

class GIN(nn.Module):
    def __init__(self, nfeat, nhid, dropout): 
        super(GIN, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(nfeat, nhid), 
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Linear(nhid, nhid), 
        )
        self.conv1 = GINConv(self.mlp1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, x, edge_index, edge_weight=None): 
        x = self.conv1(x, edge_index)
        return x

class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, dropout): 
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv1.aggr = 'mean'
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Dropout(p=dropout)
        )
        self.conv2 = SAGEConv(nhid, nhid, normalize=True)
        self.conv2.aggr = 'mean'
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight=None): 
        if edge_weight is not None:
            try:
                x = self.conv1(x, edge_index, edge_weight=edge_weight)
                x = self.transition(x)
                x = self.conv2(x, edge_index, edge_weight=edge_weight)
            except:
                x = self.conv1(x, edge_index)
                x = self.transition(x)
                x = self.conv2(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = self.transition(x)
            x = self.conv2(x, edge_index)
        return x