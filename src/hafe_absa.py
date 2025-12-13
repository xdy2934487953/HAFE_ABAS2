import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv  # 移除标准GCN
from type_aware_gcn import TypeAwareGCNConv  # 导入类型感知GCN
from torch_sparse import SparseTensor
import hashlib

import sys
sys.path.append('.')
from fairPHM import InformationTheoryProcessor, InfoTheoryHeteroDetector, EnhancedUnsupervisedF3Module

class HAFE_ABSA_Model(nn.Module):
    """HAFE-ABSA完整模型（改进版：类型感知GCN）"""
    
    def __init__(self, input_dim, hidden_dim, num_classes=3, device='cuda', 
                 dataset_name='absa', use_type_aware=True):
        super(HAFE_ABSA_Model, self).__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dataset_name = dataset_name
        self.use_type_aware = use_type_aware
        
        # HAFE F3模块
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
        
        # ===== 修改：使用类型感知GCN =====
        if use_type_aware:
            print("使用类型感知GCN (Type-Aware GCN)")
            self.gcn1 = TypeAwareGCNConv(input_dim, hidden_dim, num_edge_types=4)
            self.gcn2 = TypeAwareGCNConv(hidden_dim, hidden_dim, num_edge_types=4)
        else:
            print("使用标准GCN")
            from torch_geometric.nn import GCNConv
            self.gcn1 = GCNConv(input_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # 情感分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
        # F3预处理标志
        self.f3_preprocessed = False
    
    def _compute_graph_hash(self, num_nodes, num_edges):
        """计算图的哈希值，用于缓存验证"""
        hash_str = f"{self.dataset_name}_{num_nodes}_{num_edges}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]
        
    def preprocess_f3(self, all_graphs):
        """F3预处理：检测异质边并训练估计器"""
        if self.f3_preprocessed:
            return
        
        print("\n开始HAFE-ABSA预处理...")
        
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
        
        # 生成唯一的缓存名称
        graph_hash = self._compute_graph_hash(num_nodes, num_edges)
        cache_name = f"{self.dataset_name}_{graph_hash}"
        
        print(f"缓存标识: {cache_name}")
        
        # 转换为SparseTensor格式
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
            
            wrapped_edge_index = EdgeIndexWrapper(combined_edge_index)
        else:
            wrapped_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        
        # 异质邻居检测
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
        print("HAFE-ABSA预处理完成!\n")
    
    def forward(self, features, edge_index, aspect_indices, edge_types=None):
        """
        前向传播
        
        Args:
            features: [num_nodes, input_dim]
            edge_index: [2, num_edges]
            aspect_indices: [num_aspects] aspect节点的索引
            edge_types: [num_edges] 边类型（新增参数）
            
        Returns:
            logits: [num_aspects, num_classes]
        """
        # 1. HAFE特征增强
        if self.f3_preprocessed:
            enhanced_features = self.f3_module.enhance_features(features)
        else:
            enhanced_features = features
        
        # 2. GNN编码（传入edge_types）
        if self.use_type_aware:
            h = self.gcn1(enhanced_features, edge_index, edge_types)
        else:
            h = self.gcn1(enhanced_features, edge_index)
        
        h = F.relu(h)
        h = self.dropout(h)
        
        if self.use_type_aware:
            h = self.gcn2(h, edge_index, edge_types)
        else:
            h = self.gcn2(h, edge_index)
        
        # 3. 根据aspect_indices提取对应节点的embedding
        aspect_embeddings = h[aspect_indices]
        
        # 4. 情感分类
        logits = self.classifier(aspect_embeddings)
        
        return logits


class BaselineASGCN(nn.Module):
    """Baseline ASGCN模型（也可选类型感知）"""
    
    def __init__(self, input_dim, hidden_dim, num_classes=3, use_type_aware=False):
        super(BaselineASGCN, self).__init__()
        
        self.use_type_aware = use_type_aware
        
        if use_type_aware:
            print("Baseline使用类型感知GCN")
            self.gcn1 = TypeAwareGCNConv(input_dim, hidden_dim, num_edge_types=4)
            self.gcn2 = TypeAwareGCNConv(hidden_dim, hidden_dim, num_edge_types=4)
        else:
            print("Baseline使用标准GCN")
            from torch_geometric.nn import GCNConv
            self.gcn1 = GCNConv(input_dim, hidden_dim)
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, features, edge_index, aspect_indices, edge_types=None):
        if self.use_type_aware:
            h = self.gcn1(features, edge_index, edge_types)
        else:
            h = self.gcn1(features, edge_index)
        
        h = F.relu(h)
        h = self.dropout(h)
        
        if self.use_type_aware:
            h = self.gcn2(h, edge_index, edge_types)
        else:
            h = self.gcn2(h, edge_index)
        
        aspect_embeddings = h[aspect_indices]
        logits = self.classifier(aspect_embeddings)
        
        return logits