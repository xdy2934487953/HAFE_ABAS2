import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class TypeAwareGCNConv(MessagePassing):
    """
    类型感知的图卷积层
    
    为不同类型的边学习不同的权重矩阵
    """
    
    def __init__(self, in_channels, out_channels, num_edge_types=4, 
                 aggr='add', bias=True, **kwargs):
        """
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            num_edge_types: 边类型数量（默认4：OPINION, SYNTAX_CORE, COREF, OTHER）
            aggr: 聚合方式
        """
        super(TypeAwareGCNConv, self).__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        
        # 为每种边类型创建独立的权重矩阵
        self.weight_matrices = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])
        
        # 边类型的重要性权重（可学习）
        self.edge_importance = nn.Parameter(torch.ones(num_edge_types))
        
        # 自环的权重
        self.weight_self = nn.Linear(in_channels, out_channels, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        for weight in self.weight_matrices:
            nn.init.xavier_uniform_(weight.weight)
        nn.init.xavier_uniform_(self.weight_self.weight)
        
        # 初始化边类型重要性（Opinion边最重要）
        with torch.no_grad():
            self.edge_importance[0] = 2.0  # OPINION
            self.edge_importance[1] = 1.5  # SYNTAX_CORE
            self.edge_importance[2] = 1.0  # COREF
            self.edge_importance[3] = 0.5  # OTHER
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_types=None):
        """
        前向传播
        
        Args:
            x: [num_nodes, in_channels] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_types: [num_edges] 边类型（如果为None，则所有边视为同一类型）
        
        Returns:
            out: [num_nodes, out_channels] 输出特征
        """
        # 如果没有提供边类型，默认所有边为OTHER类型
        if edge_types is None:
            edge_types = torch.full((edge_index.shape[1],), 3, 
                                   dtype=torch.long, device=x.device)
        
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 为自环添加一个特殊类型（使用-1标记）
        self_loop_types = torch.full((x.size(0),), -1, 
                                    dtype=torch.long, device=x.device)
        edge_types = torch.cat([edge_types, self_loop_types])
        
        # 归一化（按度数）
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 按边类型分别处理
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        
        for edge_type in range(self.num_edge_types):
            # 筛选该类型的边
            mask = (edge_types == edge_type)
            
            if mask.sum() == 0:
                continue
            
            type_edge_index = edge_index[:, mask]
            type_norm = norm[mask]
            
            # 该类型的变换
            x_transformed = self.weight_matrices[edge_type](x)
            
            # 消息传递
            out_type = self.propagate(
                type_edge_index, 
                x=x_transformed, 
                norm=type_norm
            )
            
            # 加权累加（使用可学习的重要性权重）
            out += out_type * self.edge_importance[edge_type]
        
        # 处理自环
        self_loop_mask = (edge_types == -1)
        if self_loop_mask.sum() > 0:
            out += self.weight_self(x)
        
        # 添加偏置
        if self.bias is not None:
            out += self.bias
        
        return out
    
    def message(self, x_j, norm):
        """消息函数：带归一化的特征"""
        return norm.view(-1, 1) * x_j
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, num_types={self.num_edge_types})'