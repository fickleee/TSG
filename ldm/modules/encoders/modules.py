# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import copy
import math
       
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class ResBlockTime(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlockTime, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv1d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm1d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class DomainUnifiedEncoder(nn.Module):
    '''
    The input are encoded into two parts, invariant part and specific part. The specific part is generated attending to a random initialized latent vector pool.
    The length of the two part are equal in this implementation.
    '''
    def __init__(self, dim, window, num_channels=3, latent_dim=32, bn=True, **kwargs):
        super().__init__()
        dim_out = latent_dim
        flatten_dim = int(dim * window / 4)
        self.in_encoder = nn.Sequential(
            nn.Conv1d(num_channels, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
            )

        self.out_encoder = nn.Sequential(
            ResBlockTime(dim, dim, bn=bn),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            ResBlockTime(dim, dim, bn=bn),
            View((-1, flatten_dim)),                  # batch_size x 2048
            nn.Linear(flatten_dim, dim_out)
        )
            
    def forward(self, x):
        h = self.in_encoder(x)
        mask = None

        out = self.out_encoder(h)[:,None]   # b, 1, d
        return out, mask

class SimpleGCNLayer(nn.Module):
    """
    简单的图卷积层实现（不依赖PyTorch Geometric）
    实现全连接图的GCN操作
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x, adj=None):
        """
        Args:
            x: (N, K) 节点特征，N是节点数（变量数），K是特征维度
            adj: (N, N) 邻接矩阵，如果为None则使用全连接图
        Returns:
            out: (N, K) 输出特征
        """
        # 如果adj为None，使用全连接图（所有节点互相连接）
        if adj is None:
            N = x.shape[0]  # 节点数
            adj = torch.ones(N, N, device=x.device, dtype=x.dtype)
            adj = adj - torch.eye(N, device=x.device, dtype=x.dtype)  # 移除自连接
        
        # 归一化邻接矩阵（对称归一化）
        # D^(-1/2) * A * D^(-1/2)
        rowsum = adj.sum(dim=1, keepdim=True)
        d_inv_sqrt = torch.pow(rowsum + 1e-8, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
        adj_norm = d_inv_sqrt * adj * d_inv_sqrt.t()
        
        # 图卷积：H' = A_norm * H * W
        # 先做线性变换：H * W -> (N, out_features)
        support = torch.mm(x, self.weight.t())  # (N, out_features)
        # 再做图卷积：A_norm * support
        output = torch.mm(adj_norm, support)  # (N, out_features)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class DomainUnifiedPrototyper(nn.Module):
    '''
    The input are encoded into two parts, invariant part and specific part. The specific part is generated attending to a random initialized latent vector pool.
    The length of the two part are equal in this implementation.
    
    Enhanced with GNN support for multi-variable interaction.
    '''
    def __init__(self, dim, window, num_latents=16, num_channels=3, latent_dim=32, bn=True, 
                 use_gnn=False, num_variables=None, gnn_layers=2, gnn_hidden_dim=None, 
                 gnn_type='simple_gcn', **kwargs):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.num_variables = num_variables
        # use_gnn: 如果num_variables>1则启用，或者如果use_gnn=True则允许在forward时动态启用
        self.use_gnn = use_gnn  # 不在初始化时禁用，允许forward时动态处理
        self.gnn_type = gnn_type
        
        flatten_dim = int(dim * window / 4)
        
        # 共享编码器
        self.share_encoder = nn.Sequential(
            nn.Conv1d(num_channels, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
            )
        
        # 原型参数
        self.latents = nn.Parameter(torch.empty(num_latents, self.latent_dim), requires_grad=False)
        nn.init.orthogonal_(self.latents)
        self.init_latents = copy.deepcopy(self.latents.detach())
        
        # PAM权重生成网络
        self.mask_ffn = nn.Sequential(
            ResBlockTime(dim, dim, bn=bn),
            View((-1, flatten_dim)),                  # batch_size x 2048
            nn.Linear(flatten_dim, self.num_latents),
        )
        self.sigmoid = nn.Sigmoid()
        
        # GNN层（如果启用）
        if self.use_gnn:
            gnn_hidden_dim = gnn_hidden_dim or num_latents
            
            if gnn_type == 'simple_gcn':
                # 使用简单的GCN实现（不依赖外部库）
                import math
                self.gnn_layers = nn.ModuleList()
                for i in range(gnn_layers):
                    in_dim = num_latents if i == 0 else gnn_hidden_dim
                    out_dim = num_latents if i == gnn_layers - 1 else gnn_hidden_dim
                    self.gnn_layers.append(SimpleGCNLayer(in_dim, out_dim))
            else:
                # 尝试使用PyTorch Geometric（如果可用）
                try:
                    from torch_geometric.nn import GCNConv, GATConv
                    if gnn_type == 'gcn':
                        self.gnn_layers = nn.ModuleList([
                            GCNConv(num_latents if i == 0 else gnn_hidden_dim,
                                   num_latents if i == gnn_layers - 1 else gnn_hidden_dim)
                            for i in range(gnn_layers)
                        ])
                    elif gnn_type == 'gat':
                        self.gnn_layers = nn.ModuleList([
                            GATConv(num_latents if i == 0 else gnn_hidden_dim,
                                   num_latents if i == gnn_layers - 1 else gnn_hidden_dim,
                                   heads=4, concat=False)
                            for i in range(gnn_layers)
                        ])
                    else:
                        raise ValueError(f"Unknown GNN type: {gnn_type}")
                except ImportError:
                    print("Warning: PyTorch Geometric not available. Falling back to simple_gcn.")
                    import math
                    self.gnn_layers = nn.ModuleList([
                        SimpleGCNLayer(num_latents if i == 0 else gnn_hidden_dim,
                                      num_latents if i == gnn_layers - 1 else gnn_hidden_dim)
                        for i in range(gnn_layers)
                    ])
            
            if num_variables is not None and num_variables > 1:
                print(f"GNN enabled: type={gnn_type}, layers={gnn_layers}, fixed_num_variables={num_variables}")
            else:
                print(f"GNN enabled: type={gnn_type}, layers={gnn_layers}, dynamic_num_variables (from data_key)")
        else:
            self.gnn_layers = None
            
    def _build_adjacency_matrix(self, num_nodes, graph_type='fully_connected'):
        """
        构建图的邻接矩阵
        Args:
            num_nodes: 节点数（变量数）
            graph_type: 图类型，'fully_connected' 表示全连接图
        Returns:
            adj: (num_nodes, num_nodes) 邻接矩阵
        """
        if graph_type == 'fully_connected':
            # 全连接图：所有节点互相连接
            adj = torch.ones(num_nodes, num_nodes)
            adj = adj - torch.eye(num_nodes)  # 移除自连接
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        
        return adj
    
    def _build_edge_index(self, num_nodes, batch_size, device):
        """
        构建边索引（用于PyTorch Geometric）
        Args:
            num_nodes: 每个batch的节点数
            batch_size: batch大小
            device: 设备
        Returns:
            edge_index: (2, num_edges) 边索引
        """
        edge_list = []
        for b in range(batch_size):
            node_offset = b * num_nodes
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # 全连接，移除自连接
                        edge_list.append([node_offset + i, node_offset + j])
        
        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
        return edge_index
            
    def forward(self, x, data_key=None, num_variables_dict=None):
        """
        Args:
            x: (B*N, T, C) 输入数据
            data_key: (B*N,) 数据集标识，用于确定每个样本属于哪个数据集
            num_variables_dict: {data_key: num_variables} 每个数据集的变量数映射
        """
        b_total = x.shape[0]  # B*N（batch和变量flattened）
        
        # 1. 编码阶段
        h = self.share_encoder(x)  # (B*N, dim, T')
        mask_logit_init = self.mask_ffn(h)  # (B*N, K) - 初始PAM权重
        
        # 2. GNN交互阶段（如果启用）
        if self.use_gnn and self.gnn_layers is not None:
            # 2.1 确定变量数
            # 如果提供了data_key和num_variables_dict，使用动态变量数
            # 否则使用固定的num_variables
            use_dynamic_vars = (data_key is not None and 
                              num_variables_dict is not None and 
                              len(num_variables_dict) > 0)
            
            if use_dynamic_vars:
                # 动态变量数模式：根据data_key分组处理
                mask_logit_refined = mask_logit_init.clone()
                
                # 确保data_key在正确的设备上，并且是1D tensor
                if data_key is not None:
                    if isinstance(data_key, torch.Tensor):
                        if data_key.device != x.device:
                            data_key = data_key.to(x.device)
                        # 确保data_key是1D且长度匹配
                        if data_key.dim() > 1:
                            data_key = data_key.flatten()
                        if len(data_key) != b_total:
                            # data_key长度不匹配，跳过动态变量数模式，使用固定模式
                            use_dynamic_vars = False
                    else:
                        # data_key不是tensor，跳过动态变量数模式
                        use_dynamic_vars = False
                else:
                    use_dynamic_vars = False
                
                if use_dynamic_vars:
                    # 按数据集分组处理
                    unique_data_keys = torch.unique(data_key)
                    for data_key_val in unique_data_keys:
                        # 找到属于当前数据集的所有样本索引
                        mask_indices = (data_key == data_key_val).nonzero(as_tuple=True)[0]
                        num_samples = len(mask_indices)
                        
                        if num_samples == 0:
                            continue
                        
                        # 确保索引在有效范围内
                        max_idx = mask_indices.max().item()
                        if max_idx >= b_total:
                            # 索引超出范围，跳过
                            continue
                        
                        # 获取该数据集的变量数
                        data_key_int = data_key_val.item()
                        if data_key_int not in num_variables_dict:
                            # 如果找不到，跳过GNN（保持原样）
                            continue
                        
                        num_vars = num_variables_dict[data_key_int]
                        
                        # 只有当变量数>1且样本数能被变量数整除时才应用GNN
                        if num_vars > 1 and num_samples % num_vars == 0:
                            B_local = num_samples // num_vars
                            
                            # 确保索引在有效范围内
                            if mask_indices.max().item() >= b_total or mask_indices.min().item() < 0:
                                continue
                            
                            try:
                                mask_local = mask_logit_init[mask_indices]  # (num_samples, K)
                                mask_reshaped = mask_local.view(B_local, num_vars, self.num_latents)  # (B_local, N, K)
                            except (IndexError, RuntimeError) as e:
                                # 索引错误，跳过这个数据集
                                continue
                            
                            # 对每个batch独立处理GNN
                            for b in range(B_local):
                                mask_batch = mask_reshaped[b]  # (N, K)
                                adj = self._build_adjacency_matrix(num_vars, 'fully_connected')
                                adj = adj.to(x.device)
                                
                                mask_graph = mask_batch
                                for i, gnn_layer in enumerate(self.gnn_layers):
                                    if self.gnn_type == 'simple_gcn':
                                        mask_graph = gnn_layer(mask_graph, adj)
                                    else:
                                        edge_index = self._build_edge_index(num_vars, 1, device=x.device)
                                        mask_graph = gnn_layer(mask_graph, edge_index)
                                    
                                    if i < len(self.gnn_layers) - 1:
                                        mask_graph = F.relu(mask_graph)
                                
                                # 更新对应位置的mask（使用安全的索引方式）
                                start_idx = b * num_vars
                                end_idx = (b + 1) * num_vars
                                if end_idx <= len(mask_indices):
                                    indices_to_update = mask_indices[start_idx:end_idx]
                                    # 再次检查索引范围
                                    if len(indices_to_update) > 0:
                                        max_idx = indices_to_update.max().item()
                                        min_idx = indices_to_update.min().item()
                                        if max_idx < b_total and min_idx >= 0:
                                            try:
                                                mask_logit_refined[indices_to_update] = mask_graph
                                            except (IndexError, RuntimeError):
                                                # 如果还是出错，跳过这个batch，保持原样
                                                pass
                
            else:
                # 固定变量数模式（原有逻辑）
                if self.num_variables is None:
                    # 如果没有指定变量数，禁用GNN
                    mask_logit_refined = mask_logit_init
                else:
                    B = b_total // self.num_variables
                    assert b_total % self.num_variables == 0, \
                        f"b_total ({b_total}) must be divisible by num_variables ({self.num_variables})"
                    
                    # 2.2 Reshape: (B*N, K) → (B, N, K)
                    mask_reshaped = mask_logit_init.view(B, self.num_variables, self.num_latents)
                    
                    # 2.3 对每个batch独立处理GNN
                    mask_refined_list = []
                    for b in range(B):
                        # 提取当前batch的mask: (N, K)
                        mask_batch = mask_reshaped[b]  # (N, K)
                        
                        # 构建邻接矩阵
                        adj = self._build_adjacency_matrix(self.num_variables, 'fully_connected')
                        adj = adj.to(x.device)
                        
                        # GNN前向传播
                        mask_graph = mask_batch  # (N, K)
                        for i, gnn_layer in enumerate(self.gnn_layers):
                            if self.gnn_type == 'simple_gcn':
                                # 使用简单GCN实现
                                mask_graph = gnn_layer(mask_graph, adj)
                            else:
                                # 使用PyTorch Geometric
                                # 构建边索引
                                edge_index = self._build_edge_index(self.num_variables, 1, device=x.device)
                                mask_graph = gnn_layer(mask_graph, edge_index)
                            
                            # 激活函数（最后一层除外）
                            if i < len(self.gnn_layers) - 1:
                                mask_graph = F.relu(mask_graph)
                        
                        mask_refined_list.append(mask_graph)
                    
                    # 2.4 合并所有batch: (B, N, K) → (B*N, K)
                    mask_logit_refined = torch.stack(mask_refined_list, dim=0)  # (B, N, K)
                    mask_logit_refined = mask_logit_refined.view(B * self.num_variables, self.num_latents)  # (B*N, K)
        else:
            # 不使用GNN，保持原样
            mask_logit_refined = mask_logit_init
        
        # 3. 生成输出（不变）
        latents = repeat(self.latents, 'n d -> b n d', b=b_total)
        mask = mask_logit_refined  # (B*N, K)
        out = latents  # (B*N, K, d)
        
        return out, mask
        
