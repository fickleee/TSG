# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
import math

import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import Spatial1DTransformer
from ldm.util import default
from .util import Return


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return th.ones(shape, device = device, dtype = th.bool)
    elif prob == 0:
        return th.zeros(shape, device = device, dtype = th.bool)
    else:
        return th.zeros(shape, device = device).float().uniform_(0, 1) < prob

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, mask=None, data_key=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, Spatial1DTransformer):
                x = layer(x, context, mask=mask)
            elif isinstance(layer, GNNBlock):
                # GNNBlock需要data_key来查找变量数
                x = layer(x, data_key=data_key)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        cond_emb_channels=None,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        if cond_emb_channels is not None:
            self.cond_emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    cond_emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
        else:
            self.cond_emb_layers = None
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class GNNBlock(nn.Module):
    """
    在UNet Bottleneck层使用的GNN模块，用于处理变量间的交互
    输入输出形状保持一致，可以无缝插入到middle_block中
    """
    def __init__(self, channels, gnn_type='gat', gnn_layers=2, 
                 gnn_hidden_dim=None, num_heads=4, use_checkpoint=False,
                 num_variables_dict=None, use_gate=True):
        super().__init__()
        self.channels = channels
        self.gnn_type = gnn_type
        self.use_checkpoint = use_checkpoint
        self.num_variables_dict = num_variables_dict or {}  # {data_key: num_variables}
        gnn_hidden_dim = gnn_hidden_dim or channels
        
        # 【第一剂：必做】输入归一化层，解决特征分布不匹配问题
        self.input_norm = nn.LayerNorm(channels)
        
        # 【第二剂：必做】Learnable Gate，让UNet有拒绝GNN输出的权利
        if use_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # (B, C, T) -> (B, C, 1)
                nn.Conv1d(channels, channels, 1),
                nn.Sigmoid()
            )
            # 初始化门控，让初始时倾向于不使用GNN（sigmoid(-2) ≈ 0.12）
            nn.init.zeros_(self.gate[1].weight)
            nn.init.constant_(self.gate[1].bias, -2.0)
        else:
            self.gate = None
        
        # 初始化GNN层（复用PAM中的实现逻辑）
        if gnn_type == 'simple_gcn':
            from ldm.modules.encoders.modules import SimpleGCNLayer
            self.gnn_layers = nn.ModuleList([
                SimpleGCNLayer(channels if i == 0 else gnn_hidden_dim,
                              channels if i == gnn_layers - 1 else gnn_hidden_dim)
                for i in range(gnn_layers)
            ])
        else:
            try:
                from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
                if gnn_type == 'gcn':
                    self.gnn_layers = nn.ModuleList([
                        GCNConv(channels if i == 0 else gnn_hidden_dim,
                               channels if i == gnn_layers - 1 else gnn_hidden_dim)
                        for i in range(gnn_layers)
                    ])
                elif gnn_type == 'gat':
                    self.gnn_layers = nn.ModuleList([
                        GATConv(channels if i == 0 else gnn_hidden_dim,
                               channels if i == gnn_layers - 1 else gnn_hidden_dim,
                               heads=num_heads, concat=False)
                        for i in range(gnn_layers)
                    ])
                elif gnn_type == 'gatv2':
                    self.gnn_layers = nn.ModuleList([
                        GATv2Conv(channels if i == 0 else gnn_hidden_dim,
                                 channels if i == gnn_layers - 1 else gnn_hidden_dim,
                                 heads=num_heads, concat=False)
                        for i in range(gnn_layers)
                    ])
                else:
                    raise ValueError(f"Unknown GNN type: {gnn_type}")
            except ImportError:
                print("Warning: PyTorch Geometric not available. Falling back to simple_gcn.")
                from ldm.modules.encoders.modules import SimpleGCNLayer
                self.gnn_layers = nn.ModuleList([
                    SimpleGCNLayer(channels if i == 0 else gnn_hidden_dim,
                                  channels if i == gnn_layers - 1 else gnn_hidden_dim)
                    for i in range(gnn_layers)
                ])
        
        # 【必须加】零初始化最后一层GNN层，防止破坏UNet已学到的特征
        if len(self.gnn_layers) > 0:
            self._zero_init_last_layer()
    
    def _zero_init_last_layer(self):
        """零初始化最后一层GNN层，确保训练初期GNN输出接近0"""
        last_layer = self.gnn_layers[-1]
        
        if self.gnn_type == 'simple_gcn':
            # SimpleGCNLayer有weight和bias属性
            nn.init.zeros_(last_layer.weight)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)
        else:
            # PyTorch Geometric的层
            # GCNConv使用lin属性
            if hasattr(last_layer, 'lin'):
                nn.init.zeros_(last_layer.lin.weight)
                if last_layer.lin.bias is not None:
                    nn.init.zeros_(last_layer.lin.bias)
            # GATConv和GATv2Conv可能使用lin_src和lin_dst
            elif hasattr(last_layer, 'lin_src'):
                nn.init.zeros_(last_layer.lin_src.weight)
                if last_layer.lin_src.bias is not None:
                    nn.init.zeros_(last_layer.lin_src.bias)
                if hasattr(last_layer, 'lin_dst'):
                    nn.init.zeros_(last_layer.lin_dst.weight)
                    if last_layer.lin_dst.bias is not None:
                        nn.init.zeros_(last_layer.lin_dst.bias)
            # 如果都没有，尝试直接访问weight和bias
            elif hasattr(last_layer, 'weight'):
                nn.init.zeros_(last_layer.weight)
                if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                    nn.init.zeros_(last_layer.bias)
    
    def _build_edge_index(self, num_nodes, num_samples, num_timesteps, device):
        """
        构建边索引：每个时间步独立构建全连接图
        使用向量化操作，大幅提升性能并允许中断
        Args:
            num_nodes: 变量数
            num_samples: 样本数（N，不是B）
            num_timesteps: 时间步数
        Returns:
            edge_index: (2, num_edges) 边索引
        """
        # 如果只有1个节点，无法构建边
        if num_nodes <= 1:
            return th.empty((2, 0), dtype=th.long, device=device)
        
        # 为单个图构建边索引（全连接，移除自连接）
        # 使用meshgrid创建所有节点对
        i, j = th.meshgrid(
            th.arange(num_nodes, device=device),
            th.arange(num_nodes, device=device),
            indexing='ij'
        )
        # 移除自连接
        mask = i != j
        i, j = i[mask], j[mask]  # (V*(V-1),)
        
        # 单个图的边索引
        single_graph_edges = th.stack([i, j], dim=0)  # (2, V*(V-1))
        num_edges_per_graph = single_graph_edges.shape[1]
        
        # 为所有样本和时间步复制边（向量化）
        total_graphs = num_samples * num_timesteps
        total_edges = num_edges_per_graph * total_graphs
        
        # 创建节点偏移量
        # 对于(n, t)，节点偏移 = (n * num_timesteps + t) * num_nodes
        graph_indices = th.arange(total_graphs, device=device)  # (total_graphs,)
        node_offsets = graph_indices * num_nodes  # (total_graphs,)
        
        # 扩展边索引（完全向量化，快速且可中断）
        # single_graph_edges: (2, V*(V-1))
        # node_offsets: (total_graphs,)
        # 结果: (2, total_graphs * V*(V-1))
        edge_src = (single_graph_edges[0].unsqueeze(0) + node_offsets.unsqueeze(1)).flatten()  # (total_edges,)
        edge_dst = (single_graph_edges[1].unsqueeze(0) + node_offsets.unsqueeze(1)).flatten()  # (total_edges,)
        
        edge_index = th.stack([edge_src, edge_dst], dim=0)  # (2, total_edges)
        
        return edge_index
    
    def _build_adjacency_matrix(self, num_nodes, device):
        """为SimpleGCNLayer构建邻接矩阵"""
        # 全连接图（移除自连接）
        adj = th.ones(num_nodes, num_nodes, device=device)
        adj = adj - th.eye(num_nodes, device=device)
        return adj
    
    def forward(self, x, data_key=None):
        """
        Args:
            x: (B, C, T) - B是batch（可能包含变量信息），C是特征通道数，T是时间步
            data_key: (B,) - 每个样本的数据集标识，用于查找对应的变量数
        Returns:
            x: (B, C, T) - 经过GNN处理后的特征
        """
        B, C, T = x.shape
        
        # 如果没有提供data_key或num_variables_dict，无法确定变量数，直接返回
        if data_key is None or len(self.num_variables_dict) == 0:
            return x
        
        # 根据data_key获取每个样本的变量数
        # data_key是数据集索引（整数），对应num_variables_dict的键
        # 注意：B可能包含变量信息，即 B = N * V（N是样本数，V是变量数）
        # 我们需要根据data_key来确定每个样本的变量数
        
        # 获取第一个样本的data_key，假设batch中所有样本来自同一数据集
        # 如果batch混合了不同数据集，需要更复杂的处理
        if isinstance(data_key, torch.Tensor):
            first_data_key = int(data_key[0].item())
        else:
            first_data_key = int(data_key[0]) if hasattr(data_key, '__getitem__') else 0
        
        # 从num_variables_dict中查找变量数（data_key是索引）
        num_variables = self.num_variables_dict.get(first_data_key, 1)
        
        # 如果变量数为1，直接返回
        if num_variables <= 1:
            return x
        
        # 检查batch_size是否能被变量数整除
        if B % num_variables != 0:
            # 如果无法整除，说明batch中混合了不同数据集的样本，暂时跳过GNN
            return x
        
        # 重新组织数据：将(B, C, T) -> (N, V, C, T)，其中N = B // V
        N = B // num_variables
        x_reshaped = x.view(N, num_variables, C, T)  # (N, V, C, T)
        
        # 对每个样本独立处理：将每个时间步的V个变量作为图的节点
        # (N, V, C, T) -> (N*T, V, C) -> (N*T*V, C)
        x_reshaped = x_reshaped.permute(0, 3, 1, 2).contiguous()  # (N, T, V, C)
        x_reshaped = x_reshaped.view(N * T, num_variables, C)  # (N*T, V, C)
        x_reshaped = x_reshaped.view(N * T * num_variables, C)  # (N*T*V, C)
        
        # 【第一剂：必做】输入归一化，解决特征分布不匹配问题
        x_reshaped_norm = self.input_norm(x_reshaped)  # (N*T*V, C)
        
        # 应用GNN层：对每个时间步的变量进行交互
        h = x_reshaped_norm
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'simple_gcn':
                # SimpleGCNLayer需要邻接矩阵
                adj = self._build_adjacency_matrix(num_variables, x.device)
                # 对每个时间步独立处理
                h_reshaped = h.view(N * T, num_variables, C)
                h_list = []
                for t_idx in range(N * T):
                    h_t = h_reshaped[t_idx]  # (V, C)
                    h_t = gnn_layer(h_t, adj=adj)  # (V, C)
                    h_list.append(h_t)
                h = th.stack(h_list, dim=0).view(N * T * num_variables, C)
            else:
                # PyTorch Geometric的层使用edge_index
                # 为每个时间步构建独立的图
                edge_index = self._build_edge_index(num_variables, N, T, x.device)
                if edge_index.shape[1] == 0:
                    # 如果无法构建边，直接返回原始输入（跳过GNN）
                    h = x_reshaped_norm
                    break
                h = gnn_layer(h, edge_index)
            
            # 在中间层之间添加LayerNorm和激活函数
            if i < len(self.gnn_layers) - 1:
                # 对每个时间步的变量特征进行LayerNorm
                h_reshaped = h.view(N * T, num_variables, C)
                h_reshaped = F.layer_norm(h_reshaped, (C,))
                h = h_reshaped.view(N * T * num_variables, C)
                h = F.relu(h)
        
        # Reshape回: (N*T*V, C) -> (N, V, C, T) -> (B, C, T)
        h = h.view(N * T, num_variables, C)
        h = h.view(N, T, num_variables, C)
        h = h.permute(0, 2, 3, 1).contiguous()  # (N, V, C, T)
        h = h.view(B, C, T)  # (B, C, T)
        
        # 【第二剂：必做】Learnable Gate，让UNet有拒绝GNN输出的权利
        # 由于最后一层已零初始化，训练初期h≈0，gate也会倾向于0，所以x+gate*h≈x
        if self.gate is not None:
            gate_weight = self.gate(x)  # (B, C, 1)
            return x + gate_weight * h
        else:
            # 如果没有gate，使用简单的残差连接
            return x + h


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        seq_len,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        legacy=True,
        repre_emb_channels=32,
        latent_unit=6,
        use_cfg=True,
        cond_drop_prob=0.5,
        use_pam=False,
        # 新增GNN相关参数
        use_gnn_in_unet=False,  # 是否在UNet中使用GNN
        unet_gnn_type='gat',     # GNN类型
        unet_gnn_layers=2,       # GNN层数
        unet_gnn_hidden_dim=None, # GNN隐藏层维度
        unet_gnn_heads=4,        # GAT的注意力头数
        num_variables_dict=None,  # 变量数字典 {data_key: num_variables}
    ):
        super().__init__()
        # if use_spatial_transformer:
        #     assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_cfg = use_cfg
        self.cond_drop_prob = cond_drop_prob
        self.latent_unit = latent_unit
        self.latent_dim = repre_emb_channels
        self.use_pam = use_pam
        # GNN相关参数
        self.use_gnn_in_unet = use_gnn_in_unet
        self.unet_gnn_type = unet_gnn_type
        self.unet_gnn_layers = unet_gnn_layers
        self.unet_gnn_hidden_dim = unet_gnn_hidden_dim
        self.unet_gnn_heads = unet_gnn_heads
        self.num_variables_dict = num_variables_dict or {}
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        if self.use_cfg:
            self.cond_emb_channels = repre_emb_channels * latent_unit if self.use_cfg else None
            self.null_classes_emb = nn.Parameter(th.randn(1, repre_emb_channels))
        else:
            self.cond_emb_channels = None
                    
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        cond_emb_channels=self.cond_emb_channels,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else Spatial1DTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, use_pam=self.use_pam
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            cond_emb_channels=self.cond_emb_channels,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        
        # 构建middle_block层列表
        middle_block_layers = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                cond_emb_channels=self.cond_emb_channels,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else Spatial1DTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, use_pam=self.use_pam
                        ),
        ]
        
        # 在Bottleneck层添加GNN（在Attention之后，第二个ResBlock之前）
        if self.use_gnn_in_unet:
            middle_block_layers.append(
                GNNBlock(
                    ch,
                    gnn_type=self.unet_gnn_type,
                    gnn_layers=self.unet_gnn_layers,
                    gnn_hidden_dim=self.unet_gnn_hidden_dim,
                    num_heads=self.unet_gnn_heads,
                    use_checkpoint=use_checkpoint,
                    num_variables_dict=self.num_variables_dict
                )
            )
            print(f"GNN added to UNet Bottleneck: type={self.unet_gnn_type}, layers={self.unet_gnn_layers}")
        
        middle_block_layers.append(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                cond_emb_channels=self.cond_emb_channels,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        
        self.middle_block = TimestepEmbedSequential(*middle_block_layers)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        cond_emb_channels=self.cond_emb_channels,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else Spatial1DTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim, use_pam=self.use_pam
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            cond_emb_channels=self.cond_emb_channels,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def _forward(self, x, timesteps=None, context=None, mask=None, y=None, cond_drop_prob=0, data_key=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn/adaln
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # context = None
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        bs, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        if context is not None:
            c_num = context.shape[1]
            
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((bs, c_num, 1), 1 - cond_drop_prob, device = device)
                null_classes_emb = repeat(self.null_classes_emb, '1 d -> b n d', b = bs, n = c_num)
                context_emb = context * keep_mask + (~keep_mask) * null_classes_emb
                
            else:
                context_emb = context
        else:
            context_emb = None
        
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        k = 0
        for module in self.input_blocks:
            h = module(h, emb, context_emb, mask=mask, data_key=data_key)
            hs.append(h)
            if k == 5:
                a = 1
            k += 1
        h = self.middle_block(h, emb, context_emb, mask=mask, data_key=data_key)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context_emb, mask=mask, data_key=data_key)
        h = h.type(x.dtype)
        pred = self.out(h)
        return Return(pred = pred)

    def forward(self, x, timesteps=None, context=None, mask=None, y=None, cond_drop_prob=0, **kwargs):
        out = self._forward(x, timesteps, context, mask, y, cond_drop_prob, **kwargs)
        return out
    
    def forward_with_cfg(self, x, timesteps=None, context=None, y=None, cfg_scale=1,**kwargs):
        model_out = self._forward(x=x, timesteps=timesteps, context=context, y=y, cond_drop_prob=0.,**kwargs)
        if cfg_scale == 1:
            return model_out
        
        null_context_out = self._forward(x=x, timesteps=timesteps, context=context, y=y, cond_drop_prob=1.,**kwargs)
        cfg_grad = model_out.pred - null_context_out.pred
        scaled_out = null_context_out.pred + cfg_scale * cfg_grad
       
        return Return(pred=scaled_out)
