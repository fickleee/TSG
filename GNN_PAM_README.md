# GNN增强PAM模块使用说明

## 概述

本次更新在 `DomainUnifiedPrototyper` (PAM) 模块中引入了图神经网络（GNN）支持，实现了多变量之间的交互。核心思路是"先提取，再交互"：

1. **提取阶段**：使用Conv1D生成初始的PAM权重 `(B*N, K)`
2. **交互阶段**：Reshape为 `(B, N, K)`，使用GNN让N个变量之间交互
3. **输出阶段**：Reshape回 `(B*N, K)`，保持与原有接口兼容

## 架构变更

### 原版流程
```
Input (B*N, T) → Conv1D (PAM) → Weights (B*N, K) → Diffusion
```

### GNN增强版流程
```
Input (B*N, T) → Conv1D (PAM) → Weights_Init (B*N, K) 
    → Reshape (B, N, K) → GNN/GraphConv → Weights_Refined (B, N, K)
    → Reshape (B*N, K) → Diffusion
```

## 使用方法

### 1. 命令行参数

新增了以下命令行参数：

```bash
--use_gnn              # 启用GNN多变量交互（默认：False）
--num_variables N      # 变量数（启用GNN时必须指定）
--gnn_layers 2         # GNN层数（默认：2）
--gnn_type TYPE        # GNN类型：simple_gcn / gcn / gat（默认：simple_gcn）
```

### 2. 配置文件

在 `configs/multi_domain_timedp.yaml` 中已添加GNN相关参数：

```yaml
cond_stage_config:
  target: ldm.modules.encoders.modules.DomainUnifiedPrototyper  
  params:
    # ... 其他参数 ...
    use_gnn: false      # 是否启用GNN
    num_variables: null # 变量数
    gnn_layers: 2       # GNN层数
    gnn_hidden_dim: null # GNN隐藏层维度
    gnn_type: simple_gcn # GNN类型
```

### 3. 使用示例

#### 示例1：使用简单GCN（无需额外依赖）

```bash
python main_train.py \
  --base configs/multi_domain_timedp.yaml \
  --gpus 0, \
  --logdir ./logs/ \
  -sl 168 \
  -up \
  -nl 16 \
  --batch_size 128 \
  -lr 0.0001 \
  -s 0 \
  --use_gnn \
  --num_variables 5 \
  --gnn_layers 2 \
  --gnn_type simple_gcn
```

#### 示例2：使用PyTorch Geometric的GCN（需要安装torch-geometric）

```bash
# 首先安装依赖
pip install torch-geometric

# 然后运行
python main_train.py \
  --base configs/multi_domain_timedp.yaml \
  --gpus 0, \
  --logdir ./logs/ \
  -sl 168 \
  -up \
  -nl 16 \
  --batch_size 128 \
  -lr 0.0001 \
  -s 0 \
  --use_gnn \
  --num_variables 5 \
  --gnn_layers 2 \
  --gnn_type gcn
```

#### 示例3：使用GAT（Graph Attention Network）

```bash
python main_train.py \
  --base configs/multi_domain_timedp.yaml \
  --gpus 0, \
  --logdir ./logs/ \
  -sl 168 \
  -up \
  -nl 16 \
  --batch_size 128 \
  -lr 0.0001 \
  -s 0 \
  --use_gnn \
  --num_variables 5 \
  --gnn_layers 2 \
  --gnn_type gat
```

## GNN类型说明

### 1. `simple_gcn`（推荐，默认）

- **优点**：无需额外依赖，纯PyTorch实现
- **实现**：简单的图卷积层，使用对称归一化的邻接矩阵
- **适用场景**：快速实验，无PyTorch Geometric环境

### 2. `gcn`

- **优点**：使用PyTorch Geometric的优化实现
- **依赖**：需要安装 `torch-geometric`
- **适用场景**：需要更高效的GCN实现

### 3. `gat`

- **优点**：使用注意力机制，可以学习变量间的重要性
- **依赖**：需要安装 `torch-geometric`
- **适用场景**：变量间关系复杂，需要自适应权重

## 重要注意事项

### 1. 变量数的确定（已支持动态变量数）

**✅ 已解决**：系统现在支持**动态变量数**，可以自动处理不同数据集的不同变量数。

**方案A：自动推断（推荐）**
- 系统会自动从数据中推断每个数据集的变量数
- 在数据加载时记录原始通道数
- 根据 `data_key` 动态选择变量数
- **无需手动配置**

**方案B：手动指定（精确控制）**
- 在配置文件中为每个数据集指定变量数：
  ```yaml
  data:
    params:
      num_variables_dict:
        solar: 1
        traffic: 3
        kddcup: 5
  ```

**方案C：固定变量数（所有数据集相同）**
- 在命令行指定 `--num_variables N`
  - 适用于所有数据集变量数相同的情况
  - 确保 `batch_size` 能被 `num_variables` 整除

### 2. Batch大小要求

**动态变量数模式**：
- 系统会自动处理，如果某个数据集的样本数不能被变量数整除，会跳过GNN（保持原样）
- 不会报错，只是该数据集不使用GNN

**固定变量数模式**：
- 确保 `batch_size % num_variables == 0`，否则会报错
- 例如：
  - `batch_size=128`, `num_variables=5` ❌ (128 % 5 != 0)
  - `batch_size=125`, `num_variables=5` ✅ (125 % 5 == 0)
  - `batch_size=128`, `num_variables=4` ✅ (128 % 4 == 0)

### 3. 向后兼容性

- 如果不指定 `--use_gnn`，行为与原来完全一致
- 所有输出形状保持不变：`(B*N, K)` 和 `(B*N, K, d)`
- 与UNet的接口完全兼容

## 实现细节

### 1. SimpleGCNLayer

实现了简单的图卷积层，不依赖外部库：

```python
class SimpleGCNLayer(nn.Module):
    def forward(self, x, adj=None):
        # x: (N, K) - 节点特征
        # adj: (N, N) - 邻接矩阵（全连接图）
        # 返回: (N, K) - 更新后的节点特征
```

### 2. 图结构

当前实现使用**全连接图**：
- 所有变量节点互相连接
- 移除自连接（可选）
- 使用对称归一化：`D^(-1/2) * A * D^(-1/2)`

未来可以扩展：
- 基于先验知识的图结构（如相关性矩阵）
- 可学习的图结构（Graph Attention）

### 3. 前向传播流程

```python
# 1. 初始PAM权重
mask_logit_init = self.mask_ffn(h)  # (B*N, K)

# 2. Reshape为图格式
mask_reshaped = mask_logit_init.view(B, N, K)  # (B, N, K)

# 3. 对每个batch独立处理GNN
for b in range(B):
    mask_batch = mask_reshaped[b]  # (N, K)
    # GNN处理
    for gnn_layer in self.gnn_layers:
        mask_batch = gnn_layer(mask_batch, adj)
    mask_refined_list.append(mask_batch)

# 4. Reshape回原格式
mask_logit_refined = torch.stack(mask_refined_list).view(B*N, K)
```

## 代码修改清单

1. **ldm/modules/encoders/modules.py**
   - 添加 `SimpleGCNLayer` 类
   - 修改 `DomainUnifiedPrototyper` 类，添加GNN支持

2. **utils/cli_utils.py**
   - 添加 `--use_gnn`, `--num_variables`, `--gnn_layers`, `--gnn_type` 参数

3. **utils/init_utils.py**
   - 在 `init_model_data_trainer` 和 `load_model_data` 中添加GNN参数传递

4. **configs/multi_domain_timedp.yaml**
   - 添加GNN相关配置参数

## 测试建议

1. **单变量场景**（当前默认）
   ```bash
   # 不启用GNN，应该与原来行为一致
   python main_train.py --base configs/multi_domain_timedp.yaml -up -sl 168
   ```

2. **多变量场景（模拟）**
   ```bash
   # 假设有5个变量
   python main_train.py --base configs/multi_domain_timedp.yaml \
     -up -sl 168 --use_gnn --num_variables 5 --gnn_type simple_gcn
   ```

3. **对比实验**
   ```bash
   # 原版PAM
   python main_train.py ... -up
   
   # GNN增强PAM
   python main_train.py ... -up --use_gnn --num_variables N
   ```

## 故障排除

### 问题1：`b_total must be divisible by num_variables`

**原因**：batch大小不能被变量数整除

**解决**：调整 `--batch_size` 或 `--num_variables`

### 问题2：`PyTorch Geometric not available`

**原因**：使用了 `gcn` 或 `gat` 但未安装torch-geometric

**解决**：
- 安装：`pip install torch-geometric`
- 或使用 `--gnn_type simple_gcn`

### 问题3：GNN没有效果

**原因**：可能 `num_variables=1` 或未正确启用

**解决**：
- 检查 `--use_gnn` 是否设置
- 检查 `--num_variables` 是否 > 1
- 查看日志中的 "GNN enabled" 消息

## 未来扩展

1. **可学习的图结构**：使用Graph Attention Network学习变量间关系
2. **基于先验的图**：从数据相关性构建图结构
3. **分层GNN**：不同层次的变量交互
4. **动态图**：根据输入动态调整图结构

## 参考文献

- Graph Convolutional Networks (GCN): Kipf & Welling, 2017
- Graph Attention Networks (GAT): Veličković et al., 2018
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

