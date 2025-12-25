# 运行脚本示例

本文档提供了使用UNet Bottleneck GNN功能的运行脚本示例。

## 基本训练命令

### 1. 仅使用PAM（不使用GNN）

```bash
python main_train.py \
  -b configs/multi_domain_timedp.yaml \
  -up \
  -sl 168 \
  -nl 16 \
  -bs 128 \
  -lr 1e-4 \
  -s 0
```

### 2. 使用PAM + PAM中的GNN

```bash
python main_train.py \
  -b configs/multi_domain_timedp.yaml \
  -up \
  --use_gnn \
  --gnn_type gat \
  --gnn_layers 2 \
  -sl 168 \
  -nl 16 \
  -bs 128 \
  -lr 1e-4 \
  -s 0
```

### 3. 使用PAM + UNet Bottleneck中的GNN

```bash
python main_train.py \
  -b configs/multi_domain_timedp.yaml \
  -up \
  --use_gnn_in_unet \
  --unet_gnn_type gat \
  --unet_gnn_layers 2 \
  --unet_gnn_heads 4 \
  -sl 168 \
  -nl 16 \
  -bs 128 \
  -lr 1e-4 \
  -s 0
```

### 4. 同时使用PAM GNN和UNet GNN（推荐）

```bash
python main_train.py \
  -b configs/multi_domain_timedp.yaml \
  -up \
  --use_gnn \
  --gnn_type gat \
  --gnn_layers 2 \
  --use_gnn_in_unet \
  --unet_gnn_type gat \
  --unet_gnn_layers 2 \
  --unet_gnn_heads 4 \
  -sl 168 \
  -nl 16 \
  -bs 128 \
  -lr 1e-4 \
  -s 0
```

### 5. 使用GATv2（更强大的注意力机制）

```bash
python main_train.py \
  -b configs/multi_domain_timedp.yaml \
  -up \
  --use_gnn \
  --gnn_type gatv2 \
  --gnn_layers 2 \
  --use_gnn_in_unet \
  --unet_gnn_type gatv2 \
  --unet_gnn_layers 2 \
  --unet_gnn_heads 4 \
  -sl 168 \
  -nl 16 \
  -bs 128 \
  -lr 1e-4 \
  -s 0
```

## 参数说明

### PAM GNN参数
- `--use_gnn`: 启用PAM中的GNN（用于domain prompt的变量交互）
- `--gnn_type`: GNN类型，可选 `simple_gcn`, `gcn`, `gat`, `gatv2`
- `--gnn_layers`: GNN层数（默认2）
- `--num_variables`: 变量数（如果为None则从数据推断）

### UNet Bottleneck GNN参数
- `--use_gnn_in_unet`: 启用UNet Bottleneck中的GNN（用于去噪过程的变量交互）
- `--unet_gnn_type`: UNet GNN类型，可选 `simple_gcn`, `gcn`, `gat`, `gatv2`（默认`gat`）
- `--unet_gnn_layers`: UNet GNN层数（默认2）
- `--unet_gnn_hidden_dim`: UNet GNN隐藏层维度（默认None，使用channels）
- `--unet_gnn_heads`: GAT的注意力头数（默认4）

### 其他常用参数
- `-b, --base`: 配置文件路径
- `-up, --use_pam`: 使用PAM（Prototype Attention Mechanism）
- `-sl, --seq_len`: 序列长度（默认24）
- `-nl, --num_latents`: 原型数量（默认16）
- `-bs, --batch_size`: 批次大小（默认128）
- `-lr, --overwrite_learning_rate`: 学习率
- `-s, --seed`: 随机种子（默认23）
- `-n, --name`: 实验名称（可选）
- `--no-test`: 禁用测试

## 实验命名规则

实验目录名称会自动包含以下信息：
- 序列长度（`_168`）
- 原型数量（`_nl_16`）
- 学习率（`_lr1.0e-04`）
- 批次大小（`_bs128`）
- PAM GNN类型（`_gnngat`，如果启用）
- UNet GNN类型（`_unetgnngat`，如果启用）
- PAM标记（`_pam`，如果启用）
- 随机种子（`_seed0`）

例如：`multi_domain_timedp_168_nl_16_lr1.0e-04_bs128_gnngat_unetgnngat_pam_seed0`

## 注意事项

1. **PyTorch Geometric依赖**：使用`gcn`、`gat`或`gatv2`需要安装PyTorch Geometric：
   ```bash
   pip install torch-geometric
   ```

2. **计算资源**：同时启用PAM GNN和UNet GNN会增加计算量，建议：
   - 使用GPU训练
   - 适当减小batch size
   - 监控显存使用

3. **变量数设置**：对于多变量数据集，建议通过`--num_variables`明确指定变量数，以确保GNN正常工作。

4. **实验对比**：建议分别运行以下配置进行对比：
   - 基线：仅PAM，无GNN
   - PAM GNN：仅启用PAM中的GNN
   - UNet GNN：仅启用UNet中的GNN
   - 两者都启用：PAM GNN + UNet GNN

## 查看结果

训练完成后，可以使用以下命令查看指标：

```bash
python utils/show_metrics.py
```

或者查看特定实验的指标：

```bash
python utils/format_metrics.py logs/multi_domain_timedp/实验目录名/metric_dict.pkl
```

