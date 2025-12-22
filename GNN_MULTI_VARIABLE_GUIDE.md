# 多数据集不同变量数的GNN支持指南

## 问题背景

不同数据集可能有不同的变量数：
- 单变量数据集：solar, electricity 等（变量数=1）
- 多变量数据集：某些数据集可能有3、5、10个变量

当前实现已经支持**动态变量数**，可以根据每个数据集自动调整。

## 解决方案

### 方案1：自动推断（推荐）

系统会自动从数据中推断变量数：

1. **数据加载时**：记录每个数据集的原始通道数
2. **GNN处理时**：根据`data_key`查找对应的变量数
3. **自动处理**：如果变量数=1，自动禁用GNN；如果>1，应用GNN

**优点**：
- 无需手动配置
- 自动适配不同数据集
- 向后兼容

**使用示例**：
```bash
python main_train.py \
  --base configs/multi_domain_timedp.yaml \
  --gpus 0, \
  -up \
  --use_gnn \
  --gnn_type simple_gcn
```

### 方案2：手动指定（精确控制）

在配置文件中为每个数据集指定变量数：

```yaml
data:
  params:
    num_variables_dict:
      solar: 1          # 单变量，GNN会被禁用
      electricity: 1    # 单变量
      traffic: 3        # 3个变量，会应用GNN
      kddcup: 5         # 5个变量，会应用GNN
      # ... 其他数据集
```

**使用场景**：
- 需要覆盖自动推断的值
- 数据预处理后变量数与原始不同
- 想要精确控制每个数据集的GNN行为

## 实现细节

### 1. 数据加载阶段

```python
# ldm/data/tsg_dataset.py
def prepare_data(self):
    for data_name, data_path in self.data_path_dict.items():
        this_data = load_data_from_file(data_path)
        
        # 记录原始通道数
        if this_data.ndim == 3:  # (N, T, C)
            original_channels = this_data.shape[2]
        else:
            original_channels = 1
        
        # 存储变量数
        if data_name not in self.num_variables_dict:
            self.num_variables_dict[data_name] = original_channels
```

### 2. GNN处理阶段

```python
# ldm/modules/encoders/modules.py
def forward(self, x, data_key=None, num_variables_dict=None):
    # 如果提供了data_key和num_variables_dict，使用动态变量数
    if use_dynamic_vars:
        # 按数据集分组处理
        for data_key_val in unique_data_keys:
            # 获取该数据集的变量数
            num_vars = num_variables_dict[data_key_int]
            
            # 只有当变量数>1时才应用GNN
            if num_vars > 1:
                # 应用GNN...
```

### 3. 数据流传递

```
Batch (包含data_key)
  ↓
shared_step() → forward() → get_learned_conditioning()
  ↓
DomainUnifiedPrototyper.forward(x, data_key, num_variables_dict)
  ↓
根据data_key查找变量数 → 应用GNN
```

## 使用示例

### 示例1：自动推断（最简单）

```bash
# 系统会自动从数据推断变量数
python main_train.py \
  --base configs/multi_domain_timedp.yaml \
  -up \
  --use_gnn \
  --gnn_type simple_gcn
```

### 示例2：配置文件指定

在 `configs/multi_domain_timedp.yaml` 中：

```yaml
data:
  params:
    num_variables_dict:
      solar: 1
      electricity: 1
      traffic: 3      # 假设traffic有3个变量
      kddcup: 5       # 假设kddcup有5个变量
```

然后运行：
```bash
python main_train.py \
  --base configs/multi_domain_timedp.yaml \
  -up \
  --use_gnn \
  --gnn_type simple_gcn
```

### 示例3：混合场景

- 单变量数据集：自动禁用GNN
- 多变量数据集：自动应用GNN

系统会自动处理，无需额外配置。

## 注意事项

### 1. Batch大小要求

对于多变量数据集，batch中的样本数必须能被变量数整除。

例如：
- `traffic`数据集有3个变量
- batch中traffic的样本数必须是3的倍数

**解决方案**：
- 使用`drop_last=True`（已默认启用）
- 或调整batch_size使其能被所有变量数整除

### 2. 变量数推断

系统会从数据的原始形状推断变量数：
- `(N, T, C)` → 变量数 = C
- `(N, T)` → 变量数 = 1

如果数据已经被转换为单变量格式，需要手动指定。

### 3. 性能考虑

- 动态变量数模式会按数据集分组处理，可能略慢
- 如果所有数据集变量数相同，使用固定模式更快

## 调试技巧

### 查看变量数信息

训练时会输出：
```
Loaded data: solar; Train shape: ..., Validation shape: ...
Original channels: 1, Variables for GNN: 1
```

### 检查GNN是否启用

查看日志中的：
```
GNN enabled: type=simple_gcn, layers=2, num_variables=3
```

或：
```
Warning: GNN requested but disabled (num_variables=1)
```

### 验证data_key传递

可以在`DomainUnifiedPrototyper.forward()`中添加调试输出：
```python
if data_key is not None:
    print(f"data_key: {data_key.unique()}")
    print(f"num_variables_dict: {num_variables_dict}")
```

## 常见问题

### Q1: 为什么GNN没有生效？

**A**: 检查：
1. `--use_gnn` 是否设置
2. 数据集的变量数是否>1
3. 查看日志中的"GNN enabled"消息

### Q2: 如何知道每个数据集的变量数？

**A**: 
1. 查看训练日志中的"Original channels"信息
2. 检查数据文件的形状
3. 在配置文件中手动指定

### Q3: 可以混合单变量和多变量数据集吗？

**A**: 可以！系统会自动处理：
- 单变量数据集：跳过GNN
- 多变量数据集：应用GNN

### Q4: 性能影响？

**A**: 
- 动态模式：按数据集分组，略慢但更灵活
- 固定模式：所有数据集相同变量数，更快

## 总结

当前实现已经支持：
- ✅ 自动推断变量数
- ✅ 手动指定变量数
- ✅ 混合单变量和多变量数据集
- ✅ 向后兼容（不启用GNN时行为不变）

只需启用`--use_gnn`，系统会自动处理不同数据集的变量数差异！

