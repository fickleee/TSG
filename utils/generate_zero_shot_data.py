# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
只生成零样本测试数据的简化脚本
用于在已有训练数据的情况下，生成stock和web的零样本测试数据
"""

import numpy as np
import pandas as pd
from einops import rearrange
from pathlib import Path
import os
import sys

# 添加路径以便导入工具函数
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from utils.data_utils import convert_tsf_to_dataframe

# 使用环境变量DATA_ROOT，如果不存在则使用默认路径
if 'DATA_ROOT' in os.environ and os.path.exists(os.environ['DATA_ROOT']):
    DATA_ROOT = os.environ['DATA_ROOT']
    if not DATA_ROOT.endswith('/') and not DATA_ROOT.endswith('\\'):
        DATA_ROOT = DATA_ROOT + '/'
else:
    # 默认使用dataset目录
    DATA_ROOT = './dataset/'

PREFIX = DATA_ROOT

def more_data_loading(data_name, seq_len=168, stride=1, univar=False):
    """加载数据并切分成序列"""
    data_name_path_map = {
        'stock': 'stock_data.csv',
        'web': './data/web.csv',
    }
    
    if data_name not in data_name_path_map:
        raise ValueError(f"不支持的数据集: {data_name}")
    
    data_path = PREFIX + data_name_path_map[data_name]
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        # 尝试在dataset目录下查找
        alt_path = os.path.join(PREFIX, os.path.basename(data_path))
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"数据文件不存在: {data_path}。请先运行 utils/prepare_datasets.py 准备数据。")
    
    if data_name == 'stock':
        ori_data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    else:  # web
        ori_data = pd.read_csv(data_path).values
    
    # 切分成序列
    temp_data = []
    for i in range(0, len(ori_data) - seq_len + 1, stride):
        _x = ori_data[None, i:i + seq_len]
        temp_data.append(_x)
    
    data = np.vstack(temp_data)
    if univar:
        data = rearrange(data, 'n t c -> (n c) t 1')
    
    return data

def generate_zero_shot_data():
    """生成零样本测试数据"""
    zero_shot_schedule = [3, 10, 100]
    
    print(f"使用数据根目录: {PREFIX}")
    print("开始生成零样本测试数据...")
    
    for data_name in ['web', 'stock']:
        print(f"\n处理数据集: {data_name}")
        for seq_len in [24, 96, 168, 336]:
            try:
                print(f"  序列长度: {seq_len}")
                
                if data_name == 'stock':
                    ori_data = more_data_loading(data_name, seq_len, 1, univar=False)
                    uni_ori_data = ori_data[:,:,0,None]
                    uni_ori_data /= uni_ori_data[:, :1, :]
                else:  # web
                    ori_data = more_data_loading(data_name, seq_len, seq_len, univar=False)
                    uni_ori_data = rearrange(ori_data, 'b t c -> (b c) t 1')
                
                zero_shot_data_path = Path(f'{PREFIX}/ts_data/new_zero_shot_data')
                zero_shot_data_path.mkdir(exist_ok=True, parents=True)
                
                print(f"  数据量: {len(uni_ori_data)}")
                np.random.seed(0)
                k_idx = np.random.choice(len(uni_ori_data), 2000+max(zero_shot_schedule))
                zero_shot_test_data = uni_ori_data[k_idx[-2000:]]
                
                # 保存测试数据
                test_file = zero_shot_data_path / f'{data_name}_{seq_len}_test_sample.npy'
                np.save(test_file, zero_shot_test_data)
                print(f"  已保存: {test_file}")
                
                # 保存k-shot提示数据
                for k in zero_shot_schedule:
                    zero_shot_prompt = uni_ori_data[k_idx[:k]]
                    k_file = zero_shot_data_path / f'{data_name}_{seq_len}_k_{k}_sample.npy'
                    np.save(k_file, zero_shot_prompt)
                    print(f"  已保存: {k_file}")
                    
                    # 保存CSV格式（可选）
                    csv_file = zero_shot_data_path / f'{data_name}_dim0_{seq_len}_k_{k}_sample.csv'
                    pd.DataFrame(zero_shot_prompt[:,:,0].T).to_csv(csv_file, index=False)
                    
            except FileNotFoundError as e:
                print(f"  错误: {e}")
                print(f"  跳过 {data_name} 的 seq_len={seq_len} 数据生成")
                continue
            except Exception as e:
                print(f"  处理 {data_name} seq_len={seq_len} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n零样本测试数据生成完成！")

if __name__ == '__main__':
    generate_zero_shot_data()

