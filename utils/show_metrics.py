# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
快速查看评估指标表格的便捷脚本
用法: python utils/show_metrics.py [logdir]
如果不提供logdir，会自动查找最新的metric_dict.pkl文件
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import glob

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from utils.pkl_utils import load_pkl

def format_metrics_to_table(metrics_dict):
    """将指标字典转换为表格格式"""
    rows = []
    
    for key, metrics in metrics_dict.items():
        model_name, data_name, seq_len, repeat_id = key
        
        # 提取指标值
        mdd = metrics.get('mdd', np.nan)
        if isinstance(mdd, np.ndarray):
            mdd = float(mdd.item() if mdd.size == 1 else mdd[0])
        else:
            mdd = float(mdd) if mdd is not None else np.nan
        
        flat_kl = metrics.get('flat_kl', np.nan)
        flat_kl = float(flat_kl) if flat_kl is not None else np.nan
        
        mmd_rbf = metrics.get('mmd_rbf', np.nan)
        if isinstance(mmd_rbf, np.ndarray):
            mmd_rbf = float(mmd_rbf.item() if mmd_rbf.size == 1 else mmd_rbf[0])
        else:
            mmd_rbf = float(mmd_rbf) if mmd_rbf is not None else np.nan
        
        rows.append({
            'Dataset': data_name,
            'Seq_Len': seq_len,
            'MDD': mdd,
            'Flat_KL': flat_kl,
            'MMD_RBF': mmd_rbf
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['Dataset', 'Seq_Len'])
    return df

def print_table(df, show_stats=True):
    """在终端打印格式化的表格"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', lambda x: f'{x:.6f}' if not pd.isna(x) else 'N/A')

    print("\n" + "="*90)
    print("评估指标结果表格")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)

    if show_stats:
        # 打印统计信息
        print(f"\n数据集数量: {df['Dataset'].nunique()}")
        print(f"各指标平均值:")
        print(f"  MDD:      {df['MDD'].mean():.6f}")
        print(f"  Flat_KL:  {df['Flat_KL'].mean():.6f}")
        print(f"  MMD_RBF:  {df['MMD_RBF'].mean():.6f}")
        print("="*90 + "\n")

def find_all_metric_files():
    """查找所有metric_dict.pkl文件"""
    log_dirs = [
        Path('./logs'),
        Path('../logs'),
    ]

    metric_files = []
    for log_dir in log_dirs:
        if log_dir.exists():
            metric_files.extend(log_dir.rglob('metric_dict.pkl'))

    return metric_files

def process_metric_file(metric_path):
    """处理单个metric文件"""
    csv_path = metric_path.parent / f"{metric_path.stem}_table.csv"

    # 检查是否已经存在CSV文件
    if csv_path.exists():
        print(f"跳过: {metric_path.name} (CSV已存在: {csv_path.name})")
        return

    try:
        print(f"处理: {metric_path}")
        metrics_dict = load_pkl(metric_path)
        df = format_metrics_to_table(metrics_dict)

        # 保存为CSV
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 已保存: {csv_path}")

        # 显示表格（批量处理时不显示统计信息）
        print_table(df, show_stats=False)

    except Exception as e:
        print(f"✗ 处理失败 {metric_path}: {e}")

def main():
    if len(sys.argv) > 1:
        # 如果提供了特定路径，只处理该文件
        metric_path = Path(sys.argv[1])
        if not metric_path.exists():
            print(f"错误: 文件不存在: {metric_path}")
            sys.exit(1)
        process_metric_file(metric_path)
    else:
        # 遍历所有metric文件
        metric_files = find_all_metric_files()
        if not metric_files:
            print("错误: 未找到任何metric_dict.pkl文件")
            sys.exit(1)

        print(f"找到 {len(metric_files)} 个metric_dict.pkl文件")
        print()

        processed_count = 0
        skipped_count = 0

        for metric_path in metric_files:
            csv_path = metric_path.parent / f"{metric_path.stem}_table.csv"
            if csv_path.exists():
                print(f"跳过: {metric_path.name} (CSV已存在)")
                skipped_count += 1
            else:
                process_metric_file(metric_path)
                processed_count += 1
                print()  # 空行分隔

        print("="*50)
        print(f"处理完成:")
        print(f"  新处理: {processed_count} 个")
        print(f"  跳过: {skipped_count} 个")
        print(f"  总计: {len(metric_files)} 个")

if __name__ == '__main__':
    main()

