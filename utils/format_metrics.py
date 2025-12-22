# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
将评估指标结果转换为表格格式
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import os

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from utils.pkl_utils import load_pkl

def format_metrics_to_table(metrics_dict, output_format='table'):
    """
    将指标字典转换为表格格式
    
    Args:
        metrics_dict: 指标字典，格式为 {(model_name, data_name, seq_len, repeat_id): {mdd: ..., flat_kl: ..., mmd_rbf: ...}}
        output_format: 输出格式，'table'（终端显示）、'csv'（保存CSV）、'excel'（保存Excel）或'all'（全部）
    
    Returns:
        pandas DataFrame
    """
    rows = []
    
    for key, metrics in metrics_dict.items():
        model_name, data_name, seq_len, repeat_id = key
        
        # 提取指标值（如果是numpy array，取第一个元素）
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
            'Model': model_name,
            'Dataset': data_name,
            'Seq_Len': seq_len,
            'Repeat_ID': repeat_id,
            'MDD': mdd,
            'Flat_KL': flat_kl,
            'MMD_RBF': mmd_rbf
        })
    
    df = pd.DataFrame(rows)
    
    # 按数据集和序列长度排序
    df = df.sort_values(['Dataset', 'Seq_Len', 'Model'])
    
    return df

def print_table(df):
    """在终端打印格式化的表格"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.float_format', lambda x: f'{x:.6f}' if not pd.isna(x) else 'N/A')
    
    print("\n" + "="*100)
    print("评估指标结果表格")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    # 打印统计信息
    print("\n统计摘要:")
    print("-"*100)
    print(f"总数据集数: {df['Dataset'].nunique()}")
    print(f"数据集列表: {', '.join(sorted(df['Dataset'].unique()))}")
    print(f"序列长度: {', '.join(map(str, sorted(df['Seq_Len'].unique())))}")
    print("\n各指标的平均值:")
    print(df[['MDD', 'Flat_KL', 'MMD_RBF']].mean().to_string())
    print("-"*100 + "\n")

def main():
    parser = argparse.ArgumentParser(description='将评估指标转换为表格格式')
    parser.add_argument('--metric_file', type=str, required=True,
                       help='指标文件路径 (metric_dict.pkl)')
    parser.add_argument('--output', type=str, default='table',
                       choices=['table', 'csv', 'excel', 'all'],
                       help='输出格式: table(终端显示), csv(保存CSV), excel(保存Excel), all(全部)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='输出文件路径（用于csv/excel格式）')
    
    args = parser.parse_args()
    
    # 加载指标文件
    metric_path = Path(args.metric_file)
    if not metric_path.exists():
        print(f"错误: 文件不存在: {metric_path}")
        sys.exit(1)
    
    print(f"正在加载指标文件: {metric_path}")
    metrics_dict = load_pkl(metric_path)
    
    # 转换为表格
    df = format_metrics_to_table(metrics_dict)
    
    # 根据输出格式处理
    if args.output in ['table', 'all']:
        print_table(df)
    
    if args.output in ['csv', 'all']:
        if args.output_path:
            csv_path = Path(args.output_path)
        else:
            csv_path = metric_path.parent / f"{metric_path.stem}_table.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存CSV文件: {csv_path}")
    
    if args.output in ['excel', 'all']:
        try:
            if args.output_path:
                excel_path = Path(args.output_path).with_suffix('.xlsx')
            else:
                excel_path = metric_path.parent / f"{metric_path.stem}_table.xlsx"
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # 添加汇总表
                summary = df.groupby('Dataset')[['MDD', 'Flat_KL', 'MMD_RBF']].mean()
                summary.to_excel(writer, sheet_name='Summary_by_Dataset')
            
            print(f"已保存Excel文件: {excel_path}")
        except ImportError:
            print("警告: 无法保存Excel文件，需要安装openpyxl: pip install openpyxl")
        except Exception as e:
            print(f"保存Excel文件时出错: {e}")

if __name__ == '__main__':
    main()

