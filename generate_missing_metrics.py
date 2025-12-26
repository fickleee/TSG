#!/usr/bin/env python3
"""
为所有没有metric_dict.pkl的训练实验生成评估结果
"""

import os
import sys
import torch
from pathlib import Path
from utils.cli_utils import get_parser
from utils.init_utils import init_model_data_trainer
from utils.test_utils import test_model_with_dp, test_model_unseen
from pytorch_lightning.trainer import Trainer

def generate_metrics_for_experiment(exp_dir):
    """为单个实验生成metrics"""
    exp_path = Path(exp_dir)
    exp_name = exp_path.name

    print(f"\n=== 处理实验: {exp_name} ===")

    # 检查是否有last.ckpt
    ckpt_path = exp_path / 'checkpoints' / 'last.ckpt'
    if not ckpt_path.exists():
        print(f"警告: {exp_name} 没有找到last.ckpt，跳过")
        return False

    # 检查是否已经有metrics
    metric_pkl = exp_path / 'metric_dict.pkl'
    if metric_pkl.exists():
        print(f"{exp_name} 已经有metric_dict.pkl，跳过")
        return True

    try:
        # 读取配置
        config_file = exp_path / 'configs' / f'{exp_name}-project.yaml'
        if not config_file.exists():
            # 尝试其他可能的配置文件名
            config_files = list((exp_path / 'configs').glob('*-project.yaml'))
            if config_files:
                config_file = config_files[0]
            else:
                print(f"警告: {exp_name} 没有找到配置文件，跳过")
                return False

        print(f"使用配置文件: {config_file}")
        print(f"使用检查点: {ckpt_path}")

        # 解析参数
        parser = get_parser()
        parser = Trainer.add_argparse_args(parser)

        # 设置基本参数
        sys.argv = ['main_train.py', '--base', str(config_file), '--train', 'false']

        opt = parser.parse_args(sys.argv[1:])

        # 设置logdir为当前实验目录
        opt.logdir = str(exp_path.parent)
        opt.name = exp_name

        # 初始化模型和数据
        model, data, trainer, opt, logdir, melk = init_model_data_trainer(parser)

        # 手动设置checkpoint路径
        if hasattr(trainer, 'ckpt_path'):
            trainer.ckpt_path = str(ckpt_path)

        print(f"加载检查点: {ckpt_path}")
        model.init_from_ckpt(str(ckpt_path))

        model = model.cuda()
        model.eval()

        # 运行评估
        print("开始生成samples和计算metrics...")
        test_model_with_dp(model, data, trainer, opt, str(exp_path))
        test_model_unseen(model, data, trainer, opt, str(exp_path))

        print(f"✓ {exp_name} 评估完成")
        return True

    except Exception as e:
        print(f"✗ {exp_name} 评估失败: {e}")
        return False

def main():
    """主函数"""
    logs_dir = Path('./logs/multi_domain_timedp')

    if not logs_dir.exists():
        print(f"错误: logs目录不存在: {logs_dir}")
        return

    experiments = [d for d in logs_dir.iterdir() if d.is_dir()]
    print(f"找到 {len(experiments)} 个实验")

    success_count = 0
    for exp_dir in experiments:
        if generate_metrics_for_experiment(exp_dir):
            success_count += 1

    print(f"\n=== 总结 ===")
    print(f"总实验数: {len(experiments)}")
    print(f"成功生成metrics: {success_count}")
    print(f"失败/跳过: {len(experiments) - success_count}")

if __name__ == "__main__":
    main()

