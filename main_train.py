# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os, sys

# 设置 protobuf 使用纯 Python 实现（解决 Windows DLL 问题）
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# 强制 wandb 使用离线模式（修复 Windows 上的连接和目录问题）
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_SILENT'] = 'true'

from pytorch_lightning.trainer import Trainer
from utils.cli_utils import get_parser
from utils.init_utils import init_model_data_trainer
from utils.test_utils import test_model_with_dp, test_model_uncond, test_model_unseen


if __name__ == "__main__":
    
    data_root = os.environ['DATA_ROOT']

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    model, data, trainer, opt, logdir, melk = init_model_data_trainer(parser)

    # run
    if opt.train:
        try:
            # 确保 wandb 目录存在（修复 Windows 路径问题）
            if hasattr(trainer.logger, 'save_dir') and trainer.logger.save_dir:
                wandb_dir = os.path.join(trainer.logger.save_dir, 'wandb')
                os.makedirs(wandb_dir, exist_ok=True)
                # 设置环境变量确保 wandb 使用正确的目录
                os.environ['WANDB_DIR'] = wandb_dir
            
            # 安全地更新 logger 配置（TensorBoard 不需要配置更新）
            try:
                from pytorch_lightning.loggers import TensorBoardLogger
                if not isinstance(trainer.logger, TensorBoardLogger):
                    # 只有非 TensorBoard logger 才需要更新配置
                    if hasattr(trainer.logger, 'experiment') and trainer.logger.experiment is not None:
                        trainer.logger.experiment.config.update(opt)
            except Exception as e:
                print(f"Warning: Failed to update logger config (non-critical): {e}")
                print("Training will continue...")
            
            # 注意：resume_from_checkpoint 已经在 init_model_data_trainer 中设置
            # 这里直接调用 fit() 即可
            trainer.fit(model, data)
        except Exception as e:
            # 只在 model 已初始化时才尝试保存检查点
            if hasattr(trainer, 'model') and trainer.model is not None:
                try:
                    melk()
                except Exception as melk_error:
                    print(f"Warning: Failed to save checkpoint on error: {melk_error}")
            raise
    if not opt.no_test and not trainer.interrupted:
        if opt.uncond:
            test_model_uncond(model, data, trainer, opt, logdir)
        else:
            test_model_with_dp(model, data, trainer, opt, logdir)
            test_model_unseen(model, data, trainer, opt, logdir)

