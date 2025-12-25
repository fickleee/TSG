# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import argparse
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from utils.cli_utils import nondefault_trainer_args
from utils.callback_utils import prepare_trainer_configs
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from pathlib import Path
import datetime
from utils.cli_utils import nondefault_trainer_args
from utils.data_utils import get_data_root

data_root = str(get_data_root())

def init_model_data_trainer(parser):
    
    opt, unknown = parser.parse_known_args()
    
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    # 首先从base提取cfg_name（用于目录结构）
    if opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
    else:
        cfg_name = "default"
    
    # 然后确定name（用于实验名称）
    if opt.name:
        name = opt.name
    elif opt.base:
        name = cfg_name
    else:
        name = ""

    seed_everything(opt.seed)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    # Customize config from opt:
    config.model['params']['seq_len'] = opt.seq_len
    config.model['params']['unet_config']['params']['seq_len'] = opt.seq_len
    config.data['params']['window'] = opt.seq_len
    config.data['params']['batch_size'] = opt.batch_size
    bs = opt.batch_size
    if opt.max_steps:
        config.lightning['trainer']['max_steps'] = opt.max_steps
    if opt.debug:
        config.lightning['trainer']['max_steps'] = 10
        config.lightning['callbacks']['image_logger']['params']['batch_frequency'] = 5
    if opt.overwrite_learning_rate is not None:
        config.model['base_learning_rate'] = opt.overwrite_learning_rate
        print(f"Setting learning rate (overwritting config file) to {opt.overwrite_learning_rate:.2e}")
        base_lr = opt.overwrite_learning_rate
    else:
        base_lr = config.model['base_learning_rate']
        
    nowname = f"{name.split('-')[-1]}_{opt.seq_len}_nl_{opt.num_latents}_lr{base_lr:.1e}_bs{opt.batch_size}"
    
    
    if opt.uncond:
        config.model['params']['cond_stage_config'] = "__is_unconditional__"
        config.model['params']['cond_stage_trainable'] = False
        config.model['params']['unet_config']['params']['context_dim'] = None
        nowname += f"_uncond"
    else:
        config.model['params']['cond_stage_config']['params']['window'] = opt.seq_len

        if opt.use_pam:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedPrototyper"
            config.model['params']['cond_stage_config']['params']['num_latents'] = opt.num_latents
            config.model['params']['unet_config']['params']['latent_unit'] = opt.num_latents
            config.model['params']['unet_config']['params']['use_pam'] = True
            
            # PAM GNN相关参数
            if hasattr(opt, 'use_gnn') and opt.use_gnn:
                config.model['params']['cond_stage_config']['params']['use_gnn'] = True
                if hasattr(opt, 'num_variables') and opt.num_variables is not None:
                    config.model['params']['cond_stage_config']['params']['num_variables'] = opt.num_variables
                if hasattr(opt, 'gnn_layers'):
                    config.model['params']['cond_stage_config']['params']['gnn_layers'] = opt.gnn_layers
                if hasattr(opt, 'gnn_type'):
                    config.model['params']['cond_stage_config']['params']['gnn_type'] = opt.gnn_type
                nowname += f"_gnn{opt.gnn_type}"
            else:
                config.model['params']['cond_stage_config']['params']['use_gnn'] = False
            
            nowname += f"_pam"
        
        # UNet Bottleneck GNN相关参数
        if hasattr(opt, 'use_gnn_in_unet') and opt.use_gnn_in_unet:
            config.model['params']['unet_config']['params']['use_gnn_in_unet'] = True
            if hasattr(opt, 'unet_gnn_type'):
                config.model['params']['unet_config']['params']['unet_gnn_type'] = opt.unet_gnn_type
            if hasattr(opt, 'unet_gnn_layers'):
                config.model['params']['unet_config']['params']['unet_gnn_layers'] = opt.unet_gnn_layers
            if hasattr(opt, 'unet_gnn_hidden_dim'):
                config.model['params']['unet_config']['params']['unet_gnn_hidden_dim'] = opt.unet_gnn_hidden_dim
            if hasattr(opt, 'unet_gnn_heads'):
                config.model['params']['unet_config']['params']['unet_gnn_heads'] = opt.unet_gnn_heads
            nowname += f"_unetgnn{opt.unet_gnn_type}"
        else:
            config.model['params']['unet_config']['params']['use_gnn_in_unet'] = False
        
        if not opt.use_pam:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedEncoder"
            config.model['params']['unet_config']['params']['use_pam'] = False
            
    nowname += f"_seed{opt.seed}"
    logdir = os.path.join(opt.logdir, cfg_name, nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    
    metrics_dir = Path(logdir) / 'metric_dict.pkl'
    if metrics_dir.exists():
        print(f"Metric exists! Skipping {nowname}")
        sys.exit(0)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # 先从 opt 中获取非默认参数
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    
    # 特殊处理 gpus 参数：如果命令行传入的是字符串 "0,"，但被解析为整数 0
    # 需要检查原始命令行参数并恢复字符串格式
    if hasattr(opt, 'gpus') and isinstance(opt.gpus, int) and opt.gpus == 0:
        # 检查是否在命令行中指定了 --gpus 0,（字符串格式）
        gpus_arg_found = False
        for i, arg in enumerate(sys.argv):
            if arg in ['--gpus', '-gpus'] and i + 1 < len(sys.argv):
                gpus_value = sys.argv[i + 1]
                if isinstance(gpus_value, str) and (',' in gpus_value or gpus_value.isdigit()):
                    # 如果命令行中是 "0," 或 "0"，使用字符串格式
                    trainer_config['gpus'] = gpus_value  # 使用原始的字符串格式
                    gpus_arg_found = True
                    print(f"Debug: Found gpus in command line: {gpus_value}, using string format")
                    break
        if not gpus_arg_found:
            # 如果没有找到字符串格式，但 opt.gpus 是 0，说明可能没有指定 GPU
            pass
    
    # 根据是否有 gpus 来设置 accelerator
    if "gpus" in trainer_config and trainer_config["gpus"] is not None:
        # PyTorch Lightning 1.4.2 需要字符串格式如 "0," 或整数列表，不能是整数 0
        gpus_val = trainer_config["gpus"]
        if isinstance(gpus_val, str) and gpus_val.strip():
            trainer_config["accelerator"] = "gpu"
            gpuinfo = gpus_val
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        elif isinstance(gpus_val, (int, list)) and gpus_val != 0 and gpus_val != []:
            # 如果是整数或列表且不为 0/空，使用 GPU
            trainer_config["accelerator"] = "gpu"
            gpuinfo = gpus_val
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        else:
            # gpus 为 0 或 None 或空字符串，不使用 GPU
            if "accelerator" in trainer_config:
                del trainer_config["accelerator"]
            cpu = True
    else:
        # 如果没有 gpus，删除 accelerator（如果存在）并使用 CPU
        if "accelerator" in trainer_config:
            del trainer_config["accelerator"]
        cpu = True
    # 将 OmegaConf 转换为普通字典（如果它是 OmegaConf 对象）
    if hasattr(trainer_config, '_content'):
        trainer_config_dict = dict(trainer_config)
    else:
        trainer_config_dict = trainer_config
    
    trainer_opt = argparse.Namespace(**trainer_config_dict)
    lightning_config.trainer = trainer_config_dict

    # model
    # 注意：resume 功能在 PyTorch Lightning 1.4.2 中通过 Trainer 的 resume_from_checkpoint 参数实现
    # 这里不再设置 config.model['params']['ckpt_path']，而是在创建 Trainer 后设置
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = prepare_trainer_configs(nowname, logdir, opt, lightning_config, ckptdir, model, now, cfgdir, config, trainer_opt)
    
    # 将 trainer_config_dict 中的所有参数合并到 trainer_kwargs 中
    # 这确保 gpus 和 accelerator 等参数都能正确传递（修复 PyTorch Lightning 1.4.2 的 bug）
    for k, v in trainer_config_dict.items():
        if k not in trainer_kwargs:  # 避免覆盖 prepare_trainer_configs 设置的参数
            trainer_kwargs[k] = v
    
    # 确保 gpus 和 accelerator 优先使用 trainer_config_dict 中的值
    # 重要：如果 trainer_config_dict 中的 gpus 是字符串（如 "0,"），确保保持字符串格式
    if 'gpus' in trainer_config_dict:
        gpus_val = trainer_config_dict['gpus']
        # 如果是字符串格式（如 "0,"），直接使用
        if isinstance(gpus_val, str):
            trainer_kwargs['gpus'] = gpus_val
            print(f"Debug: Using gpus string format from trainer_config_dict: {gpus_val}")
        # 如果是整数 0，检查命令行参数并尝试恢复字符串格式
        elif isinstance(gpus_val, int) and gpus_val == 0:
            for i, arg in enumerate(sys.argv):
                if arg in ['--gpus', '-gpus'] and i + 1 < len(sys.argv):
                    cmd_gpus = sys.argv[i + 1]
                    if isinstance(cmd_gpus, str) and (',' in cmd_gpus or cmd_gpus.isdigit()):
                        trainer_kwargs['gpus'] = cmd_gpus
                        trainer_kwargs['accelerator'] = 'gpu'  # 同时设置 accelerator
                        print(f"Debug: Recovered gpus from command line: {cmd_gpus}, setting accelerator='gpu'")
                        break
            else:
                # 如果没有找到，使用整数 0（不使用 GPU）
                trainer_kwargs['gpus'] = gpus_val
        else:
            trainer_kwargs['gpus'] = gpus_val
    if 'accelerator' in trainer_config_dict:
        trainer_kwargs['accelerator'] = trainer_config_dict['accelerator']
    
    # 调试信息：检查关键参数
    if 'gpus' in trainer_config_dict or 'accelerator' in trainer_config_dict:
        print(f"Debug: trainer_config_dict = {trainer_config_dict}")
        print(f"Debug: trainer_kwargs contains 'gpus': {'gpus' in trainer_kwargs}, value: {trainer_kwargs.get('gpus', 'NOT FOUND')}, type: {type(trainer_kwargs.get('gpus', None))}")
        print(f"Debug: trainer_kwargs contains 'accelerator': {'accelerator' in trainer_kwargs}, value: {trainer_kwargs.get('accelerator', 'NOT FOUND')}")
        print(f"Debug: trainer_opt contains 'gpus': {hasattr(trainer_opt, 'gpus')}, value: {getattr(trainer_opt, 'gpus', 'NOT FOUND')}, type: {type(getattr(trainer_opt, 'gpus', None))}")
        print(f"Debug: trainer_opt contains 'accelerator': {hasattr(trainer_opt, 'accelerator')}, value: {getattr(trainer_opt, 'accelerator', 'NOT FOUND')}")
    
    # 问题：Trainer.from_argparse_args 在处理参数时可能有 bug，即使 gpus 在 kwargs 中也会报错
    # 解决方案：直接从 trainer_kwargs 创建 Trainer，不依赖 from_argparse_args
    # 但需要确保所有必要的参数都在 trainer_kwargs 中（包括从 trainer_opt 中的参数）
    
    # 将 trainer_opt 中的所有参数也添加到 trainer_kwargs 中（如果还没有的话）
    for k, v in vars(trainer_opt).items():
        if k not in trainer_kwargs:
            trainer_kwargs[k] = v
    
    # 如果指定了恢复训练，在创建Trainer之前设置resume_from_checkpoint
    # 注意：resume_from_checkpoint 必须在创建 Trainer 时作为参数传入，不能之后设置
    if opt.resume:
        ckpt_path = os.path.join(logdir, "checkpoints", "last.ckpt")
        if os.path.exists(ckpt_path):
            trainer_kwargs['resume_from_checkpoint'] = ckpt_path
            print(f"Trainer will resume from checkpoint: {ckpt_path}")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}, will start from scratch")
    
    # 直接使用 Trainer(**trainer_kwargs) 创建，避免 from_argparse_args 的 bug
    trainer = Trainer(**trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    for k, v in config.data.params.data_path_dict.items():
        config.data.params.data_path_dict[k] = v.replace('{DATA_ROOT}', data_root).replace('{SEQ_LEN}', str(opt.seq_len))
    
    # 如果启用GNN，准备num_variables_dict
    num_variables_dict = None
    if hasattr(opt, 'use_gnn') and opt.use_gnn:
        # 可以从配置文件或命令行参数中获取每个数据集的变量数
        # 这里先使用一个简单的映射（可以从配置文件扩展）
        if hasattr(config.data.params, 'num_variables_dict'):
            num_variables_dict = config.data.params.num_variables_dict
        else:
            # 如果没有指定，将在数据加载时自动推断
            num_variables_dict = {}
            print("Warning: num_variables_dict not specified. Will infer from data or use default.")
    
    if num_variables_dict is not None:
        config.data.params.num_variables_dict = num_variables_dict
    
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    assert config.data.params.input_channels == 1, \
        "Assertion failed: Only univariate input is supported. Please ensure input_channels == 1."
    print("#### Data Preparation Finished #####")
    
    # 将数据集的变量数信息传递给模型（用于GNN和UNet GNN）
    if hasattr(data, 'num_variables_dict') and data.num_variables_dict:
        # 将数据集名称映射转换为索引映射（data_key是索引）
        # data.key_list 是数据集名称列表，索引对应data_key
        model_num_variables_dict = {}
        for idx, data_name in enumerate(data.key_list):
            if data_name in data.num_variables_dict:
                model_num_variables_dict[idx] = data.num_variables_dict[data_name]
        if model_num_variables_dict:
            model.num_variables_dict = model_num_variables_dict
            print(f"Model num_variables_dict: {model_num_variables_dict}")
            
            # 将变量数字典传递给UNet配置（用于UNet GNN）
            if hasattr(opt, 'use_gnn_in_unet') and opt.use_gnn_in_unet:
                config.model['params']['unet_config']['params']['num_variables_dict'] = model_num_variables_dict
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")
        
    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        # 只在 model 已初始化时才保存检查点
        if trainer.global_rank == 0 and hasattr(trainer, 'model') and trainer.model is not None:
            try:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb; # type: ignore
            pudb.set_trace()

    import signal

    # Windows 不支持 SIGUSR1 和 SIGUSR2，只在 Unix 系统上注册信号处理器
    if sys.platform != 'win32':
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, melk)
        if hasattr(signal, 'SIGUSR2'):
            signal.signal(signal.SIGUSR2, divein)
    
    return model, data, trainer, opt, logdir, melk


def load_model_data(parser):
    
    opt, unknown = parser.parse_known_args()
        
    if opt.name:
        name = opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = cfg_name
    else:
        name = ""

    seed_everything(opt.seed)

    # try:
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    # Customize config from opt:
    config.model['params']['seq_len'] = opt.seq_len
    config.model['params']['unet_config']['params']['seq_len'] = opt.seq_len
    config.data['params']['window'] = opt.seq_len
    config.data['params']['batch_size'] = opt.batch_size
    bs = opt.batch_size
    if opt.max_steps:
        config.lightning['trainer']['max_steps'] = opt.max_steps
    if opt.debug:
        config.lightning['trainer']['max_steps'] = 10
        config.lightning['callbacks']['image_logger']['params']['batch_frequency'] = 5
    if opt.overwrite_learning_rate is not None:
        config.model['base_learning_rate'] = opt.overwrite_learning_rate
        print(f"Setting learning rate (overwritting config file) to {opt.overwrite_learning_rate:.2e}")
        base_lr = opt.overwrite_learning_rate
    else:
        base_lr = config.model['base_learning_rate']
        
    nowname = f"{name.split('-')[-1]}_{opt.seq_len}_nl_{opt.num_latents}_lr{base_lr:.1e}_bs{opt.batch_size}"    
    
    if opt.uncond:
        config.model['params']['cond_stage_config'] = "__is_unconditional__"
        config.model['params']['cond_stage_trainable'] = False
        config.model['params']['unet_config']['params']['context_dim'] = None
        nowname += f"_uncond"
    else:
        config.model['params']['cond_stage_config']['params']['window'] = opt.seq_len

        if opt.use_pam:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedPrototyper"
            config.model['params']['cond_stage_config']['params']['num_latents'] = opt.num_latents
            config.model['params']['unet_config']['params']['latent_unit'] = opt.num_latents
            config.model['params']['unet_config']['params']['use_pam'] = True
            
            # GNN相关参数（load_model_data函数中也需要）
            if hasattr(opt, 'use_gnn') and opt.use_gnn:
                config.model['params']['cond_stage_config']['params']['use_gnn'] = True
                if hasattr(opt, 'num_variables') and opt.num_variables is not None:
                    config.model['params']['cond_stage_config']['params']['num_variables'] = opt.num_variables
                if hasattr(opt, 'gnn_layers'):
                    config.model['params']['cond_stage_config']['params']['gnn_layers'] = opt.gnn_layers
                if hasattr(opt, 'gnn_type'):
                    config.model['params']['cond_stage_config']['params']['gnn_type'] = opt.gnn_type
                nowname += f"_gnn{opt.gnn_type}"
            else:
                config.model['params']['cond_stage_config']['params']['use_gnn'] = False
            
            nowname += f"_pam"
        else:
            config.model['params']['cond_stage_config']['target'] = "ldm.modules.encoders.modules.DomainUnifiedEncoder"
            config.model['params']['unet_config']['params']['use_pam'] = False
        
        # UNet Bottleneck GNN相关参数（load_model_data函数中也需要）
        if hasattr(opt, 'use_gnn_in_unet') and opt.use_gnn_in_unet:
            config.model['params']['unet_config']['params']['use_gnn_in_unet'] = True
            if hasattr(opt, 'unet_gnn_type'):
                config.model['params']['unet_config']['params']['unet_gnn_type'] = opt.unet_gnn_type
            if hasattr(opt, 'unet_gnn_layers'):
                config.model['params']['unet_config']['params']['unet_gnn_layers'] = opt.unet_gnn_layers
            if hasattr(opt, 'unet_gnn_hidden_dim'):
                config.model['params']['unet_config']['params']['unet_gnn_hidden_dim'] = opt.unet_gnn_hidden_dim
            if hasattr(opt, 'unet_gnn_heads'):
                config.model['params']['unet_config']['params']['unet_gnn_heads'] = opt.unet_gnn_heads
            nowname += f"_unetgnn{opt.unet_gnn_type}"
        else:
            config.model['params']['unet_config']['params']['use_gnn_in_unet'] = False
    
    nowname += f"_seed{opt.seed}"
    logdir = os.path.join(opt.logdir, cfg_name, nowname)
    
    # model
    ckpt_name = opt.ckpt_name
    ckpt_path = logdir / 'checkpoints' / f'{ckpt_name}.ckpt'
    config.model['params']['ckpt_path'] = ckpt_path
    model = instantiate_from_config(config.model)

    # data
    for k, v in config.data.params.data_path_dict.items():
        config.data.params.data_path_dict[k] = v.replace('{DATA_ROOT}', data_root).replace('{SEQ_LEN}', str(opt.seq_len))
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data Preparation Finished #####")
    
    return model, data, opt, logdir
