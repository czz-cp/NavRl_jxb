"""
UR10e Manipulator Training with Isaac Gym
Isaac Gym 版本的训练脚本
"""
# IMPORTANT: Isaac Gym must be imported before PyTorch
# Import isaacgym modules first
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil

import argparse
import yaml
from pathlib import Path

# Import torch after isaacgym
import torch

from manipulator_env_gym import ManipulatorEnvGym
from ppo import PPOTrainer


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 递归转换为对象，并转换字符串数字
    def dict_to_obj(d):
        if isinstance(d, dict):
            obj = type('Config', (), {})()
            for k, v in d.items():
                setattr(obj, k, dict_to_obj(v))
            return obj
        else:
            return d
    
    return dict_to_obj(config_dict)


def main():
    parser = argparse.ArgumentParser(description='Train UR10e Manipulator with Isaac Gym')
    parser.add_argument('--config', type=str, default='config_gym.yaml',
                        help='Path to config file')
    parser.add_argument('--num_steps', type=int, default=10000000,
                        help='Total training steps')
    args = parser.parse_args()
    
    # 加载配置
    print(f"[Main] 加载配置: {args.config}")
    config = load_config(args.config)
    
    # 创建环境
    print("[Main] 创建环境...")
    env = ManipulatorEnvGym(config)
    
    # 创建训练器
    print("[Main] 创建PPO训练器...")
    trainer = PPOTrainer(env, config)
    
    # 开始训练
    trainer.train(args.num_steps)
    
    # 关闭环境
    env.close()
    print("[Main] 训练完成")


if __name__ == '__main__':
    main()
