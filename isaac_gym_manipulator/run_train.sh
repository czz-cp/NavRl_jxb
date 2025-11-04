#!/bin/bash
# 设置库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 启用 CUDA 同步错误报告
export CUDA_LAUNCH_BLOCKING=1

# 切换到脚本目录
cd "$(dirname "$0")"

# 运行训练
python train_gym.py --config config_gym.yaml "$@"
