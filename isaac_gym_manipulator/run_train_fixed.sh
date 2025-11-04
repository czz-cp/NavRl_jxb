#!/bin/bash

# 🔧 修复 Isaac Gym 环境变量问题
# 设置 LD_LIBRARY_PATH 以找到 libpython3.8.so.1.0

echo "======================================================================"
echo "Isaac Gym UR10 训练脚本（带环境修复）"
echo "======================================================================"

# 设置 Python 库路径
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    echo "[Fix] 设置 LD_LIBRARY_PATH=$CONDA_PREFIX/lib"
else
    echo "[Warning] CONDA_PREFIX 未设置，跳过环境修复"
fi

# 进入工作目录
cd /home/zar/Downloads/NavRL-main/isaac_gym_manipulator

# 运行训练
echo "[Train] 启动训练..."
python train_gym.py --config config_gym.yaml "$@"

