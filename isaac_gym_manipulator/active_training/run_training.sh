#!/bin/bash
# Active Training 启动脚本

# 设置库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 获取脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 设置 Python 路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 切换到脚本目录
cd "$SCRIPT_DIR"

# 运行训练
python train.py "$@"
