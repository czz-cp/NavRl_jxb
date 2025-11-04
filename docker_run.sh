#!/bin/bash
# Docker 运行脚本
# Run NavRL Manipulator Training Container

set -e

echo "=========================================="
echo "启动 NavRL Manipulator 训练容器"
echo "Starting NavRL Manipulator Training Container"
echo "=========================================="
echo ""

# 检查镜像是否存在
if ! docker images navrl-manipulator:latest | grep -q navrl-manipulator; then
    echo "❌ 镜像不存在，请先构建镜像"
    echo "运行: ./docker_build.sh"
    exit 1
fi

# 创建必要的目录
echo "准备工作目录..."
mkdir -p checkpoints
mkdir -p logs
echo "✓ 目录已创建"
echo ""

# 设置 X11 显示（如果需要可视化）
if [ -n "$DISPLAY" ]; then
    xhost +local:docker > /dev/null 2>&1 || true
    echo "✓ X11 显示已配置"
fi

# 停止并删除已存在的容器
if docker ps -a | grep -q navrl_training; then
    echo "停止已存在的容器..."
    docker stop navrl_training > /dev/null 2>&1 || true
    docker rm navrl_training > /dev/null 2>&1 || true
fi

# 启动容器
echo ""
echo "启动容器..."
echo "注意: 容器会自动激活 NavRL conda 环境"
echo ""

docker run -it --rm \
    --name navrl_training \
    --runtime=nvidia \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --env DISPLAY=${DISPLAY:-:0} \
    --env QT_X11_NO_MITSHM=1 \
    --env WANDB_API_KEY=${WANDB_API_KEY:-} \
    --env ENV_NAME=NavRL \
    --volume $(pwd)/isaac-training:/workspace/isaac-training \
    --volume $(pwd)/quick-demos:/workspace/quick-demos \
    --volume $(pwd)/checkpoints:/workspace/checkpoints \
    --volume $(pwd)/logs:/workspace/logs \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --shm-size 8g \
    --workdir /workspace/isaac-training/training/scripts \
    navrl-manipulator:latest \
    bash

echo ""
echo "容器已退出"

