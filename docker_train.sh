#!/bin/bash
# Docker 训练脚本
# Quick training script with Docker

set -e

echo "=========================================="
echo "开始 NavRL Manipulator 训练"
echo "Starting NavRL Manipulator Training"
echo "=========================================="
echo ""

# 检查镜像
if ! docker images navrl-manipulator:latest | grep -q navrl-manipulator; then
    echo "❌ 镜像不存在，请先构建"
    echo "运行: ./docker_build.sh"
    exit 1
fi

# 创建目录
mkdir -p checkpoints logs

# 检查 Weights & Biases API key
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  未设置 WANDB_API_KEY"
    echo "如需使用 W&B 日志，请设置:"
    echo "  export WANDB_API_KEY=your_api_key"
    echo ""
    read -p "继续训练? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 停止已存在的容器
docker stop navrl_training_bg > /dev/null 2>&1 || true
docker rm navrl_training_bg > /dev/null 2>&1 || true

# 启动训练容器（后台）
echo "启动训练容器..."
echo "注意: 容器会自动激活 NavRL conda 环境并开始训练"
echo ""

docker run -d \
    --name navrl_training_bg \
    --runtime=nvidia \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    --env WANDB_API_KEY=${WANDB_API_KEY:-} \
    --env ENV_NAME=NavRL \
    --volume $(pwd)/isaac-training:/workspace/isaac-training \
    --volume $(pwd)/quick-demos:/workspace/quick-demos \
    --volume $(pwd)/checkpoints:/workspace/checkpoints \
    --volume $(pwd)/logs:/workspace/logs \
    --shm-size 8g \
    --workdir /workspace/isaac-training/training/scripts \
    navrl-manipulator:latest \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate NavRL && python train_manipulator.py"

echo "✓ 训练容器已启动"
echo ""
echo "监控命令:"
echo "  查看日志: docker logs -f navrl_training_bg"
echo "  进入容器: docker exec -it navrl_training_bg /bin/bash"
echo "  停止训练: docker stop navrl_training_bg"
echo ""
echo "日志文件位置:"
echo "  ./logs/"
echo "  ./checkpoints/"
echo ""

# 开始跟踪日志
echo "开始显示训练日志 (Ctrl+C 退出，不会停止训练):"
echo "=========================================="
docker logs -f navrl_training_bg

