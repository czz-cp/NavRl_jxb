#!/bin/bash
# Docker 构建脚本
# Build Docker image for NavRL Manipulator Training

set -e

echo "=========================================="
echo "构建 NavRL Manipulator 训练镜像"
echo "Building NavRL Manipulator Training Image"
echo "=========================================="
echo ""

# 检查 Docker 和 NVIDIA Docker
echo "检查环境..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装"
    echo "请访问: https://docs.docker.com/engine/install/"
    exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker runtime 未正确配置"
    echo "请访问: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "✓ Docker 环境正常"
echo ""

# 显示系统信息
echo "系统信息:"
echo "  Docker 版本: $(docker --version)"
echo "  GPU 信息:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
echo ""

# 构建镜像
echo "开始构建镜像..."
echo "这可能需要 10-30 分钟，取决于网络速度..."
echo ""

docker build \
    --tag navrl-manipulator:latest \
    --progress=plain \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 镜像构建成功！"
    echo "=========================================="
    echo ""
    echo "镜像信息:"
    docker images navrl-manipulator:latest
    echo ""
    echo "下一步:"
    echo "  启动容器: ./docker_run.sh"
    echo "  或使用 docker-compose: docker-compose up -d"
else
    echo ""
    echo "❌ 镜像构建失败"
    exit 1
fi

