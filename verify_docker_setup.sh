#!/bin/bash
# Docker 环境验证脚本
# Verify Docker setup for NavRL training

echo "=========================================="
echo "验证 Docker 训练环境"
echo "Verifying Docker Training Environment"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2"
        return 1
    fi
}

check_pass=true

# 1. 检查 Docker
echo "1. 检查 Docker..."
if check_command docker "Docker 已安装"; then
    docker --version
else
    echo -e "${RED}   请安装 Docker: https://docs.docker.com/engine/install/${NC}"
    check_pass=false
fi
echo ""

# 2. 检查 Docker 权限
echo "2. 检查 Docker 权限..."
if docker ps &> /dev/null; then
    echo -e "${GREEN}✓${NC} 可以运行 Docker (无需 sudo)"
else
    echo -e "${YELLOW}⚠${NC}  需要 sudo 运行 Docker"
    echo "   建议将用户添加到 docker 组:"
    echo "   sudo usermod -aG docker $USER"
    echo "   newgrp docker"
fi
echo ""

# 3. 检查 NVIDIA Driver
echo "3. 检查 NVIDIA 驱动..."
if check_command nvidia-smi "NVIDIA 驱动已安装"; then
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "   驱动版本: $driver_version"
    
    # 检查驱动版本是否 >= 525
    major_version=$(echo $driver_version | cut -d'.' -f1)
    if [ "$major_version" -ge 525 ]; then
        echo -e "${GREEN}✓${NC} 驱动版本符合要求 (>= 525)"
    else
        echo -e "${RED}✗${NC} 驱动版本过低，需要 >= 525"
        check_pass=false
    fi
else
    echo -e "${RED}   请安装 NVIDIA 驱动${NC}"
    check_pass=false
fi
echo ""

# 4. 检查 GPU
echo "4. 检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✓${NC} 检测到 $gpu_count 个 GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | nl
else
    echo -e "${RED}✗${NC} 未检测到 GPU"
    check_pass=false
fi
echo ""

# 5. 检查 NVIDIA Container Toolkit
echo "5. 检查 NVIDIA Container Toolkit..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA Container Toolkit 正常"
else
    echo -e "${RED}✗${NC} NVIDIA Container Toolkit 未正确配置"
    echo "   安装方法: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    check_pass=false
fi
echo ""

# 6. 检查磁盘空间
echo "6. 检查磁盘空间..."
available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$available_space" -ge 100 ]; then
    echo -e "${GREEN}✓${NC} 可用空间: ${available_space}GB (>= 100GB)"
else
    echo -e "${YELLOW}⚠${NC}  可用空间: ${available_space}GB (建议 >= 100GB)"
fi
echo ""

# 7. 检查内存
echo "7. 检查内存..."
total_mem=$(free -g | awk '/^Mem:/{print $2}')
if [ "$total_mem" -ge 32 ]; then
    echo -e "${GREEN}✓${NC} 总内存: ${total_mem}GB (>= 32GB)"
else
    echo -e "${YELLOW}⚠${NC}  总内存: ${total_mem}GB (建议 >= 32GB)"
fi
echo ""

# 8. 检查项目文件
echo "8. 检查项目文件..."
required_files=(
    "Dockerfile"
    "docker-compose.yml"
    "docker_build.sh"
    "docker_run.sh"
    "docker_train.sh"
    "isaac-training/training/scripts/train_manipulator.py"
    "isaac-training/training/scripts/manipulator_env.py"
    "isaac-training/training/scripts/ppo_manipulator.py"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file (缺失)"
        all_files_exist=false
        check_pass=false
    fi
done
echo ""

# 9. 检查脚本权限
echo "9. 检查脚本执行权限..."
scripts=("docker_build.sh" "docker_run.sh" "docker_train.sh")
for script in "${scripts[@]}"; do
    if [ -x "$script" ]; then
        echo -e "${GREEN}✓${NC} $script 可执行"
    else
        echo -e "${YELLOW}⚠${NC}  $script 不可执行 (将自动修复)"
        chmod +x "$script"
    fi
done
echo ""

# 10. 检查网络连接
echo "10. 检查网络连接..."
if ping -c 1 google.com &> /dev/null || ping -c 1 docker.io &> /dev/null; then
    echo -e "${GREEN}✓${NC} 网络连接正常"
else
    echo -e "${YELLOW}⚠${NC}  网络连接可能有问题"
    echo "   构建镜像需要稳定的网络连接"
fi
echo ""

# 总结
echo "=========================================="
if [ "$check_pass" = true ]; then
    echo -e "${GREEN}✅ 所有检查通过！${NC}"
    echo ""
    echo "下一步:"
    echo "  1. 构建 Docker 镜像: ./docker_build.sh"
    echo "  2. 开始训练: ./docker_train.sh"
    echo ""
    echo "详细文档:"
    echo "  - DOCKER_README.md"
    echo "  - DOCKER_DEPLOYMENT_GUIDE.md"
else
    echo -e "${RED}❌ 部分检查未通过${NC}"
    echo ""
    echo "请根据上面的提示修复问题后再继续"
    echo "详细安装指南: DOCKER_DEPLOYMENT_GUIDE.md"
fi
echo "=========================================="

