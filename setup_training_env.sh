#!/bin/bash
# UR10e机械臂训练环境配置脚本
# Training Environment Setup for UR10e Manipulator

set -e  # 遇到错误立即退出

echo "=========================================="
echo "UR10e 机械臂训练环境配置"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Conda未安装！请先安装Anaconda或Miniconda${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Conda已安装${NC}"

# 环境名称
ENV_NAME="env_isaaclab"

# 激活环境
echo ""
echo "激活环境: ${ENV_NAME}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# 安装PyTorch（CUDA 11.8）
echo ""
echo "=========================================="
echo "步骤 2/5: 安装PyTorch (CUDA 11.8)"
echo "=========================================="
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 安装TorchRL和TensorDict
echo ""
echo "=========================================="
echo "步骤 3/5: 安装TorchRL和TensorDict"
echo "=========================================="
pip install tensordict==0.2.0
pip install torchrl==0.2.0

# 安装其他依赖
echo ""
echo "=========================================="
echo "步骤 4/5: 安装其他依赖"
echo "=========================================="
pip install numpy==1.24.3
pip install scipy
pip install matplotlib
pip install omegaconf
pip install hydra-core
pip install wandb
pip install einops
pip install pyyaml
pip install packaging==23.0  # Isaac Sim 要求的版本

# 安装Isaac Sim相关（如果需要）
echo ""
echo "=========================================="
echo "步骤 5/5: 配置Isaac Sim"
echo "=========================================="

echo -e "${YELLOW}"
echo "注意: Isaac Sim需要单独安装"
echo "请按照以下步骤:"
echo "1. 访问 https://developer.nvidia.com/isaac-sim"
echo "2. 下载并安装 Isaac Sim 2023.1.0 或更高版本"
echo "3. 设置环境变量（见下方）"
echo -e "${NC}"

# 创建环境变量脚本
cat > activate_isaac.sh << 'EOF'
#!/bin/bash
# Isaac Sim环境变量配置

# 请根据实际安装路径修改
export ISAAC_SIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2023.1.0"

# 如果Isaac Sim安装在其他位置，请修改上面的路径
if [ -d "$ISAAC_SIM_PATH" ]; then
    export PYTHONPATH="${ISAAC_SIM_PATH}/exts/omni.isaac.kit:${ISAAC_SIM_PATH}/exts/omni.isaac.core:${PYTHONPATH}"
    echo "✓ Isaac Sim环境已配置: $ISAAC_SIM_PATH"
else
    echo "⚠️  警告: Isaac Sim未找到，请修改 activate_isaac.sh 中的路径"
fi
EOF

chmod +x activate_isaac.sh

# 验证安装
echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="

python << 'PYEOF'
import sys
print(f"Python版本: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch导入失败: {e}")

try:
    import tensordict
    print(f"✓ TensorDict {tensordict.__version__}")
except ImportError as e:
    print(f"❌ TensorDict导入失败: {e}")

try:
    import torchrl
    print(f"✓ TorchRL {torchrl.__version__}")
except ImportError as e:
    print(f"❌ TorchRL导入失败: {e}")

try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
except ImportError as e:
    print(f"❌ NumPy导入失败: {e}")

try:
    import omegaconf
    print(f"✓ OmegaConf {omegaconf.__version__}")
except ImportError as e:
    print(f"❌ OmegaConf导入失败: {e}")

try:
    import wandb
    print(f"✓ Weights & Biases {wandb.__version__}")
except ImportError as e:
    print(f"❌ Wandb导入失败: {e}")

print("\n所有核心依赖已安装！")
PYEOF

# 完成
echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo -e "${GREEN}下一步操作:${NC}"
echo ""
echo "1. 激活环境:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "2. 配置Isaac Sim（如果尚未安装）:"
echo "   - 下载: https://developer.nvidia.com/isaac-sim"
echo "   - 安装后修改 activate_isaac.sh 中的路径"
echo "   - 运行: source activate_isaac.sh"
echo ""
echo "3. 登录Wandb（可选）:"
echo "   wandb login"
echo ""
echo "4. 开始训练:"
echo "   cd isaac-training/training/scripts/"
echo "   python train_manipulator.py"
echo ""
echo -e "${YELLOW}注意: 首次训练会比较慢，因为需要编译CUDA kernels${NC}"
echo ""

