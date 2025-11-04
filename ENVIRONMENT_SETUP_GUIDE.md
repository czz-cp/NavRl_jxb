# 训练环境完整配置指南
# Complete Training Environment Setup Guide

## 📋 系统要求

### **硬件要求**

| 组件 | 最低要求 | 推荐配置 |
|-----|---------|---------|
| **GPU** | NVIDIA RTX 2060+ (6GB) | RTX 3090/4090 (24GB) |
| **CPU** | 8核 | 16核+ |
| **内存** | 16GB | 32GB+ |
| **存储** | 100GB 可用空间 | 500GB SSD |

### **软件要求**

- Ubuntu 20.04/22.04
- NVIDIA驱动 >= 525.x
- CUDA 11.8
- Python 3.10
- Conda/Miniconda

---

## 🚀 快速开始（自动配置）

### **方法1: 一键安装**

```bash
cd /home/zar/Downloads/NavRL-main

# 添加执行权限
chmod +x setup_training_env.sh

# 运行安装脚本
./setup_training_env.sh
```

**等待10-15分钟，脚本会自动完成所有配置。**

---

## 🔧 手动配置（详细步骤）

如果自动脚本失败，请按以下步骤手动配置。

### **步骤1: 创建Conda环境**

```bash
# 创建Python 3.10环境
conda create -n navrl_manipulator python=3.10 -y

# 激活环境
conda activate navrl_manipulator

# 验证Python版本
python --version  # 应该显示 Python 3.10.x
```

---

### **步骤2: 安装PyTorch（CUDA 11.8）**

```bash
# 方法A: 使用pip（推荐）
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 方法B: 使用conda（备选）
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**验证安装**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 预期输出:
# PyTorch: 2.0.1+cu118
# CUDA: True
```

---

### **步骤3: 安装TensorDict和TorchRL**

```bash
# 安装TensorDict
pip install tensordict==0.2.0

# 安装TorchRL
pip install torchrl==0.2.0

# 验证
python -c "import tensordict; import torchrl; print('✓ TensorDict & TorchRL installed')"
```

---

### **步骤4: 安装其他依赖**

```bash
# 核心依赖
pip install numpy==1.24.3
pip install scipy
pip install matplotlib

# 配置管理
pip install omegaconf
pip install hydra-core

# 日志和可视化
pip install wandb
pip install tensorboard

# 其他工具
pip install einops
pip install pyyaml
pip install tqdm
```

**创建requirements.txt**:
```bash
cat > requirements.txt << EOF
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
tensordict==0.2.0
torchrl==0.2.0
numpy==1.24.3
scipy
matplotlib
omegaconf
hydra-core
wandb
tensorboard
einops
pyyaml
tqdm
EOF

# 一键安装
pip install -r requirements.txt
```

---

### **步骤5: 安装Isaac Sim**

#### **5.1 下载Isaac Sim**

访问: https://developer.nvidia.com/isaac-sim

1. 注册NVIDIA开发者账号
2. 下载Isaac Sim 2023.1.0（推荐）或更高版本
3. 选择适合您系统的版本

#### **5.2 安装Isaac Sim**

```bash
# 方法A: Omniverse Launcher（推荐）
# 1. 下载Omniverse Launcher
# 2. 在Launcher中安装Isaac Sim

# 方法B: 独立安装包
# 下载 .AppImage 或 容器版本
chmod +x isaac-sim.AppImage
./isaac-sim.AppImage
```

#### **5.3 配置环境变量**

```bash
# 找到Isaac Sim安装路径（通常是下面之一）
# ~/.local/share/ov/pkg/isaac_sim-2023.1.0
# ~/isaac-sim
# /opt/isaac-sim

# 设置环境变量
export ISAAC_SIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2023.1.0"

# 添加到~/.bashrc（永久生效）
echo 'export ISAAC_SIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2023.1.0"' >> ~/.bashrc
echo 'export PYTHONPATH="${ISAAC_SIM_PATH}/exts/omni.isaac.kit:${PYTHONPATH}"' >> ~/.bashrc

source ~/.bashrc
```

#### **5.4 验证Isaac Sim**

```bash
# 启动Isaac Sim
${ISAAC_SIM_PATH}/isaac-sim.sh

# 如果成功打开GUI，说明安装成功
```

---

### **步骤6: 配置Wandb（可选但推荐）**

```bash
# 登录Wandb
wandb login

# 输入API Key（从 https://wandb.ai/authorize 获取）

# 或者设置为离线模式
wandb offline
```

---

## ✅ 环境验证

### **完整验证脚本**

```bash
cd /home/zar/Downloads/NavRL-main

# 创建验证脚本
cat > verify_environment.py << 'EOF'
#!/usr/bin/env python3
"""环境验证脚本"""
import sys

def check_python():
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor == 10:
        print("✓ Python版本正确")
        return True
    else:
        print("❌ 需要Python 3.10")
        return False

def check_pytorch():
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA {torch.version.cuda}")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ GPU数量: {torch.cuda.device_count()}")
            return True
        else:
            print("  ❌ CUDA不可用")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def check_packages():
    packages = {
        'tensordict': 'TensorDict',
        'torchrl': 'TorchRL',
        'numpy': 'NumPy',
        'omegaconf': 'OmegaConf',
        'hydra': 'Hydra',
        'wandb': 'Weights & Biases',
        'einops': 'Einops',
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name} {version}")
        except ImportError:
            print(f"❌ {name} 未安装")
            all_ok = False
    
    return all_ok

def check_isaac_sim():
    import os
    isaac_path = os.environ.get('ISAAC_SIM_PATH')
    
    if isaac_path and os.path.exists(isaac_path):
        print(f"✓ Isaac Sim路径: {isaac_path}")
        return True
    else:
        print("❌ Isaac Sim环境变量未设置或路径不存在")
        print("  请运行: export ISAAC_SIM_PATH=<your_isaac_sim_path>")
        return False

def main():
    print("="*60)
    print("环境验证")
    print("="*60)
    print()
    
    results = []
    
    print("1. Python版本")
    results.append(check_python())
    print()
    
    print("2. PyTorch和CUDA")
    results.append(check_pytorch())
    print()
    
    print("3. 依赖包")
    results.append(check_packages())
    print()
    
    print("4. Isaac Sim")
    results.append(check_isaac_sim())
    print()
    
    print("="*60)
    if all(results):
        print("✓ 所有检查通过！环境配置正确。")
        print("\n可以开始训练:")
        print("  cd isaac-training/training/scripts/")
        print("  python train_manipulator.py")
    else:
        print("❌ 部分检查失败，请根据上面的提示修复。")
    print("="*60)

if __name__ == '__main__':
    main()
EOF

chmod +x verify_environment.py

# 运行验证
python verify_environment.py
```

---

## 🐛 常见问题

### **问题1: CUDA不可用**

```bash
# 检查NVIDIA驱动
nvidia-smi

# 如果失败，重新安装驱动
sudo ubuntu-drivers autoinstall
sudo reboot

# 验证CUDA
nvcc --version
```

### **问题2: PyTorch CUDA版本不匹配**

```bash
# 卸载旧版本
pip uninstall torch torchvision torchaudio

# 重新安装正确版本
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### **问题3: Isaac Sim启动失败**

```bash
# 检查显卡驱动
nvidia-smi

# 检查Vulkan支持
vulkaninfo

# 如果缺少Vulkan
sudo apt install vulkan-utils
```

### **问题4: 内存不足**

```bash
# 减少并行环境数
# 在 train_manipulator.yaml 中:
env:
  num_envs: 128  # 从256减少到128

# 或者使用更小的批次
algo:
  training_frame_num: 8  # 从16减少到8
```

### **问题5: ImportError: omni.isaac.core**

```bash
# Isaac Sim环境变量未正确设置
export ISAAC_SIM_PATH="/path/to/isaac_sim"
export PYTHONPATH="${ISAAC_SIM_PATH}/exts/omni.isaac.kit:${PYTHONPATH}"

# 或使用Isaac Sim的Python
${ISAAC_SIM_PATH}/python.sh train_manipulator.py
```

---

## 📦 完整依赖列表

```txt
# requirements.txt
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
tensordict==0.2.0
torchrl==0.2.0
numpy==1.24.3
scipy>=1.9.0
matplotlib>=3.5.0
omegaconf>=2.3.0
hydra-core>=1.3.0
wandb>=0.15.0
tensorboard>=2.13.0
einops>=0.6.0
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 🚀 开始训练

### **激活环境并训练**

```bash
# 1. 激活conda环境
conda activate navrl_manipulator

# 2. 进入训练目录
cd /home/zar/Downloads/NavRL-main/isaac-training/training/scripts/

# 3. 开始训练（简单场景）
python train_manipulator.py \
    env.num_static_obstacles=0 \
    env.num_dynamic_obstacles=0

# 4. 完整训练
python train_manipulator.py
```

### **监控训练**

```bash
# 终端输出会显示:
# - 训练进度
# - FPS
# - 成功率
# - 损失值

# Wandb仪表板（如果启用）:
# 访问: https://wandb.ai/<your-account>/manipulator-navigation
```

---

## 💡 性能优化建议

### **GPU优化**

```yaml
# train_manipulator.yaml
env:
  num_envs: 512  # RTX 3090: 512, RTX 4090: 1024

algo:
  num_minibatches: 8  # 根据GPU内存调整
```

### **训练加速**

```bash
# 使用混合精度（如果支持）
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# 禁用Wandb（如果不需要）
wandb offline
```

---

## 📞 获取帮助

如果遇到问题:

1. 检查错误日志
2. 运行 `verify_environment.py`
3. 查看 `TROUBLESHOOTING.md`
4. 检查Isaac Sim文档

---

**祝您训练顺利！** 🚀

