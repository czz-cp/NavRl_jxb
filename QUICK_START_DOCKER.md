# 🚀 Docker 训练快速开始

## 📋 前提条件

确保服务器已安装：
- Ubuntu 20.04/22.04
- NVIDIA GPU (8GB+ 显存)
- NVIDIA Driver >= 525
- Docker
- NVIDIA Container Toolkit

## ⚡ 快速开始（3 步）

### 步骤 1: 验证环境

```bash
cd /home/zar/Downloads/NavRL-main
./verify_docker_setup.sh
```

如果所有检查通过，继续下一步。如果有问题，参考 [`DOCKER_DEPLOYMENT_GUIDE.md`](DOCKER_DEPLOYMENT_GUIDE.md) 的安装说明。

### 步骤 2: 构建镜像

```bash
./docker_build.sh
```

⏱️ 预计时间：10-30 分钟（取决于网络速度）

### 步骤 3: 开始训练

```bash
# 可选：设置 Weights & Biases
export WANDB_API_KEY="your_api_key"

# 开始训练
./docker_train.sh
```

## 📊 监控训练

```bash
# 实时查看日志
docker logs -f navrl_training_bg

# 监控 GPU
watch -n 1 nvidia-smi

# 查看训练文件
ls -lh checkpoints/  # 模型检查点
ls -lh logs/         # 训练日志
```

## ⚙️ 配置调整

根据你的 GPU 显存调整环境数：

编辑 `isaac-training/training/cfg/train_manipulator.yaml`：

```yaml
train:
  num_envs: 128  # 根据显存: 12GB→64, 24GB→128, 40GB→256
```

## 🎯 训练进度

典型训练进度（RTX 3090 / 24GB）：

- **总帧数**: 50,000,000
- **并行环境**: 128
- **预计时长**: 24-48 小时
- **检查点保存**: 每 1,000,000 帧

训练指标：
- `reward_mean`: 平均奖励（越高越好）
- `success_rate`: 成功率（目标 > 0.8）
- `collision_rate`: 碰撞率（目标 < 0.1）

## 🛠️ 常用命令

```bash
# 查看运行中的容器
docker ps

# 进入容器
docker exec -it navrl_training_bg bash

# 停止训练
docker stop navrl_training_bg

# 重启训练
./docker_train.sh

# 清理容器
docker rm navrl_training_bg

# 清理镜像
docker rmi navrl-manipulator:latest
```

## 📁 输出文件

训练完成后，检查以下文件：

```
checkpoints/
├── checkpoint_1000000.pt   # 100万帧检查点
├── checkpoint_5000000.pt   # 500万帧检查点
├── ...
└── checkpoint_50000000.pt  # 5000万帧检查点（最终）

logs/
├── training.log            # 训练日志
└── tensorboard/            # TensorBoard 日志
```

## 🔍 评估模型

训练完成后，使用检查点进行评估：

```bash
# 进入容器
docker exec -it navrl_training_bg bash

# 运行评估
python eval_manipulator.py \
    --checkpoint /workspace/NavRL-main/checkpoints/checkpoint_50000000.pt \
    --num_episodes 100
```

## 🐛 遇到问题？

### GPU 不可用

```bash
# 测试 NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 如果失败，重启 Docker
sudo systemctl restart docker
```

### 内存不足

```yaml
# 减少并行环境数
# 编辑 train_manipulator.yaml
num_envs: 64  # 改为更小的值
```

### 训练中断

```bash
# Docker 会自动保存检查点
# 查看最新的检查点
ls -lth checkpoints/

# 从检查点恢复（需要在代码中实现）
python train_manipulator.py --resume checkpoints/checkpoint_XXX.pt
```

## 📚 详细文档

- **完整部署指南**: [`DOCKER_DEPLOYMENT_GUIDE.md`](DOCKER_DEPLOYMENT_GUIDE.md)
- **快速参考**: [`DOCKER_README.md`](DOCKER_README.md)
- **实现指南**: [`MANIPULATOR_IMPLEMENTATION_GUIDE.md`](MANIPULATOR_IMPLEMENTATION_GUIDE.md)
- **系统概述**: [`MANIPULATOR_SYSTEM_OVERVIEW.md`](MANIPULATOR_SYSTEM_OVERVIEW.md)

## ✅ 下一步

训练完成后：

1. **评估模型**: 在仿真环境测试
2. **部署到实机**: 参考 ROS1 部署文档
3. **参数微调**: 根据实际表现调整
4. **继续训练**: 从检查点继续优化

---

## 🎉 就是这么简单！

3 个命令开始训练：

```bash
./verify_docker_setup.sh  # 验证环境
./docker_build.sh          # 构建镜像
./docker_train.sh          # 开始训练
```

**祝训练顺利！** 🚀

