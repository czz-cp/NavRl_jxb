# NavRL Manipulator Docker 快速参考

## 🚀 三步开始训练

```bash
# 1. 构建镜像
./docker_build.sh

# 2. 设置 W&B (可选)
export WANDB_API_KEY="your_api_key"

# 3. 开始训练
./docker_train.sh
```

## 📋 文件说明

| 文件 | 用途 |
|------|------|
| `Dockerfile` | Docker 镜像定义 |
| `docker-compose.yml` | Docker Compose 配置 |
| `docker_build.sh` | 构建镜像脚本 |
| `docker_run.sh` | 交互式运行容器 |
| `docker_train.sh` | 后台运行训练 |
| `.dockerignore` | Docker 构建忽略文件 |
| `DOCKER_DEPLOYMENT_GUIDE.md` | 完整部署指南 |

## ⚡ 常用命令

### 构建和运行

```bash
# 构建镜像
./docker_build.sh

# 交互式进入容器
./docker_run.sh

# 后台运行训练
./docker_train.sh

# 使用 docker-compose
docker-compose up -d
docker-compose exec navrl-training bash
```

### 监控和管理

```bash
# 查看日志
docker logs -f navrl_training_bg

# 查看 GPU
nvidia-smi

# 进入运行中的容器
docker exec -it navrl_training_bg bash

# 停止训练
docker stop navrl_training_bg
```

### 清理

```bash
# 停止并删除容器
docker stop navrl_training_bg
docker rm navrl_training_bg

# 删除镜像
docker rmi navrl-manipulator:latest

# 清理所有未使用的 Docker 资源
docker system prune -a
```

## 🔧 配置调整

### 根据 GPU 显存调整环境数

编辑 `isaac-training/training/cfg/train_manipulator.yaml`:

```yaml
train:
  num_envs: 128  # 12GB: 64, 24GB: 128, 40GB: 256
```

### 修改障碍物数量

```yaml
env:
  num_static_obstacles: 5   # 静态障碍物
  num_dynamic_obstacles: 3  # 动态气球
```

## 📊 训练输出

- **检查点**: `./checkpoints/`
- **日志**: `./logs/`
- **W&B**: https://wandb.ai/your_entity/your_project

## 🐛 常见问题

### GPU 不可用

```bash
# 检查 NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 重启 Docker
sudo systemctl restart docker
```

### 内存不足

```yaml
# 减少环境数
num_envs: 64  # 改为 32 或 64
```

### 镜像构建失败

```bash
# 清理缓存
docker system prune -a

# 重新构建
./docker_build.sh
```

## 📚 详细文档

完整部署指南请参考: [`DOCKER_DEPLOYMENT_GUIDE.md`](DOCKER_DEPLOYMENT_GUIDE.md)

## 🎯 训练流程

1. **环境准备** → 安装 Docker 和 NVIDIA Container Toolkit
2. **构建镜像** → `./docker_build.sh` (10-30 分钟)
3. **配置参数** → 编辑 `train_manipulator.yaml`
4. **开始训练** → `./docker_train.sh`
5. **监控进度** → W&B / TensorBoard / 日志
6. **评估模型** → 使用保存的检查点

## 💡 提示

- **首次运行**: 需要下载 ~10GB 的基础镜像
- **GPU 利用**: 训练时 GPU 利用率应在 80%+
- **保存频率**: 默认每 100 万帧保存一次检查点
- **训练时长**: 5000 万帧约需 24-48 小时 (取决于硬件)

## ✅ 环境要求

- **操作系统**: Ubuntu 20.04/22.04
- **GPU**: NVIDIA (8GB+ 显存)
- **驱动**: >= 525
- **内存**: 32GB+
- **磁盘**: 100GB+

---

**需要帮助?** 查看 [完整部署指南](DOCKER_DEPLOYMENT_GUIDE.md) 或提交 Issue。

