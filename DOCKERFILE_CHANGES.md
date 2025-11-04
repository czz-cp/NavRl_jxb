# Dockerfile 修改说明

## 🔄 主要修改

根据您的新 Dockerfile，我已经更新了相关的 Docker 脚本和配置文件。

### 📋 修改的文件

1. **`docker_run.sh`** - 交互式运行脚本
2. **`docker_train.sh`** - 后台训练脚本  
3. **`docker-compose.yml`** - Docker Compose 配置

---

## 🆕 新 Dockerfile 特点

### 基础镜像
```dockerfile
FROM nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix.1
```
- 使用 Isaac Sim 2023.1.0 版本
- 包含完整的 Isaac Sim 环境

### Conda 环境
```dockerfile
ENV ENV_NAME=NavRL
RUN conda create -n $ENV_NAME python=3.10 -c conda-forge -y
```
- 创建名为 `NavRL` 的 conda 环境
- Python 3.10 版本

### 依赖安装
- 自动安装 PyTorch 2.0.1
- 安装 TensorDict 和 TorchRL
- 配置 Isaac Sim 相关依赖

---

## 🔧 脚本修改详情

### 1. docker_run.sh 修改

**主要变化**:
```bash
# 旧版本
--volume $(pwd)/isaac-training:/workspace/NavRL-main/isaac-training

# 新版本  
--volume $(pwd)/isaac-training:/workspace/isaac-training
```

**新增环境变量**:
```bash
--env ENV_NAME=NavRL
```

**工作目录调整**:
```bash
--workdir /workspace/isaac-training/training/scripts
```

### 2. docker_train.sh 修改

**训练命令**:
```bash
# 旧版本
python train_manipulator.py

# 新版本
bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate NavRL && python train_manipulator.py"
```

**卷挂载调整**:
```bash
# 旧版本：使用 NavRL-main 前缀
--volume $(pwd)/isaac-training:/workspace/NavRL-main/isaac-training

# 新版本：直接挂载到 workspace
--volume $(pwd)/isaac-training:/workspace/isaac-training
--volume $(pwd)/quick-demos:/workspace/quick-demos
```

### 3. docker-compose.yml 修改

**环境变量**:
```yaml
environment:
  - ENV_NAME=NavRL  # 新增
```

**卷挂载**:
```yaml
volumes:
  - ./isaac-training:/workspace/isaac-training
  - ./quick-demos:/workspace/quick-demos
  - ./checkpoints:/workspace/checkpoints
  - ./logs:/workspace/logs
```

---

## 🚀 使用方法

### 方法 1: 使用脚本

```bash
# 构建镜像
./docker_build.sh

# 交互式运行
./docker_run.sh

# 后台训练
./docker_train.sh
```

### 方法 2: 使用 docker-compose

```bash
# 构建并启动
docker-compose up --build

# 后台运行
docker-compose up -d

# 进入容器
docker-compose exec navrl-training bash
```

---

## 📁 目录结构

### 容器内目录结构
```
/workspace/
├── isaac-training/          # 挂载宿主机 isaac-training/
│   └── training/
│       └── scripts/
│           ├── train_manipulator.py
│           ├── manipulator_env.py
│           └── ppo_manipulator.py
├── quick-demos/             # 挂载宿主机 quick-demos/
├── checkpoints/             # 挂载到宿主机 checkpoints/
└── logs/                    # 挂载到宿主机 logs/
```

### 宿主机目录结构
```
NavRL-main/
├── isaac-training/          # 训练代码和配置
├── quick-demos/            # 演示代码
├── checkpoints/            # 训练检查点
├── logs/                   # 训练日志
└── ...
```

---

## ⚙️ 环境激活

容器启动后，conda 环境会自动激活：

```bash
# 容器内会自动执行
source /opt/conda/etc/profile.d/conda.sh
conda activate NavRL
```

### 验证环境
```bash
# 进入容器后验证
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import tensordict; print('TensorDict:', tensordict.__version__)"
python -c "import torchrl; print('TorchRL:', torchrl.__version__)"
```

---

## 🔍 调试和监控

### 查看容器状态
```bash
docker ps
docker logs navrl_training
```

### 进入运行中的容器
```bash
# 使用脚本
docker exec -it navrl_training bash

# 使用 docker-compose
docker-compose exec navrl-training bash
```

### 检查 conda 环境
```bash
# 在容器内
conda info
conda list
```

---

## 🐛 常见问题

### 1. 环境未激活
**症状**: `conda: command not found`

**解决**: 容器会自动激活环境，如果手动进入需要：
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate NavRL
```

### 2. 路径问题
**症状**: 找不到训练脚本

**解决**: 确保工作目录正确：
```bash
cd /workspace/isaac-training/training/scripts
ls -la  # 应该看到 train_manipulator.py
```

### 3. 权限问题
**症状**: 无法写入 checkpoints 或 logs

**解决**: 检查宿主机目录权限：
```bash
chmod 755 checkpoints logs
```

---

## 📊 性能优化

### 1. 共享内存
```bash
--shm-size 8g  # 根据系统内存调整
```

### 2. GPU 内存
```bash
# 监控 GPU 使用
nvidia-smi
```

### 3. 磁盘空间
```bash
# 检查可用空间
df -h
```

---

## ✅ 验证清单

启动前确认：

- [ ] Docker 和 NVIDIA Container Toolkit 已安装
- [ ] GPU 驱动版本 >= 525
- [ ] 磁盘空间 >= 100GB
- [ ] 内存 >= 32GB
- [ ] 网络连接正常

启动后验证：

- [ ] 容器成功启动
- [ ] conda 环境已激活
- [ ] PyTorch 可以导入
- [ ] GPU 可用 (`torch.cuda.is_available()`)
- [ ] 训练脚本可以运行

---

## 🎯 下一步

1. **构建镜像**: `./docker_build.sh`
2. **测试环境**: `./docker_run.sh`
3. **开始训练**: `./docker_train.sh`
4. **监控进度**: `docker logs -f navrl_training_bg`

---

**修改完成！现在可以使用新的 Dockerfile 进行训练了。** 🚀
