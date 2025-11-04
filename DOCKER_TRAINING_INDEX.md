# 📚 Docker 训练文档索引

NavRL Manipulator 的完整 Docker 训练文档。

---

## 🚀 快速开始

从这里开始：

### [QUICK_START_DOCKER.md](QUICK_START_DOCKER.md)
**3 步开始训练，5 分钟上手**

适合：已经安装好 Docker 和 NVIDIA 环境，想要快速开始训练。

---

## 📖 详细文档

### 1. [DOCKER_README.md](DOCKER_README.md)
**快速参考和常用命令**

内容：
- 三步开始训练
- 文件说明
- 常用命令
- 配置调整
- 常见问题

适合：快速查找命令和解决方案。

### 2. [DOCKER_DEPLOYMENT_GUIDE.md](DOCKER_DEPLOYMENT_GUIDE.md)
**完整部署指南（70+ 页）**

内容：
- 环境要求详解
- Docker 和 NVIDIA Container Toolkit 安装
- 镜像构建和运行
- 训练配置和优化
- 监控和管理
- 故障排除（10+ 个常见问题）
- 高级用法

适合：首次部署或遇到问题时查阅。

---

## 🛠️ 脚本和工具

### Docker 脚本

| 脚本 | 功能 | 使用场景 |
|------|------|---------|
| `verify_docker_setup.sh` | 验证环境配置 | 部署前检查 |
| `docker_build.sh` | 构建训练镜像 | 首次使用或更新代码 |
| `docker_run.sh` | 交互式运行容器 | 调试和测试 |
| `docker_train.sh` | 后台运行训练 | 正式训练 |

### Docker 配置

| 文件 | 说明 |
|------|------|
| `Dockerfile` | 镜像定义（基于 Isaac Sim） |
| `docker-compose.yml` | Docker Compose 配置 |
| `.dockerignore` | 构建时忽略的文件 |

---

## 📊 训练相关

### 训练脚本

位置：`isaac-training/training/scripts/`

| 文件 | 功能 |
|------|------|
| `train_manipulator.py` | 主训练脚本 |
| `manipulator_env.py` | Isaac Sim 环境定义 |
| `ppo_manipulator.py` | PPO 策略网络 |
| `utils.py` | 工具函数 |

### 配置文件

位置：`isaac-training/training/cfg/`

| 文件 | 配置内容 |
|------|---------|
| `train_manipulator.yaml` | 训练参数、环境设置、PPO 超参数 |
| `drone.yaml` | （无人机版本，参考） |
| `ppo.yaml` | （无人机版本，参考） |

---

## 📚 系统文档

### [MANIPULATOR_IMPLEMENTATION_GUIDE.md](MANIPULATOR_IMPLEMENTATION_GUIDE.md)
**完整实现指南**

内容：
- 系统架构
- Isaac Sim 训练环境
- ROS1 部署
- 关键模块详解

### [MANIPULATOR_SYSTEM_OVERVIEW.md](MANIPULATOR_SYSTEM_OVERVIEW.md)
**系统概述**

内容：
- 整体架构
- 数据流
- 模块交互

### [BALLOON_OBSTACLES_GUIDE.md](BALLOON_OBSTACLES_GUIDE.md)
**气球障碍物详解**

内容：
- 气球特性
- 检测配置
- 避障策略

---

## 🎯 使用流程

### 新手入门

```
验证环境 → 构建镜像 → 开始训练 → 监控进度
    ↓           ↓          ↓          ↓
verify      docker_    docker_    docker logs
_setup.sh   build.sh   train.sh   -f ...
```

**推荐阅读顺序**:
1. `QUICK_START_DOCKER.md` - 快速开始
2. `DOCKER_README.md` - 常用命令
3. `DOCKER_DEPLOYMENT_GUIDE.md` - 遇到问题时查阅

### 进阶使用

**调整训练参数**:
1. 编辑 `isaac-training/training/cfg/train_manipulator.yaml`
2. 重新构建镜像：`./docker_build.sh`
3. 开始训练：`./docker_train.sh`

**自定义环境**:
1. 修改 `isaac-training/training/scripts/manipulator_env.py`
2. 修改 `isaac-training/training/scripts/ppo_manipulator.py`
3. 重新构建和训练

**部署到实机**:
1. 训练完成，获得检查点
2. 参考 `MANIPULATOR_IMPLEMENTATION_GUIDE.md` 的 ROS1 部署部分
3. 使用 `ros1/navigation_runner/scripts/manipulator_navigation.py`

---

## 🆘 获取帮助

### 常见问题

查看 [`DOCKER_DEPLOYMENT_GUIDE.md`](DOCKER_DEPLOYMENT_GUIDE.md) 的"故障排除"章节。

### 问题分类

| 问题类型 | 查看文档 |
|---------|---------|
| 环境安装 | `DOCKER_DEPLOYMENT_GUIDE.md` → 步骤 1 |
| 镜像构建 | `DOCKER_DEPLOYMENT_GUIDE.md` → 故障排除 → 问题 1 |
| GPU 问题 | `DOCKER_DEPLOYMENT_GUIDE.md` → 故障排除 → 问题 2 |
| 内存不足 | `DOCKER_DEPLOYMENT_GUIDE.md` → 故障排除 → 问题 3 |
| 训练速度 | `DOCKER_DEPLOYMENT_GUIDE.md` → 故障排除 → 问题 5 |
| 参数调整 | `DOCKER_README.md` → 配置调整 |

### 快速诊断

```bash
# 运行诊断脚本
./verify_docker_setup.sh

# 查看详细错误
docker logs navrl_training_bg

# 检查 GPU
nvidia-smi
```

---

## 📞 技术栈

- **仿真**: NVIDIA Isaac Sim 4.0
- **RL 框架**: PyTorch + TorchRL
- **容器**: Docker + NVIDIA Container Toolkit
- **监控**: Weights & Biases / TensorBoard
- **部署**: ROS1 (Noetic)

---

## ✅ 检查清单

### 训练前

- [ ] 阅读 `QUICK_START_DOCKER.md`
- [ ] 运行 `./verify_docker_setup.sh`
- [ ] 确认所有检查通过
- [ ] 设置 `WANDB_API_KEY` (可选)

### 训练中

- [ ] 监控 GPU 利用率 (应在 80%+)
- [ ] 定期查看日志
- [ ] 检查 checkpoint 是否正常保存

### 训练后

- [ ] 验证最终模型
- [ ] 备份 checkpoints
- [ ] 准备部署到实机

---

## 🎓 学习路径

### 初级 (0-1 天)

1. ✅ 验证环境
2. ✅ 构建镜像
3. ✅ 开始训练
4. ✅ 监控基础指标

**文档**: `QUICK_START_DOCKER.md`, `DOCKER_README.md`

### 中级 (1-3 天)

1. ⚙️ 调整训练参数
2. 📊 理解奖励函数
3. 🔍 分析训练曲线
4. 🛠️ 优化性能

**文档**: `DOCKER_DEPLOYMENT_GUIDE.md`, `train_manipulator.yaml`

### 高级 (3+ 天)

1. 🏗️ 修改环境结构
2. 🧠 调整策略网络
3. 📈 实验不同算法
4. 🚀 部署到实机

**文档**: `MANIPULATOR_IMPLEMENTATION_GUIDE.md`, 源代码

---

## 📈 预期结果

训练收敛后（约 3000-5000 万帧），应该看到：

- **Success Rate**: > 0.8
- **Collision Rate**: < 0.1
- **Mean Reward**: > 50
- **Episode Length**: < 500 步

---

## 🔗 快速链接

| 想要... | 查看... |
|--------|--------|
| 🏃 立即开始训练 | [`QUICK_START_DOCKER.md`](QUICK_START_DOCKER.md) |
| 📖 查找命令 | [`DOCKER_README.md`](DOCKER_README.md) |
| 🔧 解决问题 | [`DOCKER_DEPLOYMENT_GUIDE.md`](DOCKER_DEPLOYMENT_GUIDE.md) → 故障排除 |
| ⚙️ 调整参数 | `isaac-training/training/cfg/train_manipulator.yaml` |
| 🏗️ 理解架构 | [`MANIPULATOR_SYSTEM_OVERVIEW.md`](MANIPULATOR_SYSTEM_OVERVIEW.md) |
| 🚀 部署实机 | [`MANIPULATOR_IMPLEMENTATION_GUIDE.md`](MANIPULATOR_IMPLEMENTATION_GUIDE.md) |

---

**准备好了？开始训练！** 🚀

```bash
./verify_docker_setup.sh  # ← 从这里开始
```

