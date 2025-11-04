# GPU 设备配置指南（服务器多GPU训练）

## 快速配置

在服务器上训练时，如果有多块GPU，可以通过修改 `config.yaml` 来指定使用的GPU设备号。

### 方法1：修改 config.yaml（推荐）

编辑 `config.yaml` 文件，修改 `env.gpu_device_id` 参数：

```yaml
env:
  gpu_device_id: 1  # 使用GPU 1（可以是 0, 1, 2, ...）
```

**示例：**
- `gpu_device_id: 0` → 使用GPU 0
- `gpu_device_id: 1` → 使用GPU 1
- `gpu_device_id: 2` → 使用GPU 2

### 方法2：命令行设置环境变量（临时）

如果你想在运行时临时指定GPU，可以使用环境变量：

```bash
# 使用GPU 1
CUDA_VISIBLE_DEVICES=1 python train.py

# 使用GPU 2和3（如果支持多GPU训练）
CUDA_VISIBLE_DEVICES=2,3 python train.py
```

**注意：** 使用 `CUDA_VISIBLE_DEVICES` 时，PyTorch会将可见的GPU重新编号为 0, 1, ...，所以 `config.yaml` 中的 `gpu_device_id` 应该设置为 `0`。

## 工作原理

代码会自动：

1. **读取配置**：从 `config.yaml` 中读取 `env.gpu_device_id`
2. **验证设备**：检查指定的GPU是否可用
3. **设置PyTorch设备**：自动设置为 `cuda:{gpu_device_id}`
4. **设置Isaac Gym设备**：计算设备和图形设备都使用相同的GPU ID
5. **错误处理**：如果指定的GPU不存在，会自动回退到GPU 0并给出警告

## 验证GPU设备

训练开始时会打印设备信息：

```
[Env] GPU设备配置: PyTorch=cuda:1, Isaac Gym计算设备=1, Isaac Gym图形设备=1
```

你也可以在Python中检查：

```python
import torch
print(f"可用GPU数量: {torch.cuda.device_count()}")
print(f"当前GPU: {torch.cuda.current_device()}")
print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

## 多GPU训练（未来扩展）

当前版本使用单GPU训练。如果需要多GPU并行训练，需要额外的修改：
- 使用 `torch.nn.DataParallel` 或 `torch.nn.parallel.DistributedDataParallel`
- 修改 `num_envs` 以充分利用多GPU资源

## 故障排除

### 问题1：GPU设备不可用

**错误信息：**
```
[Warning] GPU 2 not available, only 2 GPUs found. Using GPU 0.
```

**解决方案：**
- 检查GPU数量：`nvidia-smi`
- 修改 `config.yaml` 中的 `gpu_device_id` 为可用值（0 到 N-1）

### 问题2：GPU内存不足

**解决方案：**
- 减少 `num_envs`（并行环境数量）
- 减少 `voxel.size`（体素网格大小）
- 减少 `ppo.batch_size` 和 `ppo.rollout_steps`

### 问题3：Isaac Gym创建失败

**解决方案：**
- 确保GPU支持图形渲染（某些服务器GPU可能不支持）
- 检查CUDA驱动和Isaac Gym版本兼容性
- 如果不需要可视化，确保 `enable_viewer: false`

## 配置示例

### 单GPU服务器（默认）
```yaml
env:
  gpu_device_id: 0
```

### 多GPU服务器（使用GPU 1）
```yaml
env:
  gpu_device_id: 1
```

### CPU模式（调试用）
```yaml
env:
  gpu_device_id: -1  # 使用CPU（不推荐，速度很慢）
```

## 相关文件

- `config.yaml` - GPU设备配置
- `env.py` - 设备初始化和Isaac Gym设置
- `train.py` - 训练主循环（使用 `env.device`）
- `ppo.py` - PPO训练器（接收 `device` 参数）


