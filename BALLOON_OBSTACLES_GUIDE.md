# 气球动态障碍物说明
# Balloon Dynamic Obstacles Guide

## 🎈 概述

已将动态障碍物从**人体**改为**气球**，更适合机械臂工作场景中的轻量级动态避障测试。

---

## 🔄 主要变化对比

| 特性 | 人体障碍物 (原版) | 气球障碍物 (新版) |
|-----|-----------------|-----------------|
| **形状** | 圆柱体 | 球体 |
| **尺寸** | 0.5×0.5×1.7m | 0.16-0.24m直径 |
| **运动空间** | 2D地面 (XY平面) | 3D空间 (XYZ) |
| **速度范围** | 0.3-1.0 m/s | 0.05-0.25 m/s |
| **高度** | 固定0.85m | 0.3-1.1m漂浮 |
| **运动模式** | 目标导向行走 | 随机漂浮+气流扰动 |
| **碰撞阈值** | 0.3m | 0.15m |
| **颜色** | 青色 | 5种彩色（红/蓝/黄/绿/粉） |

---

## 🎯 气球特性详解

### **1. 物理模型**

```python
# 气球尺寸
balloon_radius = 0.10 + random(-0.02, 0.02)  # 8-12cm半径
diameter = 0.16-0.24m  # 直径

# 形状
shape = Sphere  # 完美球体

# 颜色（循环5种）
colors = [红色, 蓝色, 黄色, 绿色, 粉色]
```

### **2. 运动行为**

#### **3D随机漂浮**
```python
# 目标采样（3D空间）
goal_x = origin_x + random(-0.4, 0.4)
goal_y = origin_y + random(-0.4, 0.4)
goal_z = origin_z + random(-0.3, 0.3)  # 新增Z轴

# 工作空间限制
x: [-0.7, 0.7]
y: [-0.7, 0.7]
z: [0.3, 1.1]  # 漂浮高度范围
```

#### **速度模型**
```python
# 基础速度（朝向目标）
base_velocity = norm(0.05-0.25) * direction

# 气流扰动（随机漂移）
random_drift = random(-0.15, 0.15) in each axis

# 最终速度
velocity = base_velocity + random_drift

# 更新频率
update_interval = 3.0 seconds  # 比人更慢
```

#### **运动特点**
- ✅ 缓慢漂浮（平均速度0.15 m/s）
- ✅ 随机扰动（模拟气流）
- ✅ 3D全方向移动
- ✅ 平滑轨迹（无突变）
- ✅ 到达阈值0.2m（比人更精确）

---

## 📊 训练配置调整

### **环境配置**

```yaml
# train_manipulator.yaml

env:
  num_dynamic_obstacles: 3  # 增加到3个气球（原2个人）

env_dyn:
  vel_range: [0.05, 0.25]     # 气球速度（原0.3-1.0）
  local_range: [0.4, 0.4, 0.3]  # 3D漂浮范围（原[0.5, 0.5, 0.0]）
```

### **碰撞检测调整**

```python
# manipulator_env.py

# 碰撞距离阈值
balloon_radius = 0.1m
robot_radius = 0.05m
collision_threshold = 0.15m  # (原0.3m)

# 检测逻辑
collision = (distance < 0.15).any()
```

---

## 🔧 ROS部署配置

### **检测器参数**

```yaml
# manipulator_detector_param.yaml

# 尺寸约束
target_object_size: [0.2, 0.2, 0.2]  # 球形20cm（原[0.5, 0.5, 1.7]）

# 速度阈值
dynamic_velocity_threshold: 0.05  # 气球更慢（原0.2）

# 卡尔曼滤波
e_q_pos: 0.2  # 位置噪声更小（原0.3）
e_q_vel: 0.2  # 速度噪声更小（原0.3）
kalman_filter_averaging_frames: 10  # 更多平均（原8）
```

---

## 🎮 训练策略

### **课程学习建议**

```bash
# 阶段1: 静态气球（验证碰撞检测）
python train_manipulator.py \
    env.num_static_obstacles=3 \
    env.num_dynamic_obstacles=0

# 阶段2: 1个气球（学习基本避障）
python train_manipulator.py \
    env.num_static_obstacles=3 \
    env.num_dynamic_obstacles=1 \
    env_dyn.vel_range=[0.05,0.15]  # 先慢速

# 阶段3: 3个气球（完整挑战）
python train_manipulator.py \
    env.num_static_obstacles=5 \
    env.num_dynamic_obstacles=3 \
    env_dyn.vel_range=[0.05,0.25]
```

### **奖励权重调优**

```python
# 气球场景建议权重
reward_velocity: 1.0        # 主要奖励
reward_safety_static: 0.5   # 静态障碍
reward_safety_dynamic: 0.7  # 动态障碍（提高）
reward_distance: 0.5
penalty_smooth: 0.1

# 在 manipulator_env.py 中
self.reward = (
    reward_vel + 
    1.0 +
    reward_safety_static * 0.5 + 
    reward_safety_dynamic * 0.7 +  # 增加权重
    reward_distance -
    penalty_smooth * 0.1
)
```

---

## 🎯 预期行为

### **训练表现**

| 指标 | 无障碍 | 静态障碍 | 静态+气球 |
|-----|-------|---------|----------|
| **成功率** | >95% | >85% | >75% |
| **碰撞率** | <2% | <8% | <15% |
| **平均步数** | 80 | 120 | 160 |
| **气球碰撞率** | - | - | <5% |

### **可视化效果**

```
Isaac Sim中的场景:
┌────────────────────────────────┐
│  🎈 红气球 (0.5, 0.3, 0.8)     │
│  🎈 蓝气球 (-0.2, 0.5, 0.6)    │
│  🎈 黄气球 (0.3, -0.4, 0.9)    │
│                                 │
│  🦾 UR10e                       │
│                                 │
│  📦 静态障碍物                   │
│  🎯 ArUco目标                   │
└────────────────────────────────┘

气球缓慢漂浮，机械臂平滑避让
```

---

## 🔍 调试技巧

### **可视化气球轨迹**

```python
# 在 manipulator_env.py 中添加
def visualize_balloon_trajectory(self):
    """可视化气球运动轨迹"""
    for i in range(self.cfg.env.num_dynamic_obstacles):
        pos = self.dyn_obs_state[i, :3].cpu().numpy()
        vel = self.dyn_obs_vel[i].cpu().numpy()
        
        print(f"Balloon {i}: pos={pos}, vel={vel}, speed={np.linalg.norm(vel):.3f}")

# 在训练循环中调用
if step % 100 == 0:
    env.visualize_balloon_trajectory()
```

### **检查气球运动合理性**

```python
# 速度应在合理范围
assert 0.0 <= speed <= 0.35, "气球速度异常"

# 高度应在范围内
assert 0.25 <= z <= 1.15, "气球高度异常"

# 位置应在工作空间内
assert -0.75 <= x <= 0.75, "气球X坐标超界"
assert -0.75 <= y <= 0.75, "气球Y坐标超界"
```

---

## 🚀 实机部署注意

### **真实气球检测**

如果要在实机上使用真实气球：

```yaml
# manipulator_detector_param.yaml

# 调整YOLO检测（如果有训练气球类别）
yolo_overwrite_distance: 2.0

# 或使用颜色检测（HSV空间）
# 在 dynamicDetector.cpp 中添加颜色过滤

# 气球特征:
# - 圆形轮廓
# - 均匀颜色
# - 较高的反射率
```

### **实机参数调优**

```yaml
# 真实气球可能更飘忽
env_dyn:
  vel_range: [0.02, 0.20]  # 实机可能更慢

# 检测器更新频率
detector_rate: 30Hz  # 保持

# 卡尔曼滤波更保守
e_q_pos: 0.15  # 进一步降低
e_r_pos: 0.4   # 提高观测噪声（真实检测不稳定）
```

---

## 📈 性能对比

### **训练时间估计**

```
气球场景 vs 人体场景:

训练难度: 相似（气球更慢但3D空间）
收敛速度: 略快（碰撞阈值更小，反馈更清晰）
GPU时间:
  - 无障碍: 8-12小时
  - 静态障碍: 24-36小时
  - 气球障碍: 40-60小时（比人体少10%）

原因:
  + 气球更慢，轨迹更可预测
  + 球形碰撞检测更简单
  - 3D空间增加复杂度（但影响小）
```

### **实机性能**

```
部署效率: 相同（20Hz控制）
推理延迟: <50ms
避障成功率: 预期75-85%

优势:
  ✅ 气球碰撞安全（软体）
  ✅ 容易获取和部署
  ✅ 适合演示和测试

劣势:
  ⚠️ 实际环境中气流影响大
  ⚠️ 可能需要实机微调
```

---

## 🎓 应用场景

### **适合气球的场景**

1. **实验室测试** ✅
   - 安全无害
   - 易于观察
   - 成本低

2. **算法验证** ✅
   - 轻量级动态障碍
   - 3D空间避障
   - 预测轨迹学习

3. **演示展示** ✅
   - 视觉效果好
   - 交互性强
   - 易于理解

### **不适合气球的场景**

1. **重工业应用** ❌
   - 需要更真实障碍物
   - 建议用人体或其他机械臂

2. **快速运动场景** ❌
   - 气球速度太慢
   - 建议用无人机作为动态障碍

---

## 🔄 切换回人体障碍物

如果需要切换回人体障碍物：

```bash
# 恢复原始配置
git checkout manipulator_env.py
git checkout train_manipulator.yaml
git checkout manipulator_detector_param.yaml

# 或手动修改关键参数:
# 1. SphereCfg → CylinderCfg
# 2. vel_range: [0.3, 1.0]
# 3. local_range: [0.5, 0.5, 0.0]
# 4. target_object_size: [0.5, 0.5, 1.7]
# 5. collision_threshold: 0.3
```

---

## ✅ 快速验证清单

训练前检查：
- [ ] 气球形状为球体（Sphere）
- [ ] 气球数量 = 3
- [ ] 速度范围 = [0.05, 0.25]
- [ ] 3D漂浮范围正确
- [ ] 碰撞阈值 = 0.15m

部署前检查：
- [ ] 检测器尺寸约束 = [0.2, 0.2, 0.2]
- [ ] 速度阈值 = 0.05
- [ ] 卡尔曼参数更新
- [ ] 可视化颜色正确

---

## 🎉 总结

气球动态障碍物提供了：
- ✅ **安全性**: 软体，碰撞无害
- ✅ **可视化**: 彩色，易于区分
- ✅ **3D运动**: 全空间避障挑战
- ✅ **真实性**: 实际可部署（真实气球）
- ✅ **趣味性**: 演示效果好

**开始训练吧！** 🚀

```bash
cd isaac-training/training/scripts/
python train_manipulator.py
```

Good luck! 🎈

