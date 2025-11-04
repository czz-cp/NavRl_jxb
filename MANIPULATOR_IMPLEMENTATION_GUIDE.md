# UR10e机械臂导航系统实施指南
# Manipulator Navigation Implementation Guide

## 📋 概述

将NavRL从无人机导航改造为UR10e机械臂路径规划系统。

**功能特性**:
- ✅ 未知环境的实时路径规划
- ✅ 基于深度图的障碍物感知
- ✅ 动态障碍物避让（人体协作）
- ✅ ArUco码目标跟踪
- ✅ 末端6D速度控制
- ✅ RL策略 + 安全保护

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    硬件层                                 │
├─────────────────────────────────────────────────────────┤
│  UR10e + Realsense D435i (眼在手上)                      │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│                   感知层                                  │
├─────────────────────────────────────────────────────────┤
│  • occupancyMap: 静态环境建模                            │
│  • dynamicDetector: 动态障碍物检测                       │
│  • ArUco检测: 目标位置获取                               │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│                   决策层                                  │
├─────────────────────────────────────────────────────────┤
│  • NavRL Policy: 学习的路径规划                          │
│  • SafeAction (可选): ORCA避碰                           │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│                   控制层                                  │
├─────────────────────────────────────────────────────────┤
│  • UR10e末端速度控制器                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 文件结构

### **新增文件**

```
isaac-training/training/
├── scripts/
│   ├── manipulator_env.py          ⭐ 训练环境
│   ├── ppo_manipulator.py          ⭐ PPO网络
│   └── train_manipulator.py        ⭐ 训练脚本
└── cfg/
    └── train_manipulator.yaml      ⭐ 训练配置

ros1/navigation_runner/
├── scripts/
│   ├── manipulator_navigation.py   ⭐ 导航节点
│   └── aruco_target_provider.py    ⭐ ArUco提供者
├── cfg/
│   └── manipulator_nav.yaml        ⭐ 导航配置
└── launch/
    └── manipulator_navigation.launch ⭐ 启动文件

ros1/map_manager/cfg/
└── manipulator_occupancy_map.yaml  ⭐ 地图配置

ros1/onboard_detector/cfg/
└── manipulator_detector_param.yaml ⭐ 检测器配置
```

### **复用文件（需要复制）**

```bash
# 从无人机版本复制
cp ros1/navigation_runner/scripts/ppo.py \
   ros1/navigation_runner/scripts/ppo.py
   
cp ros1/navigation_runner/scripts/utils.py \
   ros1/navigation_runner/scripts/utils.py
```

---

## 🚀 实施步骤

### **阶段1: 环境准备 (1天)**

#### **1.1 安装依赖**

```bash
# Isaac Sim (已安装)
# 确保可以运行无人机训练

# ROS1 Noetic
sudo apt install ros-noetic-desktop-full

# UR驱动（如果还没有）
sudo apt install ros-noetic-ur-robot-driver

# 其他依赖
pip install torch torchrl tensordict wandb omegaconf hydra-core
```

#### **1.2 验证现有系统**

```bash
# 测试ArUco检测
roslaunch visual_servo ur_visual_servo.launch

# 检查TF树
rosrun tf tf_echo base_link aruco_marker_0

# 应该能看到ArUco的位姿
```

---

### **阶段2: 训练准备 (2-3天)**

#### **2.1 修改Isaac Sim环境**

```bash
cd isaac-training/training/scripts/

# 检查新文件
ls -la manipulator_env.py ppo_manipulator.py train_manipulator.py

# 测试环境加载
python
>>> from manipulator_env import ManipulatorNavigationEnv
>>> # 如果有错误，修复导入问题
```

#### **2.2 准备UR10e模型**

需要在Isaac Sim中添加UR10e USD文件：

```python
# 选项A: 使用Isaac自带的UR10
from omni.isaac.orbit_assets.ur10 import UR10_CFG

# 选项B: 导入自定义URDF
# 1. 准备UR10e的URDF
# 2. 使用Isaac Sim的URDF导入器转换为USD
# 3. 在环境中引用
```

#### **2.3 首次训练测试**

```bash
cd isaac-training/training/scripts/

# 简化测试：无障碍物
# 修改 train_manipulator.yaml:
# env.num_static_obstacles: 0
# env.num_dynamic_obstacles: 0

# 启动训练
python train_manipulator.py

# 观察Wandb:
# - 确保能收敛到目标
# - 成功率 > 90%
```

---

### **阶段3: 完整训练 (1-2周)**

#### **3.1 课程学习训练**

```bash
# 第1阶段: 无障碍物 (500万帧)
# num_static_obstacles: 0
# num_dynamic_obstacles: 0
python train_manipulator.py

# 第2阶段: 静态障碍物 (1500万帧)
# num_static_obstacles: 3
# num_dynamic_obstacles: 0
# 加载第1阶段检查点
python train_manipulator.py wandb.run_id=<stage1_run_id>

# 第3阶段: 动态障碍物 (2000万帧)
# num_static_obstacles: 5
# num_dynamic_obstacles: 2
python train_manipulator.py wandb.run_id=<stage2_run_id>
```

#### **3.2 训练监控**

```python
# Wandb指标
重点关注:
- train/stats.reach_goal > 0.8  # 成功率
- train/stats.collision < 0.1   # 碰撞率
- train/stats.return > 50       # 累积奖励
- actor_loss, critic_loss       # 损失收敛

调优:
- 如果碰撞率高 → 增加 reward_safety 权重
- 如果不收敛 → 降低 learning_rate
- 如果路径抖动 → 增加 penalty_smooth 权重
```

---

### **阶段4: ROS集成 (3-5天)**

#### **4.1 准备ROS工作空间**

```bash
cd ~/catkin_ws/src/
ln -s /home/zar/Downloads/NavRL-main/ros1/* .

cd ~/catkin_ws
catkin_make

source devel/setup.bash
```

#### **4.2 修改你的UR启动文件**

在 `visual_servo/launch/UR_robot.launch` 中添加：

```xml
<!-- 发布末端位姿为话题（给occupancyMap用） -->
<node pkg="tf2_ros" type="static_transform_publisher" 
      name="ee_pose_publisher">
  <!-- 或者写一个节点从TF转换 -->
</node>

<!-- 确保相机TF正确 -->
<!-- base_link -> ee_link -> camera_link -> camera_depth_optical_frame -->
```

#### **4.3 测试各个模块**

```bash
# 测试1: occupancyMap
roslaunch map_manager occupancy_map.launch \
  config:=manipulator_occupancy_map.yaml

# 检查话题
rostopic echo /occupancy_map/voxel_map

# 测试2: dynamicDetector
roslaunch onboard_detector detector.launch \
  config:=manipulator_detector_param.yaml

# 检查服务
rosservice call /onboard_detector/get_dynamic_obstacles

# 测试3: ArUco Provider
rosrun navigation_runner aruco_target_provider.py

# 检查话题
rostopic echo /rl_navigation/aruco_target
```

---

### **阶段5: 低速仿真测试 (2-3天)**

#### **5.1 Gazebo仿真测试**

```bash
# 如果有Gazebo模型
roslaunch ur_gazebo ur10e_bringup.launch

# 启动导航系统
roslaunch navigation_runner manipulator_navigation.launch

# 在RViz中设置目标
# 观察机械臂运动
```

#### **5.2 调试清单**

```
□ TF树正确: base_link -> ee_link -> camera_link -> aruco_marker
□ 深度图可视化正常
□ ArUco检测稳定
□ occupancyMap构建地图
□ dynamicDetector检测到人（如果有）
□ RL策略输出合理速度
□ 机械臂响应速度命令
```

---

### **阶段6: 实机部署 (1-2周)**

#### **6.1 安全准备**

```bash
# 1. 降低速度限制
修改 manipulator_nav.yaml:
  max_linear_vel: 0.1   # 从0.3降到0.1
  max_angular_vel: 0.2  # 从0.5降到0.2

# 2. 启用紧急停止
# 主控制台随时可以按Enter停止

# 3. 测试急停
roslaunch navigation_runner manipulator_navigation.launch
# 按Enter → 机械臂应该立即停止
```

#### **6.2 实机测试流程**

```bash
# Step 1: 启动机械臂
roslaunch visual_servo UR_robot.launch

# Step 2: 启动ArUco检测
rosrun visual_servo aruco_detector.py

# Step 3: 放置ArUco码
# 在机械臂工作空间内，距离30-80cm

# Step 4: 检查ArUco检测
rostopic echo /rl_navigation/aruco_target
# 应该能看到目标位姿

# Step 5: 启动导航系统
roslaunch navigation_runner manipulator_navigation.launch

# Step 6: 观察
# - RViz中应该看到:
#   - 绿色ArUco目标
#   - 红色RL速度箭头
#   - 绿色安全速度箭头
#   - 蓝色动态障碍物（如果有人）

# Step 7: 监控
# 终端应该输出:
# [Manipulator Nav]: Moving to target, distance: 0.45m
# [Manipulator Nav]: Velocity: [0.12, 0.05, -0.03, ...]

# Step 8: 到达目标
# distance < 5cm → 停止
```

#### **6.3 故障排除**

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| TF not found | TF树不完整 | 检查 `rosrun tf view_frames` |
| 深度图为空 | 相机未启动 | 检查 `rostopic list | grep depth` |
| ArUco未检测 | 光照/距离 | 调整位置，检查 `image_view` |
| 机械臂不动 | 控制器未启动 | 检查 `rostopic info /ur10e_velocity_controller/command` |
| 碰撞 | 地图未更新 | 检查 `rostopic hz /occupancy_map/voxel_map` |

---

## 🎯 关键修改点总结

### **训练环境 (manipulator_env.py)**

```python
关键修改:
1. 机器人: drone → UR10e manipulator
2. 观测: lidar (36×4) → depth (80×60)
3. 动作: 3D速度 → 6D末端速度
4. 奖励: 飞行奖励 → 到达+安全奖励
5. 终止: 碰撞+越界 → 碰撞+关节限位+工作空间
```

### **RL网络 (ppo_manipulator.py)**

```python
关键修改:
1. CNN输入: lidar → depth图
2. 新增: 关节位置编码器
3. 动作维度: 3 → 6
4. 动作缩放: 线速度±0.3m/s, 角速度±0.5rad/s
5. 保留: 动态障碍物处理（与无人机相同）
```

### **ROS导航 (manipulator_navigation.py)**

```python
关键修改:
1. 位姿源: IMU → TF (base_link → ee_link)
2. 目标源: 手动设置 → ArUco检测
3. 控制输出: 无人机速度 → 末端速度
4. 坐标系: 世界坐标 → 机械臂基座坐标
5. 保留: 深度感知、动态避障逻辑
```

### **占据地图 (manipulator_occupancy_map.yaml)**

```yaml
关键修改:
1. 相机内参: 无人机相机 → Realsense D435i
2. 地图范围: 20×20×4.5m → 2×2×1.5m
3. 分辨率: 0.1m → 0.02m (更精细)
4. 机器人尺寸: 0.5×0.5×0.3m → 0.15×0.15×0.2m
5. 坐标变换: body_to_camera更新
```

---

## 🔧 调试技巧

### **Sim调试**

```bash
# 可视化训练过程
python train_manipulator.py headless=false

# 单步调试
from manipulator_env import ManipulatorNavigationEnv
env = ManipulatorNavigationEnv(cfg)
obs = env.reset()
action = policy(obs)
next_obs = env.step(action)
```

### **ROS调试**

```bash
# 检查所有话题
rostopic list

# 检查TF
rosrun tf view_frames
evince frames.pdf

# 检查服务
rosservice list | grep -E "raycast|dynamic|safe"

# 实时监控
rqt_graph  # 查看节点连接
rqt_tf_tree  # 查看TF树
```

### **RViz配置**

```
添加显示:
- TF: 显示坐标系
- Marker: /rl_navigation/aruco_target
- MarkerArray: /onboard_detector/dynamic_bboxes
- PointCloud2: /occupancy_map/voxel_map
- PointCloud2: /camera/depth/color/points
- RobotModel: UR10e
```

---

## ⚙️ 参数调优指南

### **训练参数**

```yaml
# 如果训练不收敛:
algo.actor.learning_rate: 0.0001  # 降低学习率

# 如果碰撞率高:
# 增加安全奖励权重 (在env中)
reward = ... + reward_safety * 1.0  # 从0.5改为1.0

# 如果路径抖动:
# 增加平滑惩罚
penalty_smooth_weight: 0.2  # 从0.1改为0.2
```

### **控制参数**

```yaml
# 安全保守设置（初期）
control:
  max_linear_vel: 0.1
  max_angular_vel: 0.2
  min_obstacle_distance: 0.20

# 性能优化设置（后期）
control:
  max_linear_vel: 0.3
  max_angular_vel: 0.5
  min_obstacle_distance: 0.15
```

---

## 🎯 预期性能

### **训练指标**

| 阶段 | 训练帧数 | 成功率 | 碰撞率 | 平均步数 |
|-----|---------|-------|-------|---------|
| 无障碍 | 5M | >95% | <2% | 80 |
| 静态障碍 | 15M | >85% | <8% | 120 |
| 动态障碍 | 25M | >75% | <12% | 150 |

### **实机性能**

```
平均规划时间: < 50ms (20Hz)
平均到达时间: 8-15秒（取决于距离和障碍物）
成功率: > 80% (在训练相似的环境)
```

---

## 📊 数据流图

```
硬件传感器
    │
    ├─ Realsense → /camera/depth/image_raw
    ├─ Realsense → /camera/color/image_raw
    ├─ UR10e     → /joint_states
    └─ ArUco     → TF: aruco_marker
         │
         ▼
ArUco Provider
    │
    └─ /rl_navigation/aruco_target ──┐
                                      │
occupancyMap                          │
    │                                 │
    ├─ raycast服务 → depth_points     │
    └─ 标记动态自由区域               │
                                      │
dynamicDetector                       │
    │                                 │
    └─ get_dynamic_obstacles服务 ─────┤
                                      │
                                      ▼
                          manipulator_navigation
                                      │
                          ├─ build_observation
                          ├─ policy(obs) → ee_vel_rl
                          ├─ safety_filter → ee_vel_safe
                          └─ publish → /ur10e_velocity_controller/command
                                      │
                                      ▼
                                    UR10e
```

---

## 🎓 重要提示

### **安全第一**

```
⚠️ 实机测试前必做:
1. 设置工作空间限制（软件+物理）
2. 启用急停按钮（硬件）
3. 低速测试（0.1 m/s）
4. 人员保持安全距离
5. 随时准备手动急停
```

### **Sim-to-Real Gap**

```
需要注意:
1. 深度噪声: 仿真完美，实际有噪声
   → 训练时添加噪声
   
2. 动力学差异: 仿真响应快，实际有延迟
   → 添加执行延迟模拟
   
3. 视觉差异: 仿真光照理想，实际复杂
   → 域随机化
```

### **调试优先级**

```
1. 先保证无障碍物能到达 ✓
2. 再添加静态障碍物 ✓
3. 最后添加动态障碍物 ✓
4. 每步验证后再继续
```

---

## 📞 需要帮助？

如果遇到问题，检查：
1. ☑️ TF树是否完整
2. ☑️ 相机数据是否发布
3. ☑️ ArUco是否稳定检测
4. ☑️ occupancyMap是否更新
5. ☑️ RL策略是否加载

---

## 🚀 快速启动命令

```bash
# 完整系统启动
roslaunch navigation_runner manipulator_navigation.launch

# 检查状态
rostopic hz /rl_navigation/aruco_target  # ArUco检测
rostopic hz /occupancy_map/voxel_map     # 地图更新
rosservice list | grep dynamic           # 动态检测

# 开始导航
# 机械臂会自动向ArUco移动！
```

Good luck! 🎉

