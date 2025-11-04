# UR10e机械臂导航系统完整说明
# Complete System Overview for UR10e Manipulator Navigation

## 🎯 系统目标

**从已知起点A到动态ArUco目标B的无碰撞路径规划**

- **起点A**: 机械臂初始关节配置
- **终点B**: ArUco码位置（实时检测，一轮规划中固定）
- **障碍物**: 静态障碍物（未知） + 动态障碍物（人体协作）
- **控制**: 6D末端速度控制 [vx, vy, vz, wx, wy, wz]
- **方法**: 强化学习 (NavRL) + 动态避障

---

## 📊 完整数据流

```
┌─────────────────────────────────────────────────────────────┐
│                     输入层 (20Hz)                             │
├─────────────────────────────────────────────────────────────┤
│ 1. /joint_states          → 关节位置+速度                    │
│ 2. TF: base→ee_link       → 末端位姿                         │
│ 3. TF: base→aruco_marker  → 目标位姿 (固定一轮)              │
│ 4. /camera/depth/image    → 深度图 (未知障碍物)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  感知处理层 (20-30Hz)                         │
├─────────────────────────────────────────────────────────────┤
│ occupancyMap (20Hz):                                        │
│   深度图 → 3D点云 → 射线投射 → 占据地图                      │
│   • 输入: /camera/depth/image + TF                          │
│   • 输出: 静态障碍物3D地图                                   │
│   • 服务: raycast (模拟Realsense扫描)                       │
│                                                              │
│ dynamicDetector (30Hz):                                     │
│   深度图 → DBSCAN+UV+YOLO → 跟踪 → 分类                     │
│   • 输入: /camera/depth + /camera/color                     │
│   • 输出: 动态障碍物 (位置+速度+尺寸)                        │
│   • 服务: get_dynamic_obstacles                             │
│                                                              │
│ ArUco Provider (20Hz):                                      │
│   TF: aruco_marker → PoseStamped                            │
│   • 输入: TF树                                              │
│   • 输出: /rl_navigation/aruco_target                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  观测构建层 (20Hz)                            │
├─────────────────────────────────────────────────────────────┤
│ manipulator_navigation.py:                                  │
│                                                              │
│ 1. 机械臂状态 (8维):                                         │
│    ├─ 目标方向 (3)                                           │
│    ├─ XY距离 (1)                                             │
│    ├─ Z距离 (1)                                              │
│    └─ 末端速度 (3)                                           │
│                                                              │
│ 2. 深度图 (80×60):                                          │
│    └─ occupancyMap/raycast → 重塑为图像                     │
│                                                              │
│ 3. 关节位置 (6):                                             │
│    └─ /joint_states → [q1, q2, ..., q6]                   │
│                                                              │
│ 4. 动态障碍物 (5×10):                                       │
│    └─ dynamicDetector服务 → 最近5个                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   决策层 (20Hz)                               │
├─────────────────────────────────────────────────────────────┤
│ RL Policy (NavRL):                                          │
│   obs → Feature Extractor → Actor → 6D末端速度              │
│                                                              │
│ 网络结构:                                                    │
│   Depth CNN (80×60 → 128)                                  │
│        ↓                                                     │
│   Joint Encoder (6 → 32)                                   │
│        ↓                                                     │
│   Dyn Obs MLP (5×10 → 64)                                  │
│        ↓                                                     │
│   State (8)                                                 │
│        ↓                                                     │
│   Concat (128+32+64+8=232 → 256)                          │
│        ↓                                                     │
│   Actor (256 → 6) Beta分布                                 │
│        ↓                                                     │
│   动作缩放: [0,1] → 线速度±0.3, 角速度±0.5                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  安全过滤层 (20Hz)                            │
├─────────────────────────────────────────────────────────────┤
│ Safety Filter:                                              │
│   1. 深度检查: min_depth < 15cm → 减速                      │
│   2. 速度限制: 线速度 ≤ 0.3m/s, 角速度 ≤ 0.5rad/s          │
│   3. 接近目标: distance < 20cm → 渐进减速                   │
│   4. 到达停止: distance < 5cm → 停止                        │
│                                                              │
│ 可选: ORCA SafeAction                                       │
│   ee_vel_rl → ORCA平面 → LP求解 → ee_vel_safe              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  执行层 (125Hz)                               │
├─────────────────────────────────────────────────────────────┤
│ UR10e Velocity Controller:                                 │
│   Twist → 末端速度控制                                       │
│   • 话题: /ur10e_velocity_controller/command                │
│   • 频率: 125Hz (UR内部)                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 控制循环详解

### **单个控制周期 (50ms)**

```
T = 0ms
├─ raycast_callback() 异步更新
│  └─ occupancyMap/raycast → self.depth_points
│
├─ dynamic_obstacle_callback() 异步更新
│  └─ dynamicDetector服务 → self.dynamic_obstacles
│
└─ control_callback() 主循环
   │
   ├─ Step 1: 获取状态 (5ms)
   │  ├─ TF: base→ee_link
   │  ├─ TF: base→aruco_marker
   │  └─ /joint_states
   │
   ├─ Step 2: 构建观测 (3ms)
   │  ├─ 计算相对位置/方向
   │  ├─ 转换深度图
   │  └─ 编码动态障碍物
   │
   ├─ Step 3: RL推理 (15ms)
   │  └─ policy(obs) → ee_vel_rl
   │
   ├─ Step 4: 安全过滤 (2ms)
   │  └─ safety_filter(ee_vel_rl) → ee_vel_safe
   │
   └─ Step 5: 发布命令 (1ms)
      └─ publish(ee_vel_safe)

总耗时: ~26ms < 50ms ✓ 有裕度
```

---

## 🎯 关键设计决策

### **1. 为什么用末端速度控制？**

```
优势:
  ✅ 与训练一致（直接学习末端空间）
  ✅ 避免逆运动学求解
  ✅ 更直观的避障行为
  ✅ UR10e原生支持

劣势:
  ⚠️ 可能遇到奇异配置
  ⚠️ 关节限位需要额外处理

解决:
  → 在训练中惩罚奇异配置
  → 安全层检查关节角度
```

### **2. 为什么保留动态避障？**

```
应用场景:
  • 人机协作: 人在工作空间内移动
  • 移动物体: 传送带上的物品
  • 其他机械臂: 双臂协作

价值:
  ✅ 提高安全性
  ✅ 减少停机时间
  ✅ 实现真正的自主性

实现:
  → 复用dynamicDetector (30Hz检测+跟踪)
  → RL策略学习避让策略
  → 可选ORCA实时避碰
```

### **3. 为什么用深度图而不是点云？**

```
深度图 (80×60 = 4800点):
  ✅ 稠密规整
  ✅ CNN天然适合
  ✅ 训练更稳定
  ✅ 推理更快

点云 (随机N点):
  ⚠️ 稀疏不规整
  ⚠️ 需要PointNet
  ⚠️ 训练复杂
  ⚠️ 推理慢

选择: 深度图 ✓
```

---

## 📈 性能优化建议

### **训练加速**

```yaml
# GPU优化
env:
  num_envs: 512  # 更多并行环境（如果GPU够大）

algo:
  training_frame_num: 32  # 更大batch（从16增加）
  
# 如果显存不够:
env:
  num_envs: 128  # 减少环境
sensor:
  depth_w: 60    # 降低分辨率
  depth_h: 45
```

### **推理加速**

```python
# 模型量化
policy = torch.quantization.quantize_dynamic(
    policy, {torch.nn.Linear}, dtype=torch.qint8
)

# 降低分辨率
depth_w: 60  # 从80降到60
depth_h: 45  # 从60降到45

# TensorRT加速（高级）
import torch_tensorrt
policy_trt = torch_tensorrt.compile(policy, ...)
```

---

## 🔧 常见问题FAQ

### **Q1: ArUco检测不稳定怎么办？**

```
A: 多方面优化
1. 光照: 确保均匀光照，无强反光
2. 距离: 保持30-100cm最佳检测距离
3. 角度: ArUco码朝向相机，倾角<45°
4. 滤波: 添加卡尔曼滤波平滑位置
5. 大小: 使用更大的ArUco码（10cm以上）

代码改进:
# aruco_target_provider.py
from scipy.signal import medfilt

class ArucoTargetProvider:
    def __init__(self):
        self.position_buffer = []
    
    def update_target(self, event):
        # ... 获取trans
        
        # 中值滤波
        self.position_buffer.append([trans.x, trans.y, trans.z])
        if len(self.position_buffer) > 5:
            self.position_buffer.pop(0)
        
        # 发布滤波后的位置
        if len(self.position_buffer) >= 3:
            filtered_pos = np.median(self.position_buffer, axis=0)
```

### **Q2: 训练时如何模拟ArUco？**

```
A: 使用彩色立方体

在 manipulator_env.py 中:
# 创建目标标记（视觉参考）
aruco_visual = sim_utils.CuboidCfg(
    size=[0.1, 0.1, 0.01],
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.0, 1.0, 0.0),  # 绿色
        metallic=0.0
    ),
)

# 每个episode重置时更新位置
self.aruco_visual.set_world_poses(self.target_pos[env_ids])
```

### **Q3: 如何处理遮挡？**

```
A: RL策略自动学习

训练时的处理:
1. 深度图会自然反映遮挡
2. 策略学会绕过障碍物
3. 如果目标完全不可见 → 探索行为

部署时的处理:
1. ArUco检测丢失 → 保持上一次位置
2. 持续超过2秒 → 机械臂停止
3. 重新检测到 → 继续导航

代码:
# manipulator_navigation.py
def target_callback(self, msg):
    self.target_pose = msg
    self.target_lost_time = rospy.Time.now()

def control_callback(self, event):
    # 检查目标丢失时间
    if (rospy.Time.now() - self.target_lost_time).to_sec() > 2.0:
        rospy.logwarn("Target lost!")
        self.publish_zero_velocity()
        return
```

### **Q4: 碰撞后如何恢复？**

```
A: 自动后退+重规划

# 在 safety_filter 中
def safety_filter(self, ee_vel, ee_pos, target_pos):
    # 检测碰撞
    if self.detect_collision():
        # 后退
        retreat_vel = -ee_vel * 0.5
        return retreat_vel
    
    # 正常返回
    return ee_vel
```

### **Q5: 如何调试RL策略？**

```
A: 分步验证

# Step 1: 验证观测
obs = build_observation()
print("State:", obs["agents"]["observation"]["state"])
print("Depth range:", obs["agents"]["observation"]["depth"].min(), 
                       obs["agents"]["observation"]["depth"].max())

# Step 2: 验证策略输出
output = policy(obs)
action = output["agents", "action"]
print("Action:", action)  # 应该在合理范围内

# Step 3: 可视化
# 在RViz中添加深度图、速度箭头等

# Step 4: 记录轨迹
trajectory_buffer.add(ee_pos, time)
np.save('trajectory.npy', trajectory_buffer.get_trajectory())
```

---

## 🎓 进阶功能

### **功能1: 路径平滑**

```python
# 添加低通滤波器
from manipulator_utils import LowPassFilter

class ManipulatorNavigation:
    def __init__(self, cfg):
        self.vel_filter = LowPassFilter(alpha=0.3)
    
    def control_callback(self, event):
        # ... RL推理
        ee_vel_rl = policy(obs)
        
        # 滤波
        ee_vel_smooth = self.vel_filter.filter(ee_vel_rl)
        
        # 发布
        self.publish_ee_velocity(ee_vel_smooth)
```

### **功能2: 预测轨迹可视化**

```python
def get_predicted_trajectory(self, obs, horizon=3.0, dt=0.1):
    """预测未来轨迹"""
    pos = obs["agents"]["observation"]["state"][:3]
    traj = [pos]
    
    for _ in range(int(horizon / dt)):
        output = self.policy(obs)
        vel = output["agents", "action"][:3]
        pos = pos + vel * dt
        traj.append(pos)
        
        # 更新obs（简化）
        obs = self.update_obs(obs, pos)
    
    return np.array(traj)

# 发布到RViz
path_msg = Path()
for point in predicted_traj:
    pose = PoseStamped()
    pose.pose.position = Point(*point)
    path_msg.poses.append(pose)

self.traj_pub.publish(path_msg)
```

### **功能3: 在线学习/适应**

```python
# 收集实机数据，微调策略
class OnlineLearning:
    def __init__(self, policy):
        self.policy = policy
        self.buffer = []
    
    def collect(self, obs, action, reward, next_obs):
        """收集实机数据"""
        self.buffer.append((obs, action, reward, next_obs))
    
    def finetune(self, epochs=10):
        """微调策略"""
        # 使用收集的数据
        for epoch in range(epochs):
            loss = self.policy.train(self.buffer)
            print(f"Finetune epoch {epoch}, loss: {loss}")
```

---

## 🚀 部署清单

### **硬件清单**
- [ ] UR10e机械臂（已有）
- [ ] Realsense D435i相机（已有）
- [ ] 相机安装支架（眼在手上）
- [ ] ArUco码打印件（10×10cm，ID=0）
- [ ] 计算机（支持ROS + CUDA）
- [ ] 急停按钮（安全）

### **软件清单**
- [ ] Ubuntu 20.04
- [ ] ROS Noetic
- [ ] Isaac Sim 2023.1+
- [ ] PyTorch 2.0+
- [ ] CUDA 11.8+
- [ ] UR Robot Driver
- [ ] Realsense ROS wrapper

### **代码清单**
- [ ] `manipulator_env.py` (训练环境)
- [ ] `ppo_manipulator.py` (RL网络)
- [ ] `train_manipulator.py` (训练脚本)
- [ ] `manipulator_navigation.py` (ROS节点)
- [ ] `aruco_target_provider.py` (目标提供)
- [ ] `manipulator_nav.yaml` (配置)
- [ ] `manipulator_navigation.launch` (启动)
- [ ] 检查点文件 (训练后)

### **测试清单**
- [ ] TF树完整: `rosrun tf view_frames`
- [ ] 相机发布: `rostopic hz /camera/depth/image_raw`
- [ ] ArUco检测: `rostopic echo /rl_navigation/aruco_target`
- [ ] 占据地图: `rostopic hz /occupancy_map/voxel_map`
- [ ] 动态检测: `rosservice call /onboard_detector/get_dynamic_obstacles`
- [ ] RL策略加载: 检查日志
- [ ] 速度发布: `rostopic hz /ur10e_velocity_controller/command`
- [ ] 实际运动: 观察机械臂

---

## 📞 支持和调试

### **日志查看**

```bash
# 所有节点日志
rosnode list
rosnode info /manipulator_navigation

# 查看话题频率
rostopic hz /rl_navigation/aruco_target
rostopic hz /occupancy_map/voxel_map

# 查看TF延迟
rosrun tf tf_monitor base_link ee_link
rosrun tf tf_monitor base_link aruco_marker_0
```

### **性能监控**

```bash
# CPU/GPU使用
htop
nvidia-smi -l 1

# ROS性能
rqt_top  # 查看各节点CPU占用
rqt_graph  # 查看节点连接

# 网络带宽
rostopic bw /camera/depth/image_raw
```

---

## 🎉 成功案例

### **预期行为**

```
场景1: 简单到达
  起点: UR10e在Home位置
  终点: ArUco在前方50cm
  障碍物: 无
  
  结果:
    - 机械臂平滑移动
    - 直线路径
    - 5-8秒到达
    - 精度: < 3cm

场景2: 静态避障
  起点: UR10e在Home位置
  终点: ArUco在右前方60cm
  障碍物: 2个静态立方体
  
  结果:
    - 机械臂绕过障碍物
    - 曲线路径
    - 10-15秒到达
    - 精度: < 5cm

场景3: 动态避障
  起点: UR10e在Home位置
  终点: ArUco在左前方70cm
  障碍物: 1人在工作空间移动
  
  结果:
    - 机械臂实时调整路径
    - 避让移动的人
    - 12-20秒到达
    - 精度: < 5cm
```

---

## 🎯 下一步行动

### **立即开始**

```bash
# 1. 检查所有文件已创建
ls -la isaac-training/training/scripts/manipulator_*
ls -la ros1/navigation_runner/scripts/manipulator_*

# 2. 开始简化版训练（无障碍物）
cd isaac-training/training/scripts/
python train_manipulator.py env.num_static_obstacles=0 \
                            env.num_dynamic_obstacles=0

# 3. 监控训练
# 打开浏览器访问 wandb.ai
# 查看成功率曲线

# 4. 训练收敛后（成功率>90%）
# 添加障碍物，继续训练
```

### **预期时间线**

```
Week 1: 环境搭建 + 简单训练
Week 2: 完整训练（静态+动态障碍物）
Week 3: ROS集成 + 仿真测试
Week 4: 实机部署 + 调试
Week 5: 优化和完善

总计: 4-5周完成
```

---

祝您项目成功！如有任何问题随时询问。🚀

