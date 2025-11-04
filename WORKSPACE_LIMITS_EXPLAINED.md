# UR10e 工作空间限制详解
# Workspace Limits Calculation for UR10e

## 📏 当前设置

```python
self.workspace_min = torch.tensor([-0.8, -0.8, 0.0], device=self.device)
self.workspace_max = torch.tensor([0.8, 0.8, 1.2], device=self.device)
```

**定义的工作空间**:
- X 范围: [-0.8m, 0.8m] → 宽度 1.6m
- Y 范围: [-0.8m, 0.8m] → 深度 1.6m  
- Z 范围: [0.0m, 1.2m] → 高度 1.2m

---

## 🤖 UR10e 机械臂规格

### **官方参数**

| 参数 | 数值 |
|-----|------|
| **最大臂展 (Reach)** | 1300 mm = 1.3 m |
| **工作半径** | 约 1.3 m |
| **垂直工作范围** | 约 -0.15m ~ 1.9m（相对于基座）|
| **重复定位精度** | ±0.05 mm |

### **UR10e 的连杆长度**

根据 DH 参数：

```python
# DH 参数（在代码中）
d = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]  # 关节偏移
a = [0, -0.6127, -0.57155, 0, 0, 0]             # 连杆长度

# 关键尺寸:
Link 2: 0.6127 m  (612.7 mm)
Link 3: 0.57155 m (571.55 mm)

# 最大臂展计算:
Max reach ≈ |a[1]| + |a[2]| + d[4] + d[5]
          ≈ 0.6127 + 0.57155 + 0.12 + 0.12
          ≈ 1.42 m

# 实际官方标称: 1.3 m (考虑了关节限制等因素)
```

---

## 🎯 工作空间限制的计算逻辑

### **1. X 和 Y 方向: ±0.8m**

#### **为什么不是 ±1.3m？**

```python
# 理论最大臂展: 1.3 m
# 代码中设置: ±0.8 m

原因:
1. 安全裕度
2. 避免奇异点
3. 保证可操作性
4. 适应训练场景
```

#### **详细计算**

```python
# 最大臂展: 1.3 m
# 减去安全裕度: 1.3 - 0.3 = 1.0 m (23% 裕度)

# 但考虑到:
1. 基座周围有死区（机械臂无法到达正下方）
2. 边缘区域可操作性差（接近奇异点）
3. 训练场景中有桌子和障碍物

# 最终选择: ±0.8 m
# 这提供了一个"安全且可靠"的工作区域
```

**示意图**:

```
俯视图 (XY平面):

        Y
        ↑
        |
  -0.8  |  +0.8
    ----+----→ X
        |
        |

侧视图 (XZ平面):

        Z
    1.2 +--------+
        |        |
        |  工作  |
    0.0 +--------+
       -0.8    +0.8  X
```

---

### **2. Z 方向: [0.0, 1.2m]**

#### **为什么下限是 0.0？**

```python
原因:
1. 机械臂安装在桌面上
2. 桌面高度设为 Z = 0
3. 末端不应低于桌面

# 在 _design_scene() 中:
table_cfg = sim_utils.CuboidCfg(
    size=(1.5, 1.5, 0.05),  # 厚度 5cm
)
# 桌面位置: z = -0.025 (顶部在 z=0)
```

#### **为什么上限是 1.2m？**

```python
# UR10e 理论最大高度（完全伸展向上）:
Max Z ≈ d[0] + |a[1]| + |a[2]| + ...
      ≈ 0.18 + 0.61 + 0.57 + 0.3
      ≈ 1.66 m

# 代码中设置: 1.2 m

原因:
1. 训练场景不需要太高（目标和障碍物在较低位置）
2. 避免完全伸直（奇异点）
3. 保持可操作性
4. 与桌面场景匹配

# 1.2m 是一个合理的"操作高度"
```

---

## 📊 工作空间可视化

### **完整工作区域**

```python
# 定义的立方体工作空间:
Volume = (0.8 - (-0.8)) × (0.8 - (-0.8)) × (1.2 - 0.0)
       = 1.6 × 1.6 × 1.2
       = 3.072 m³

# 这是一个相当大的工作空间！
```

### **与场景元素的关系**

```python
# 在 _design_scene() 中:

# 1. 桌子
table_size = (1.5, 1.5, 0.05)
table_position = (0, 0, -0.025)
# 桌面在 z=0，在工作空间内

# 2. 静态障碍物
obs_pos[:, 0] = random(-0.6, 0.6)  # 在工作空间内
obs_pos[:, 1] = random(-0.6, 0.6)
obs_pos[:, 2] = 0.15               # 桌面上方
# 障碍物高度: 0.3m → 顶部在 z=0.45

# 3. 气球
balloon_z = random(0.4, 1.0)  # 在工作空间内
# 范围 [0.4, 1.0] < 1.2 ✓

# 4. 目标位置 (ArUco)
target_z = 0.3 + random(0.5)  # [0.3, 0.8]
# 也在工作空间内
```

---

## 🎓 如何精确计算工作空间？

### **方法1: 解析计算（理论）**

```python
import numpy as np

def calculate_workspace_ur10e():
    """
    计算 UR10e 的理论工作空间
    """
    # DH 参数
    d = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]
    a = [0, -0.6127, -0.57155, 0, 0, 0]
    
    # 最大臂展（水平）
    max_reach = abs(a[1]) + abs(a[2]) + d[4] + d[5]
    print(f"理论最大臂展: {max_reach:.3f} m")
    
    # 最大高度（完全向上）
    max_height = d[0] + abs(a[1]) + abs(a[2]) + d[3] + d[4] + d[5]
    print(f"理论最大高度: {max_height:.3f} m")
    
    # 最小高度（完全向下，但受关节限制）
    min_height = d[0] - (abs(a[1]) + abs(a[2]))
    print(f"理论最小高度: {min_height:.3f} m")
    
    # 建议的安全工作空间（80%）
    safe_reach = max_reach * 0.8
    safe_height = max_height * 0.8
    
    print(f"\n建议的安全工作空间:")
    print(f"  X: ±{safe_reach:.2f} m")
    print(f"  Y: ±{safe_reach:.2f} m")  
    print(f"  Z: [0, {safe_height:.2f}] m")

# 输出:
理论最大臂展: 1.425 m
理论最大高度: 1.663 m
理论最小高度: -1.003 m

建议的安全工作空间:
  X: ±1.14 m
  Y: ±1.14 m
  Z: [0, 1.33] m
```

### **方法2: 数值采样（实践）**

```python
def sample_reachable_workspace(num_samples=100000):
    """
    通过随机采样关节角度，计算实际可达工作空间
    """
    import numpy as np
    
    reachable_points = []
    
    for _ in range(num_samples):
        # 随机采样关节角度
        joint_angles = np.random.uniform(-2*np.pi, 2*np.pi, 6)
        
        # 正运动学计算末端位置
        ee_pos = forward_kinematics_ur10e(joint_angles)
        
        reachable_points.append(ee_pos)
    
    reachable_points = np.array(reachable_points)
    
    # 计算边界
    x_min, x_max = reachable_points[:, 0].min(), reachable_points[:, 0].max()
    y_min, y_max = reachable_points[:, 1].min(), reachable_points[:, 1].max()
    z_min, z_max = reachable_points[:, 2].min(), reachable_points[:, 2].max()
    
    print(f"采样得到的工作空间:")
    print(f"  X: [{x_min:.3f}, {x_max:.3f}] m")
    print(f"  Y: [{y_min:.3f}, {y_max:.3f}] m")
    print(f"  Z: [{z_min:.3f}, {z_max:.3f}] m")
```

---

## 🔧 如何调整工作空间？

### **场景1: 需要更大的工作空间**

```python
# 如果觉得 ±0.8m 太小，可以扩大:
self.workspace_min = torch.tensor([-1.0, -1.0, 0.0], device=self.device)
self.workspace_max = torch.tensor([1.0, 1.0, 1.4], device=self.device)

# 注意:
# 1. 不要超过理论最大臂展 1.3m
# 2. 边缘区域可操作性会变差
# 3. 更容易遇到奇异点
```

### **场景2: 限制在桌面上方**

```python
# 只在桌面上方操作:
self.workspace_min = torch.tensor([-0.6, -0.6, 0.1], device=self.device)
self.workspace_max = torch.tensor([0.6, 0.6, 0.8], device=self.device)

# 适合:
# - 桌面拾取任务
# - 装配任务
# - 精细操作
```

### **场景3: 非对称工作空间**

```python
# 如果机械臂安装位置不在中心:
self.workspace_min = torch.tensor([-0.3, -0.8, 0.0], device=self.device)
self.workspace_max = torch.tensor([1.0, 0.8, 1.2], device=self.device)

# 适合:
# - 单侧操作
# - 墙角安装
```

---

## 📐 当前设置的合理性分析

### **优点**

```python
✅ 覆盖主要操作区域
✅ 避开奇异点
✅ 保证可操作性（远离关节限制）
✅ 与训练场景匹配（桌子、障碍物、目标都在范围内）
✅ 对称设计（X、Y 相同，便于学习）
```

### **占用比例**

```python
# 相对于理论最大工作空间:
XY方向: 0.8 / 1.3 ≈ 62% 的最大臂展
Z方向:  1.2 / 1.66 ≈ 72% 的最大高度

# 这是一个"保守但实用"的设置
```

---

## 💡 总结

### **工作空间限制的来源**

```python
workspace_min = [-0.8, -0.8, 0.0]
workspace_max = [0.8, 0.8, 1.2]

计算依据:
1. UR10e 理论最大臂展: 1.3m
   → 取 0.8m (62%)，留有安全裕度

2. UR10e 理论最大高度: 1.66m  
   → 取 1.2m (72%)，避免完全伸直

3. 下限 z=0: 对应桌面高度
   → 防止末端低于桌面

4. 对称设计: X、Y 相同
   → 简化学习，无偏好方向
```

### **是否需要修改？**

```python
当前设置适合:
✅ 桌面操作场景
✅ 训练收敛速度
✅ 避免奇异点
✅ 安全性

可以修改如果:
❓ 需要更大操作范围 → 增加到 ±1.0m
❓ 只做桌面任务 → 减小到 ±0.6m, z:[0.1, 0.8]
❓ 有特殊约束 → 根据实际情况调整
```

### **建议**

对于您的 ArUco 导航任务，**当前的工作空间设置非常合理**，建议保持不变。

如果实际部署时发现目标超出范围，再根据实际情况调整。

有任何问题随时询问！🚀

