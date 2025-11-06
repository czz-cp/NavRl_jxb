# UR5e DDPG 特征处理和奖励函数详解

## 代码位置

- **特征处理**: `ur5e_env.py` 第118-141行
- **奖励函数**: `ur5e_env.py` 第125-135行
- **状态定义**: `ur5e_env.py` 第25行

---

## 一、状态空间（State Space）

### 状态维度定义

```python
state_dim = 13  # 状态空间维度
```

**状态组成**:
```
状态 = [6个关节角度 + 3个目标位置 + 3个末端点到目标的位置差 + 1个是否触碰到目标的标志]
     = [6 + 3 + 3 + 1] = 13维
```

### 详细组成

#### 1. 关节角度 (6维)
```python
self.arm_info['theta'][0:6]  # theta1 到 theta6
```
- **含义**: 6个关节的当前角度（弧度）
- **范围**: 每个关节有不同的角度范围限制

#### 2. 目标位置 (3维)
```python
target_position = [target_x, target_y, target_z]
```
- **含义**: 目标点的3D坐标
- **默认值**: `GOAL = {'x': 0.5, 'y': 0.5, 'z': 0.5}`

#### 3. 末端点到目标的位置差 (3维)
```python
dist4 = [goal_x - end_point_x, goal_y - end_point_y, goal_z - end_point_z]
```
- **含义**: 目标位置减去末端执行器位置的向量差
- **作用**: 告诉智能体"还需要移动多少距离"

#### 4. 是否触碰到目标 (1维)
```python
[1. if self.on_goal else 0.]
```
- **含义**: 二进制标志，表示是否到达目标
- **判定条件**: 三个方向的距离都小于0.02

---

## 二、特征处理（Feature Engineering）

### 代码位置

```python
# ur5e_env.py 第118-141行
#feature enginering
# dist1 = np.array([self.goal['x'] - self.point0['x'], ...])  # 已注释
# dist2 = np.array([self.goal['x'] - self.point1['x'], ...])  # 已注释
# dist3 = np.array([self.goal['x'] - self.point2['x'], ...])  # 已注释
dist4 = np.array([self.goal['x'] - self.end_point['x'], 
                  self.goal['y'] - self.end_point['y'], 
                  self.goal['z'] - self.end_point['z']])

# ... 奖励计算 ...

# 状态拼接
s = np.concatenate((s, dist4, [1. if self.on_goal else 0.]))
```

### 特征工程的目的

**为什么要计算 `dist4`？**

1. **提供相对位置信息**: 
   - 直接告诉智能体"目标在哪里"（相对于末端执行器）
   - 比绝对位置更有用（相对位置更直观）

2. **减少状态空间复杂性**:
   - 如果只给绝对位置，智能体需要自己计算相对位置
   - 提前计算相对位置，让学习更容易

3. **提供方向信息**:
   - `dist4` 的符号表示方向（正负）
   - 大小表示距离

### 特征工程流程

```
输入:
  - 关节角度: [theta1, theta2, ..., theta6] (6维)
  - 目标位置: [target_x, target_y, target_z] (3维)
  - 末端位置: [end_x, end_y, end_z] (从MuJoCo计算得到)

步骤1: 计算相对位置差
  dist4 = [goal_x - end_x, goal_y - end_y, goal_z - end_z]

步骤2: 检查是否到达目标
  on_goal = (|goal_x - end_x| <= 0.02) and 
            (|goal_y - end_y| <= 0.02) and 
            (|goal_z - end_z| <= 0.02)

步骤3: 拼接状态
  状态 = [关节角度(6) + 目标位置(3) + 相对位置差(3) + 到达标志(1)]
       = [13维]
```

### 状态拼接代码解析

```python
# 第93行: 基础状态
s = np.concatenate((self.arm_info['theta'], target_position))
# s = [theta1-6, target_x, target_y, target_z]  (9维)

# 第141行: 添加特征工程的结果
s = np.concatenate((s, dist4, [1. if self.on_goal else 0.]))
# s = [theta1-6, target_x, target_y, target_z, dist4_x, dist4_y, dist4_z, on_goal]  (13维)
```

### 为什么注释掉了 dist1, dist2, dist3？

```python
# dist1 = np.array([self.goal['x'] - self.point0['x'], ...])  # 已注释
# dist2 = np.array([self.goal['x'] - self.point1['x'], ...])  # 已注释
# dist3 = np.array([self.goal['x'] - self.point2['x'], ...])  # 已注释
```

**原因**:
- `point0`, `point1`, `point2` 可能是额外的参考点
- 当前实现只使用 `dist4`（末端点到目标的位置差）
- 简化状态空间，减少不必要的特征

**如果使用 dist1, dist2, dist3**:
- 状态维度会变成：`6 + 3 + 3 + 3 + 3 + 3 + 1 = 22维`
- 可能包含冗余信息
- 增加学习难度

---

## 三、奖励函数（Reward Function）

### 奖励函数设计

```python
# ur5e_env.py 第124-135行
dist5 = [(self.goal['x'] - self.end_point['x']), 
         (self.goal['y'] - self.end_point['y']), 
         (self.goal['z'] - self.end_point['z'])]

# 基础奖励：负的欧几里得距离
r = -np.sqrt(dist5[0] ** 2 + dist5[1] ** 2 + dist5[2] ** 2)

# 成功奖励：到达目标
if abs(self.goal['x'] - self.end_point['x']) <= 0.02:
    if abs(self.goal['y'] - self.end_point['y']) <= 0.02:
        if abs(self.goal['z'] - self.end_point['z']) <= 0.02:
            r += 10  # 成功奖励
            self.on_goal = True
            self.on_goal_count += 1
            if self.on_goal_count > 100:
                self.on_goal_count = 0
                done = True
```

### 奖励函数组成

#### 1. 基础奖励（距离惩罚）

```python
r = -np.sqrt(dist5[0] ** 2 + dist5[1] ** 2 + dist5[2] ** 2)
```

**数学公式**:
```
r_distance = -√[(goal_x - end_x)² + (goal_y - end_y)² + (goal_z - end_z)²]
```

**特点**:
- **负值**: 距离越远，奖励越小（惩罚）
- **单调递减**: 距离越近，奖励越大
- **范围**: 通常是负值（除非距离为0）

**示例**:
```
距离 = 0.0  → r = 0.0
距离 = 0.1  → r = -0.1
距离 = 0.5  → r = -0.5
距离 = 1.0  → r = -1.0
```

#### 2. 成功奖励（稀疏奖励）

```python
if abs(self.goal['x'] - self.end_point['x']) <= 0.02:
    if abs(self.goal['y'] - self.end_point['y']) <= 0.02:
        if abs(self.goal['z'] - self.end_point['z']) <= 0.02:
            r += 10  # 成功奖励
```

**特点**:
- **稀疏奖励**: 只有到达目标时才给予
- **阈值**: 0.02（2厘米）
- **奖励值**: +10（相对于距离惩罚，这是一个很大的奖励）

**奖励范围**:
```
距离惩罚: -∞ ~ 0
成功奖励: +10
总奖励: -∞ ~ +10
```

### 奖励函数设计分析

#### 优点

1. **简单直观**:
   - 距离惩罚直接反映任务目标
   - 成功奖励明确

2. **稀疏+密集结合**:
   - 密集奖励（距离惩罚）: 提供连续的学习信号
   - 稀疏奖励（成功奖励）: 鼓励最终成功

3. **数值范围合理**:
   - 距离惩罚通常在[-1, 0]范围
   - 成功奖励+10，提供明显的正向信号

#### 潜在问题

1. **距离惩罚可能过小**:
   - 如果距离很大（比如1.0），奖励只有-1.0
   - 成功奖励+10，可能导致智能体过度关注成功奖励

2. **没有中间奖励**:
   - 只有距离惩罚和成功奖励
   - 没有考虑动作平滑性、速度等

3. **阈值固定**:
   - 0.02的阈值是硬编码的
   - 可能不适合所有任务

### 奖励函数改进建议

#### 改进1: 缩放距离惩罚

```python
# 当前实现
r = -np.sqrt(dist5[0] ** 2 + dist5[1] ** 2 + dist5[2] ** 2)

# 改进：缩放系数
distance = np.sqrt(dist5[0] ** 2 + dist5[1] ** 2 + dist5[2] ** 2)
r = -distance * scale_factor  # 例如 scale_factor = 10
```

#### 改进2: 添加中间奖励

```python
# 添加距离改进奖励
distance_prev = ...  # 上一步的距离
distance_curr = np.sqrt(dist5[0] ** 2 + dist5[1] ** 2 + dist5[2] ** 2)
distance_improvement = distance_prev - distance_curr
r += distance_improvement * 0.1  # 奖励距离改进
```

#### 改进3: 添加动作平滑性奖励

```python
# 惩罚动作变化过大
action_change = np.linalg.norm(action - prev_action)
r -= action_change * 0.01  # 惩罚大的动作变化
```

---

## 四、终止条件（Termination）

### 终止条件

```python
# 条件1: 到达目标并保持100步
if self.on_goal_count > 100:
    self.on_goal_count = 0
    done = True

# 条件2: 超过最大步数
if self.episode_step >= 1000:
    done = True
```

**设计原因**:
1. **保持100步**: 避免偶然到达目标就结束
2. **最大步数**: 防止episode过长

---

## 五、完整的状态和奖励流程

### 完整流程示例

```
步骤1: 初始化
  关节角度: [0, 0, 0, 0, 0, 0]
  目标位置: [0.5, 0.5, 0.5]
  末端位置: [1.0, 0.0, 1.0]  (从MuJoCo计算)

步骤2: 特征工程
  dist4 = [0.5-1.0, 0.5-0.0, 0.5-1.0] = [-0.5, 0.5, -0.5]
  距离 = √[(-0.5)² + 0.5² + (-0.5)²] = √0.75 ≈ 0.866
  
  状态 = [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0]
        [6个关节角度] [3个目标位置] [3个相对位置差] [1个到达标志]

步骤3: 奖励计算
  基础奖励: r = -0.866
  到达检查: 0.866 > 0.02 → 未到达
  成功奖励: r += 0
  总奖励: r = -0.866

步骤4: 执行动作
  动作: [0.01, -0.01, 0.01, 0, 0, 0]
  更新关节角度
  重新计算末端位置

步骤5: 重复步骤2-4直到终止
```

---

## 六、与当前PPO实现的对比

### 特征处理对比

| 特性 | UR5e DDPG | 当前PPO实现 |
|------|-----------|------------|
| **输入维度** | 13维（简单状态） | 256维（融合特征） |
| **特征类型** | 关节角度 + 位置差 | 3D体素 + 辅助信息 |
| **特征工程** | 手动计算相对位置 | 深度学习自动提取 |
| **复杂度** | 低（手工设计） | 高（神经网络学习） |

### 奖励函数对比

| 特性 | UR5e DDPG | 当前PPO实现 |
|------|-----------|------------|
| **奖励类型** | 距离惩罚 + 成功奖励 | 多组件奖励（距离、碰撞、平滑等） |
| **奖励范围** | [-∞, +10] | 更复杂的奖励设计 |
| **稀疏性** | 稀疏（成功时） | 密集（每步都有） |

---

## 七、总结

### 特征处理要点

1. **状态维度**: 13维，包含关节角度、目标位置、相对位置差、到达标志
2. **特征工程**: 计算目标到末端点的相对位置差
3. **设计目的**: 提供相对位置信息，简化学习

### 奖励函数要点

1. **基础奖励**: 负的欧几里得距离（距离惩罚）
2. **成功奖励**: +10（到达目标时）
3. **设计特点**: 简单直观，结合密集和稀疏奖励

### 关键代码位置

- **状态定义**: `ur5e_env.py` 第25行
- **特征工程**: `ur5e_env.py` 第118-141行
- **奖励计算**: `ur5e_env.py` 第124-135行
- **终止条件**: `ur5e_env.py` 第133-138行

---

## 参考

1. **DDPG算法**: Deep Deterministic Policy Gradient
2. **MuJoCo**: 物理仿真引擎
3. **UR5e机械臂**: Universal Robots的6自由度机械臂
4. **特征工程**: 手动设计特征以帮助强化学习


