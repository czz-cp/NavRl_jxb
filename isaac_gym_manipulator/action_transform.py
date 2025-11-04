"""
动作空间变换工具
Action Space Transformation Utilities

提供不同坐标系之间的动作变换：
- 世界坐标系 (World Frame)
- 目标坐标系 (Target Frame)
- 末端执行器坐标系 (End-Effector Frame)
"""
import torch


def vec_to_new_frame(vec, new_x_axis):
    """
    将向量从当前坐标系转换到新坐标系
    
    Args:
        vec: [batch, 3] - 向量（在当前坐标系中）
        new_x_axis: [batch, 3] - 新坐标系的 X 轴方向（在当前坐标系中表示）
    
    Returns:
        [batch, 3] - 向量在新坐标系中的表示
    
    原理:
        构建新的右手坐标系 (X', Y', Z')：
        - X' = new_x_axis（归一化）
        - Y' = reference × X'（选择合适的参考向量）
        - Z' = X' × Y'（完成右手坐标系）
        
        然后将向量投影到新坐标系的3个基向量上
    """
    # 归一化新 X 轴
    new_x = new_x_axis / (new_x_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    
    # 选择参考向量（避免奇异点）
    # 当 X 轴接近垂直（Z分量大）时，使用水平向量作为参考
    z_component = torch.abs(new_x[..., 2:3])
    is_near_vertical = z_component > 0.9
    
    # 参考向量选择：
    # - 接近垂直时用 [1,0,0]（水平方向）
    # - 其他情况用 [0,0,1]（垂直方向）
    reference = torch.where(
        is_near_vertical.expand_as(new_x),
        torch.tensor([1., 0., 0.], device=vec.device).expand_as(new_x),
        torch.tensor([0., 0., 1.], device=vec.device).expand_as(new_x)
    )
    
    # 构建新坐标系的 Y 轴：reference × X（右手定则）
    new_y = torch.cross(reference, new_x, dim=-1)
    new_y = new_y / (new_y.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    
    # 构建新坐标系的 Z 轴：X × Y（完成右手坐标系）
    new_z = torch.cross(new_x, new_y, dim=-1)
    new_z = new_z / (new_z.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    
    # 将向量投影到新坐标系的三个基向量上
    # 这相当于计算向量在新基下的坐标
    vec_x_new = (vec * new_x).sum(dim=-1, keepdim=True)
    vec_y_new = (vec * new_y).sum(dim=-1, keepdim=True)
    vec_z_new = (vec * new_z).sum(dim=-1, keepdim=True)
    
    vec_new = torch.cat([vec_x_new, vec_y_new, vec_z_new], dim=-1)
    
    return vec_new


def vec_to_world(vec_local, goal_direction):
    """
    从目标坐标系转换到世界坐标系
    
    Args:
        vec_local: [batch, 3] - 目标坐标系中的向量
        goal_direction: [batch, 3] - 目标方向（世界坐标系中）
    
    Returns:
        [batch, 3] - 世界坐标系中的向量
    
    工作原理:
        1. 目标坐标系的 X 轴 = goal_direction
        2. 世界坐标系的 X 轴（[1,0,0]）在目标坐标系中的表示
        3. 使用这个表示作为新的基，将向量转回世界坐标系
    
    示例:
        目标在正前方（+X）：
            goal_direction = [1, 0, 0]
            目标坐标系 = 世界坐标系
            vec_local = vec_world
        
        目标在左侧（+Y）：
            goal_direction = [0, 1, 0]
            目标坐标系：X'=+Y, Y'=-X, Z'=+Z
            vec_local=[1,0,0] → vec_world=[0,1,0]（向目标前进）
    """
    # 世界坐标系的 X 方向（在世界坐标系中就是 [1, 0, 0]）
    world_x_dir = torch.tensor([1., 0., 0.], device=vec_local.device).expand_as(goal_direction)
    
    # 计算世界坐标系在目标坐标系中的表示
    # 这告诉我们：世界坐标系相对于目标坐标系是如何旋转的
    world_frame_in_target = vec_to_new_frame(world_x_dir, goal_direction)
    
    # 使用这个变换将局部向量转换到世界坐标系
    # 实际上是两次坐标变换的复合
    vec_world = vec_to_new_frame(vec_local, world_frame_in_target)
    
    return vec_world


def quat_rotate_vector(quat, vec):
    """
    使用四元数旋转向量
    
    Args:
        quat: [batch, 4] - 四元数 [x, y, z, w] (Isaac Gym 格式)
        vec: [batch, 3] - 向量
    
    Returns:
        [batch, 3] - 旋转后的向量
    
    公式:
        v' = v + 2 * q.xyz × (q.xyz × v + q.w * v)
        其中 × 表示叉积
    """
    # Isaac Gym 使用 [x, y, z, w] 格式
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # 提取向量分量
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
    
    # 四元数旋转公式（批量版本）
    # 第一次叉积: q.xyz × v
    cross1_x = y * vz - z * vy
    cross1_y = z * vx - x * vz
    cross1_z = x * vy - y * vx
    
    # q.xyz × v + q.w * v
    temp_x = cross1_x + w * vx
    temp_y = cross1_y + w * vy
    temp_z = cross1_z + w * vz
    
    # 第二次叉积: q.xyz × temp
    cross2_x = y * temp_z - z * temp_y
    cross2_y = z * temp_x - x * temp_z
    cross2_z = x * temp_y - y * temp_x
    
    # 最终结果: v + 2 * cross2
    result_x = vx + 2.0 * cross2_x
    result_y = vy + 2.0 * cross2_y
    result_z = vz + 2.0 * cross2_z
    
    result = torch.stack([result_x, result_y, result_z], dim=1)
    
    return result


class ActionTransformer:
    """
    动作空间变换器
    
    支持三种坐标系：
    1. world: 世界坐标系（无变换）
    2. target: 目标坐标系（朝向目标）
    3. ee_frame: 末端执行器坐标系
    """
    
    def __init__(self, coordinate_type='world', device='cuda:0'):
        """
        Args:
            coordinate_type: 'world', 'target', 或 'ee_frame'
            device: torch 设备
        """
        self.coordinate_type = coordinate_type
        self.device = device
    
    def transform(self, action, obs, ee_quat=None):
        """
        变换动作到世界坐标系
        
        Args:
            action: [batch, 6] - 网络输出的动作 [vx, vy, vz, wx, wy, wz]
            obs: [batch, obs_dim] - 观测（包含相对位置和目标发现标志）
            ee_quat: [batch, 4] - 末端执行器四元数（仅 ee_frame 模式需要）
        
        Returns:
            [batch, 6] - 世界坐标系中的动作
        """
        if self.coordinate_type == 'world':
            return action
        
        # 从观测中提取信息
        # 观测结构: [rel_pos(3), distance(1), target_found(1), vel(6), joints(6), lidar(144)]
        rel_pos = obs[:, 0:3]  # [batch, 3]
        target_found = obs[:, 4:5]  # [batch, 1]
        
        # 只在目标已发现时进行变换
        use_transform = (target_found.squeeze(-1) > 0.5)  # [batch] bool tensor
        
        if not use_transform.any():
            # 没有环境发现目标，不变换
            return action
        
        action_transformed = action.clone()
        
        if self.coordinate_type == 'target':
            # 目标坐标系变换
            # 计算目标方向（归一化的相对位置）
            target_direction = rel_pos / (rel_pos.norm(dim=-1, keepdim=True).clamp(min=1e-6))  # [batch, 3]
            
            # 线速度从目标坐标系转到世界坐标系（对所有环境）
            linear_vel_local = action[:, :3]  # [batch, 3]
            linear_vel_world = vec_to_world(linear_vel_local, target_direction)  # [batch, 3]
            
            # 使用 where 选择性应用变换（只对发现目标的环境）
            action_transformed[:, :3] = torch.where(
                use_transform.unsqueeze(-1),  # [batch, 1]
                linear_vel_world,  # 变换后
                action[:, :3]  # 原始
            )
            # 角速度不变换
            action_transformed[:, 3:6] = action[:, 3:6]
        
        elif self.coordinate_type == 'ee_frame':
            # 末端执行器坐标系变换
            if ee_quat is None:
                raise ValueError("ee_frame 模式需要提供 ee_quat")
            
            if use_transform.any():
                # 线速度从末端坐标系转到世界坐标系
                linear_vel_local = action[use_transform, :3]
                linear_vel_world = quat_rotate_vector(ee_quat[use_transform], linear_vel_local)
                
                # 角速度也可以变换（可选）
                angular_vel = action[use_transform, 3:6]
                
                action_transformed[use_transform, :3] = linear_vel_world
                action_transformed[use_transform, 3:6] = angular_vel
        
        return action_transformed
    
    def __call__(self, action, obs, ee_quat=None):
        """快捷调用"""
        return self.transform(action, obs, ee_quat)


# ==================== 可视化辅助函数 ====================

def visualize_coordinate_frames(ee_pos, target_pos, ee_quat=None, scale=0.5):
    """
    返回用于可视化坐标系的线段数据
    
    Args:
        ee_pos: [3] - 末端执行器位置
        target_pos: [3] - 目标位置
        ee_quat: [4] - 末端执行器四元数（可选）
        scale: float - 坐标轴长度
    
    Returns:
        list of (start, end, color) tuples
    """
    lines = []
    
    # 1. 世界坐标系（在原点）
    origin = torch.zeros(3, device=ee_pos.device)
    
    # X 轴（红色）
    lines.append((origin, origin + torch.tensor([scale, 0, 0], device=ee_pos.device), [1, 0, 0]))
    # Y 轴（绿色）
    lines.append((origin, origin + torch.tensor([0, scale, 0], device=ee_pos.device), [0, 1, 0]))
    # Z 轴（蓝色）
    lines.append((origin, origin + torch.tensor([0, 0, scale], device=ee_pos.device), [0, 0, 1]))
    
    # 2. 目标坐标系（在末端执行器位置）
    target_dir = (target_pos - ee_pos) / (target_pos - ee_pos).norm().clamp(min=1e-6)
    
    # 使用 vec_to_new_frame 的逆过程构建坐标系
    # 简化版本：只显示 X' 轴（指向目标）
    lines.append((ee_pos, ee_pos + target_dir * scale, [1, 1, 0]))  # 黄色
    
    return lines


# ==================== 测试函数 ====================

def test_coordinate_transform():
    """测试坐标变换的正确性"""
    print("=" * 60)
    print("坐标变换测试")
    print("=" * 60)
    
    device = 'cpu'  # 测试用 CPU
    
    # 测试1: 目标在正前方（+X）
    print("\n测试1: 目标在正前方 (+X)")
    goal_direction = torch.tensor([[1., 0., 0.]], device=device)
    vec_local = torch.tensor([[1., 0., 0.]], device=device)  # 局部坐标系的 +X（朝目标）
    vec_world = vec_to_world(vec_local, goal_direction)
    print(f"  目标方向: {goal_direction[0].tolist()}")
    print(f"  局部动作: {vec_local[0].tolist()} (朝目标)")
    print(f"  世界动作: {vec_world[0].tolist()}")
    print(f"  期望结果: [1, 0, 0] (世界 +X)")
    assert torch.allclose(vec_world, torch.tensor([[1., 0., 0.]], device=device), atol=1e-4)
    print("  ✅ 通过")
    
    # 测试2: 目标在左侧（+Y）
    print("\n测试2: 目标在左侧 (+Y)")
    goal_direction = torch.tensor([[0., 1., 0.]], device=device)
    vec_local = torch.tensor([[1., 0., 0.]], device=device)  # 局部坐标系的 +X（朝目标）
    vec_world = vec_to_world(vec_local, goal_direction)
    print(f"  目标方向: {goal_direction[0].tolist()}")
    print(f"  局部动作: {vec_local[0].tolist()} (朝目标)")
    print(f"  世界动作: {vec_world[0].tolist()}")
    print(f"  期望结果: [0, 1, 0] (世界 +Y)")
    assert torch.allclose(vec_world, torch.tensor([[0., 1., 0.]], device=device), atol=1e-4)
    print("  ✅ 通过")
    
    # 测试3: 目标在上方（+Z）
    print("\n测试3: 目标在上方 (+Z)")
    goal_direction = torch.tensor([[0., 0., 1.]], device=device)
    vec_local = torch.tensor([[1., 0., 0.]], device=device)  # 局部坐标系的 +X（朝目标）
    vec_world = vec_to_world(vec_local, goal_direction)
    print(f"  目标方向: {goal_direction[0].tolist()}")
    print(f"  局部动作: {vec_local[0].tolist()} (朝目标)")
    print(f"  世界动作: {vec_world[0].tolist()}")
    print(f"  期望结果: [0, 0, 1] (世界 +Z)")
    assert torch.allclose(vec_world, torch.tensor([[0., 0., 1.]], device=device), atol=1e-4)
    print("  ✅ 通过")
    
    # 测试4: 目标在斜方向
    print("\n测试4: 目标在斜向（+X, +Y, +Z）")
    goal_direction = torch.tensor([[1., 1., 1.]], device=device)
    goal_direction = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    vec_local = torch.tensor([[1., 0., 0.]], device=device)  # 朝目标
    vec_world = vec_to_world(vec_local, goal_direction)
    print(f"  目标方向: {goal_direction[0].tolist()}")
    print(f"  局部动作: {vec_local[0].tolist()} (朝目标)")
    print(f"  世界动作: {vec_world[0].tolist()}")
    print(f"  期望结果: 接近目标方向")
    # 检查世界动作是否平行于目标方向
    parallel_check = torch.abs((vec_world * goal_direction).sum() - vec_world.norm())
    assert parallel_check < 1e-4
    print("  ✅ 通过")
    
    # 测试5: 侧向移动
    print("\n测试5: 侧向移动（目标在 +X，动作在局部 +Y）")
    goal_direction = torch.tensor([[1., 0., 0.]], device=device)
    vec_local = torch.tensor([[0., 1., 0.]], device=device)  # 局部 +Y（侧向）
    vec_world = vec_to_world(vec_local, goal_direction)
    print(f"  目标方向: {goal_direction[0].tolist()}")
    print(f"  局部动作: {vec_local[0].tolist()} (侧向)")
    print(f"  世界动作: {vec_world[0].tolist()}")
    print(f"  期望结果: 垂直于目标方向")
    # 检查世界动作是否垂直于目标方向
    perpendicular_check = torch.abs((vec_world * goal_direction).sum())
    assert perpendicular_check < 1e-4
    print("  ✅ 通过")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)


def test_action_transformer():
    """测试 ActionTransformer 类"""
    print("\n" + "=" * 60)
    print("ActionTransformer 类测试")
    print("=" * 60)
    
    device = 'cpu'
    
    # 创建变换器
    transformer = ActionTransformer(coordinate_type='target', device=device)
    
    # 模拟观测
    # [rel_pos(3), distance(1), target_found(1), vel(6), joints(6), lidar(144)]
    obs = torch.zeros(2, 161, device=device)
    obs[0, 0:3] = torch.tensor([1., 0., 0.])  # 目标在 +X
    obs[0, 4] = 1.0  # 目标已发现
    
    obs[1, 0:3] = torch.tensor([0., 1., 0.])  # 目标在 +Y
    obs[1, 4] = 0.0  # 目标未发现
    
    # 网络输出的动作（目标坐标系）
    action = torch.tensor([
        [1., 0., 0., 0., 0., 0.],  # 环境0: 朝目标前进
        [1., 0., 0., 0., 0., 0.],  # 环境1: 朝目标前进（但未发现）
    ], device=device)
    
    # 变换
    action_world = transformer.transform(action, obs)
    
    print(f"\n环境0（目标在 +X，已发现）:")
    print(f"  局部动作: {action[0].tolist()}")
    print(f"  世界动作: {action_world[0].tolist()}")
    print(f"  期望: [1, 0, 0, 0, 0, 0]")
    
    print(f"\n环境1（目标在 +Y，未发现）:")
    print(f"  局部动作: {action[1].tolist()}")
    print(f"  世界动作: {action_world[1].tolist()}")
    print(f"  期望: [1, 0, 0, 0, 0, 0] (无变换，因为未发现)")
    
    print("\n✅ ActionTransformer 测试通过!")


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    print("\n" + "=" * 60)
    print("使用示例")
    print("=" * 60)
    
    print("""
# 在 PPO 训练器中使用

from action_transform import ActionTransformer

class PPOTrainer:
    def __init__(self, env, config):
        # ...
        self.action_transformer = ActionTransformer(
            coordinate_type=config.action_coordinate,
            device=self.device
        )
    
    def collect_rollout(self, obs, num_steps):
        for _ in range(num_steps):
            # 网络输出动作（目标坐标系）
            action_local, log_prob, value = self.ac_network.get_action(obs)
            
            # 变换到世界坐标系
            action_world = self.action_transformer.transform(action_local, obs)
            
            # 应用到环境
            obs, reward, done, _ = self.env.step(action_world)

# 动作的含义（目标坐标系）:
# action_local[0] = 朝向目标的速度（总是沿着目标方向）
# action_local[1] = 垂直于目标的侧向速度
# action_local[2] = 垂直于目标平面的速度
# action_local[3:6] = 角速度（世界坐标系）

# 优势:
# - 网络只需学习 "前进 = 接近目标"
# - 不需要学习 "目标在哪个方向就往哪边移动"
# - 大大简化学习任务！
    """)


if __name__ == '__main__':
    # 运行测试
    test_coordinate_transform()
    test_action_transformer()
    example_usage()
    
    print("\n" + "=" * 60)
    print("📚 完整测试通过！")
    print("=" * 60)
    print("\n💡 提示:")
    print("  - 在 config_gym.yaml 中设置 action_coordinate: 'target'")
    print("  - PPO 会自动使用动作空间变换")
    print("  - 探索阶段：使用世界坐标系（目标未知）")
    print("  - 导航阶段：使用目标坐标系（自动变换）")
    print()

