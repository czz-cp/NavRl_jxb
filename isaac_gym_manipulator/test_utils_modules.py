"""
测试工具模块
验证分离后的模块功能是否正常
"""

import torch
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    quat_rotate_vector_batch,
    compute_pose_error_three_point,
    compute_link_obstacle_distances,
    compute_downward_search_reward,
    compute_z_exploration_reward,
)

def test_math_utils():
    """测试数学工具"""
    print("=" * 60)
    print("测试数学工具模块")
    print("=" * 60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_envs = 4
    
    # 测试四元数旋转
    quat = torch.tensor([[0, 0, 0, 1]] * num_envs, device=device, dtype=torch.float32)
    vec = torch.tensor([[1, 0, 0]] * num_envs, device=device, dtype=torch.float32)
    result = quat_rotate_vector_batch(quat, vec)
    print(f"✅ 四元数旋转: {result[0]}")
    
    # 测试三点位姿误差
    ee_pos = torch.rand(num_envs, 3, device=device)
    ee_quat = torch.tensor([[0, 0, 0, 1]] * num_envs, device=device, dtype=torch.float32)
    target_pos = torch.rand(num_envs, 3, device=device)
    
    pose_error = compute_pose_error_three_point(ee_pos, ee_quat, target_pos)
    print(f"✅ 三点位姿误差: shape={pose_error.shape}, mean={pose_error.mean():.4f}")


def test_collision():
    """测试碰撞检测"""
    print("\n" + "=" * 60)
    print("测试碰撞检测模块")
    print("=" * 60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_envs = 2
    num_bodies = 5
    num_obstacles = 3
    
    # 模拟刚体状态
    rigid_body_states = torch.rand(num_envs * num_bodies, 13)  # CPU
    obstacle_positions = torch.rand(num_envs, num_obstacles, 3, device=device)  # GPU
    
    distances = compute_link_obstacle_distances(
        rigid_body_states, obstacle_positions, num_envs, num_bodies, device
    )
    print(f"✅ 连杆-障碍物距离: shape={distances.shape}, min={distances.min():.4f}")


def test_rewards():
    """测试奖励函数"""
    print("\n" + "=" * 60)
    print("测试奖励函数模块")
    print("=" * 60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_envs = 4
    
    # 测试 Z 轴探索奖励
    ee_vel = torch.tensor([[0, 0, 0.15], [0, 0, -0.15], [0, 0, 0], [0, 0, 0.3]], 
                          device=device, dtype=torch.float32)
    z_reward = compute_z_exploration_reward(ee_vel, reward_weight=10.0)
    print(f"✅ Z轴探索奖励: {z_reward}")
    print(f"   向上(0.15): {z_reward[0]:.2f}, 向下(-0.15): {z_reward[1]:.2f}")
    
    # 测试向下搜索奖励
    ee_quat = torch.tensor([[0, 0, 0, 1]] * num_envs, device=device, dtype=torch.float32)
    downward_reward = compute_downward_search_reward(ee_quat, reward_weight=5.0, device=device)
    print(f"✅ 向下搜索奖励: {downward_reward}")


def main():
    print("\n" + "🔧" * 30)
    print("工具模块测试")
    print("🔧" * 30 + "\n")
    
    try:
        test_math_utils()
        test_collision()
        test_rewards()
        
        print("\n" + "✅" * 30)
        print("所有测试通过！")
        print("✅" * 30)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())












