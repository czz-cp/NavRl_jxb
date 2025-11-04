"""
奖励函数模块
包含探索奖励、导航奖励、惩罚等计算
"""

import torch
from .math_utils import quat_rotate_vector_batch


def compute_downward_search_reward(ee_quat, reward_weight, device='cuda:0'):
    """
    计算向下搜索奖励（摄像头朝向水平面以下）
    
    Args:
        ee_quat: [num_envs, 4] 末端执行器四元数
        reward_weight: float 奖励权重
        device: str 计算设备
        
    Returns:
        reward: [num_envs] 向下搜索奖励
        
    原理：
        - 摄像头朝向 = 末端局部坐标系的 +X 轴
        - 转换到世界坐标系
        - 检查 Z 分量是否 < 0（朝向水平面以下）
        - 越朝下，奖励越高
    """
    num_envs = ee_quat.shape[0]
    
    # 摄像头朝向（局部坐标系中的 +X 方向，即末端正前方）
    cam_forward_local = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3)
    
    # 转换到世界坐标系
    cam_forward_world = quat_rotate_vector_batch(ee_quat, cam_forward_local)
    
    # 检查摄像头是否朝向水平面以下（Z分量 < 0）
    z_component = cam_forward_world[:, 2]  # Z+ 是向上，Z- 是向下
    
    # 只要朝向水平面以下就给奖励，越朝下奖励越高
    downward_component = -z_component  # 转换：0 (水平) 到 1 (垂直向下)
    downward_bonus = torch.clamp(downward_component, 0, 1.0) * reward_weight
    
    return downward_bonus


def compute_systematic_scan_reward(ee_vel_linear, reward_weight, prev_direction=None):
    """
    计算系统性扫描奖励（鼓励有规律的扫描模式）
    
    Args:
        ee_vel_linear: [num_envs, 3] 末端线速度
        reward_weight: float 奖励权重
        prev_direction: [num_envs, 3] 上一步的移动方向（可选）
        
    Returns:
        reward: [num_envs] 系统性扫描奖励
    """
    # 归一化移动方向
    current_direction = ee_vel_linear / (torch.norm(ee_vel_linear, dim=1, keepdim=True) + 1e-6)
    
    if prev_direction is not None:
        # 计算方向一致性（点积）
        direction_consistency = (current_direction * prev_direction).sum(dim=1)
        direction_consistency = torch.clamp(direction_consistency, -1.0, 1.0)
        
        # 奖励一致的移动方向（系统性扫描）
        systematic_bonus = (direction_consistency + 1.0) / 2.0 * reward_weight
    else:
        # 第一步，没有历史数据
        systematic_bonus = torch.zeros(ee_vel_linear.shape[0], device=ee_vel_linear.device)
    
    return systematic_bonus


def compute_active_exploration_reward(ee_vel_linear, ee_vel_angular, reward_weight):
    """
    奖励主动探索移动（在探索阶段鼓励移动）
    
    Args:
        ee_vel_linear: [num_envs, 3] 末端线速度
        ee_vel_angular: [num_envs, 3] 末端角速度
        reward_weight: float 奖励权重
        
    Returns:
        reward: [num_envs] 主动探索奖励
    """
    # 计算线速度和角速度的大小
    velocity_magnitude = torch.norm(ee_vel_linear, dim=1)
    angular_magnitude = torch.norm(ee_vel_angular, dim=1)
    
    # 奖励有意义的移动（线速度或角速度）
    movement_score = (velocity_magnitude / 1.0) + (angular_magnitude / 2.0)
    
    # 归一化并计算奖励
    reward = torch.clamp(movement_score, 0, 2.0) * reward_weight
    
    return reward


def compute_static_penalty(ee_vel_linear, ee_vel_angular, penalty_weight):
    """
    惩罚静止不动（探索时应该移动，不应该静止）
    
    Args:
        ee_vel_linear: [num_envs, 3] 末端线速度
        ee_vel_angular: [num_envs, 3] 末端角速度
        penalty_weight: float 惩罚权重
        
    Returns:
        penalty: [num_envs] 静止惩罚
    """
    # 计算总体运动程度
    velocity_magnitude = torch.norm(ee_vel_linear, dim=1)
    angular_magnitude = torch.norm(ee_vel_angular, dim=1)
    
    # 静止程度：速度越小，静止程度越高
    movement_score = (velocity_magnitude / 1.0) + (angular_magnitude / 2.0)
    static_score = torch.clamp(2.0 - movement_score, 0, 2.0)  # 反向：运动少 = 静止多
    
    # 惩罚静止不动的行为
    penalty = static_score * penalty_weight
    
    return penalty


def compute_z_exploration_reward(ee_vel_linear, reward_weight):
    """
    Z轴探索奖励（只奖励向上移动）
    
    Args:
        ee_vel_linear: [num_envs, 3] 末端线速度
        reward_weight: float 奖励权重
        
    Returns:
        reward: [num_envs] Z轴探索奖励
    """
    z_velocity = ee_vel_linear[:, 2]  # 不取绝对值，保留正负
    # 只奖励向上移动（正值），向下不奖励
    z_exploration_bonus = torch.clamp(z_velocity, 0, float('inf')) * reward_weight
    
    return z_exploration_bonus

