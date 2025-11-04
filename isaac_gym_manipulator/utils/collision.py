"""
碰撞检测模块
包含连杆-障碍物距离计算、碰撞判断等方法
"""

import torch
from .math_utils import compute_point_to_segment_distance


def compute_link_obstacle_distances(rigid_body_states, obstacle_positions, num_envs, num_bodies, device='cuda:0'):
    """
    A.3 障碍物距离计算 - 连杆到点的最短距离
    
    计算机械臂所有连杆到所有障碍物的最短距离
    
    Args:
        rigid_body_states: [num_envs * num_bodies, 13] 刚体状态（CPU tensor）
        obstacle_positions: [num_envs, num_obstacles, 3] 障碍物位置（GPU tensor）
        num_envs: int 环境数量
        num_bodies: int 每个机械臂的刚体数量
        device: str 计算设备
        
    Returns:
        min_distances: [num_envs] 每个环境中最近的连杆-障碍物距离
        
    数学原理：
        1. 连杆表示为线段：由两个相邻刚体位置定义
        2. 计算点到线段的最短距离
        3. 遍历所有连杆-障碍物对，找到最小距离
    """
    min_distances = torch.full((num_envs,), float('inf'), device=device)
    
    # 遍历所有连杆
    for link_idx in range(num_bodies - 1):
        # 获取连杆的两个端点（当前body和下一个body的位置）
        # rigid_body_states 是 CPU tensor，需要用 CPU 索引
        body1_idx = torch.arange(num_envs, device='cpu') * num_bodies + link_idx
        body2_idx = torch.arange(num_envs, device='cpu') * num_bodies + link_idx + 1
        
        P1 = rigid_body_states[body1_idx, :3].to(device)  # [num_envs, 3]
        P2 = rigid_body_states[body2_idx, :3].to(device)  # [num_envs, 3]
        
        # 对每个障碍物计算距离
        num_obstacles = obstacle_positions.shape[1]
        for obs_idx in range(num_obstacles):
            Q = obstacle_positions[:, obs_idx, :]  # [num_envs, 3]
            
            # 计算点到线段的距离
            distance = compute_point_to_segment_distance(Q, P1, P2)
            
            # 更新最小距离
            min_distances = torch.min(min_distances, distance)
    
    return min_distances


def detect_collision_multisource(lidar_scan, lidar_range, rigid_body_states, obstacle_positions, 
                                  num_envs, num_bodies, device='cuda:0'):
    """
    多源碰撞检测（LiDAR + 连杆距离）
    
    Args:
        lidar_scan: [num_envs, 1, h_beams, v_beams] LiDAR扫描数据
        lidar_range: float LiDAR最大范围
        rigid_body_states: 刚体状态
        obstacle_positions: 障碍物位置
        num_envs: 环境数量
        num_bodies: 刚体数量
        device: 计算设备
        
    Returns:
        min_distance: [num_envs] 综合最小距离
        lidar_distance: [num_envs] LiDAR检测距离
        link_distance: [num_envs] 连杆检测距离
    """
    # 1. LiDAR 检测（末端附近）
    lidar_values = lidar_scan.reshape(num_envs, -1)
    max_lidar_value = lidar_values.max(dim=1)[0]
    lidar_distance = lidar_range - max_lidar_value
    
    # 2. 连杆-障碍物距离（整个机械臂）
    link_distance = compute_link_obstacle_distances(
        rigid_body_states, obstacle_positions, num_envs, num_bodies, device
    )
    
    # 3. 综合距离（取最小值）
    min_distance = torch.min(lidar_distance, link_distance)
    
    return min_distance, lidar_distance, link_distance


def compute_collision_penalty(min_distance, collision_threshold=0.15, warning_threshold=0.2):
    """
    计算分级碰撞惩罚
    
    Args:
        min_distance: [num_envs] 最小障碍物距离
        collision_threshold: float 碰撞阈值
        warning_threshold: float 警告阈值
        
    Returns:
        penalty: [num_envs] 惩罚值（负数）
    """
    num_envs = min_distance.shape[0]
    penalty = torch.zeros(num_envs, device=min_distance.device)
    
    # 严重碰撞（< collision_threshold）
    collision_mask = min_distance < collision_threshold
    penalty[collision_mask] = -10.0
    
    # 接近警告（collision_threshold ~ warning_threshold）
    warning_mask = (min_distance >= collision_threshold) & (min_distance < warning_threshold)
    penalty[warning_mask] = -2.0
    
    return penalty


def check_collision(min_distance, collision_threshold=0.15):
    """
    检查是否发生碰撞
    
    Args:
        min_distance: [num_envs] 最小障碍物距离
        collision_threshold: float 碰撞阈值
        
    Returns:
        collision: [num_envs] bool tensor
    """
    return min_distance < collision_threshold

