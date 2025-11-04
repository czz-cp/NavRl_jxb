"""
传感器模块
包含 LiDAR、摄像头相关的计算和可视化
"""

import torch
import numpy as np
from .math_utils import quat_rotate_vector_batch, quat_rotate_vector


def compute_lidar_ray_directions(lidar_hbeams, lidar_vbeams, lidar_vfov, device='cuda:0'):
    """
    预计算 LiDAR 射线方向（局部坐标系）
    
    Args:
        lidar_hbeams: int 水平光束数
        lidar_vbeams: int 垂直光束数
        lidar_vfov: tuple (min_angle, max_angle) 垂直视场角（度）
        device: str 计算设备
        
    Returns:
        ray_directions: [h_beams, v_beams, 3] 射线方向向量（归一化）
    """
    # 水平角度：0° 到 360°
    h_angles = torch.linspace(0, 2 * np.pi, lidar_hbeams, device=device)
    
    # 垂直角度：vfov[0] 到 vfov[1]
    v_angles_deg = torch.linspace(lidar_vfov[0], lidar_vfov[1], lidar_vbeams, device=device)
    v_angles = torch.deg2rad(v_angles_deg)
    
    # 生成射线方向
    ray_dirs = []
    for v_angle in v_angles:
        for h_angle in h_angles:
            # 球面坐标转笛卡尔坐标
            x = torch.cos(v_angle) * torch.cos(h_angle)
            y = torch.cos(v_angle) * torch.sin(h_angle)
            z = torch.sin(v_angle)
            ray_dirs.append(torch.stack([x, y, z]))
    
    ray_directions = torch.stack(ray_dirs).reshape(lidar_hbeams, lidar_vbeams, 3)
    
    # 归一化
    ray_directions = ray_directions / (torch.norm(ray_directions, dim=2, keepdim=True) + 1e-8)
    
    return ray_directions


def check_target_in_camera_view(ee_pos, ee_quat, target_pos, camera_fov, camera_max_distance, device='cuda:0'):
    """
    检查目标是否在摄像头视野内
    
    Args:
        ee_pos: [num_envs, 3] 末端执行器位置
        ee_quat: [num_envs, 4] 末端执行器四元数
        target_pos: [num_envs, 3] 目标位置
        camera_fov: float 摄像头视场角（度）
        camera_max_distance: float 摄像头最大检测距离
        device: str 计算设备
        
    Returns:
        in_view: [num_envs] bool tensor，True表示在视野内
    """
    num_envs = ee_pos.shape[0]
    
    # 相对位置
    rel_pos = target_pos - ee_pos
    distance = torch.norm(rel_pos, dim=1)
    
    # 检查距离
    in_range = distance < camera_max_distance
    
    # 归一化相对位置向量
    rel_dir = rel_pos / (distance.unsqueeze(1) + 1e-6)
    
    # 摄像头朝向（局部坐标系中的 +X 方向，即末端正前方）
    camera_dir_local = torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3)
    
    # 转换到世界坐标系
    camera_dir_world = quat_rotate_vector_batch(ee_quat, camera_dir_local)
    
    # 计算夹角
    cos_angle = (camera_dir_world * rel_dir).sum(dim=1)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    
    # 检查是否在视野角度内
    half_fov_rad = torch.deg2rad(torch.tensor(camera_fov / 2.0, device=device))
    in_fov = angle < half_fov_rad
    
    # 综合判断
    in_view = in_range & in_fov
    
    return in_view

