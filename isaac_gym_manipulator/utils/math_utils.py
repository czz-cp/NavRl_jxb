"""
数学工具函数模块
包含四元数操作、位姿误差计算等通用数学方法
"""

import torch
import numpy as np


def quat_rotate_vector_batch(quat, vec):
    """
    批量四元数旋转向量
    
    Args:
        quat: [num_envs, 4] - 四元数 [x, y, z, w]
        vec: [num_envs, 3] - 向量
        
    Returns:
        [num_envs, 3] - 旋转后的向量
    """
    # 提取四元数分量
    quat_w = quat[:, 3:4]
    quat_xyz = quat[:, :3]
    
    # 四元数旋转公式: v' = v + 2*w*(q×v) + 2*(q×(q×v))
    uv = torch.cross(quat_xyz, vec, dim=1)
    uuv = torch.cross(quat_xyz, uv, dim=1)
    
    return vec + 2.0 * (quat_w * uv + uuv)


def quat_rotate_vector(quat, vec):
    """
    单个四元数旋转向量（NumPy版本）
    
    Args:
        quat: [4] - 四元数 [x, y, z, w]
        vec: [3] - 向量
        
    Returns:
        [3] - 旋转后的向量
    """
    quat_w = quat[3]
    quat_xyz = quat[:3]
    
    # 叉积
    uv = np.cross(quat_xyz, vec)
    uuv = np.cross(quat_xyz, uv)
    
    return vec + 2.0 * (quat_w * uv + uuv)


def compute_pose_error_three_point(ee_pos, ee_quat, target_pos, target_quat=None, d=0.1):
    """
    A.2 位姿误差计算 - 三点表示法
    
    通过在三个主轴上各取一个参考点，全面衡量位置和方向误差
    
    Args:
        ee_pos: [num_envs, 3] 末端执行器位置
        ee_quat: [num_envs, 4] 末端执行器四元数 [x,y,z,w]
        target_pos: [num_envs, 3] 目标位置
        target_quat: [num_envs, 4] 目标四元数（可选）
        d: float 参考点距离
        
    Returns:
        pose_error: [num_envs] 综合位姿误差
        
    原理：
        1. 提取三个旋转轴方向（Roll, Pitch, Yaw）
        2. 在每个轴方向上生成参考点（距离d）
        3. 计算三个参考点到目标的距离
        4. 如果有目标姿态，还计算角度误差
    """
    device = ee_pos.device
    num_envs = ee_pos.shape[0]
    
    # 1. 提取末端执行器的三个旋转轴方向
    x_axis_ee = quat_rotate_vector_batch(ee_quat, 
        torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3))
    y_axis_ee = quat_rotate_vector_batch(ee_quat,
        torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3))
    z_axis_ee = quat_rotate_vector_batch(ee_quat,
        torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(num_envs, 3))
    
    # 2. 生成三个参考点
    p_roll_ee = ee_pos + d * x_axis_ee
    p_pitch_ee = ee_pos + d * y_axis_ee
    p_yaw_ee = ee_pos + d * z_axis_ee
    
    # 3. 计算误差
    if target_quat is not None:
        # 有目标姿态：计算完整的位姿误差
        x_axis_target = quat_rotate_vector_batch(target_quat,
            torch.tensor([1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3))
        y_axis_target = quat_rotate_vector_batch(target_quat,
            torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(num_envs, 3))
        z_axis_target = quat_rotate_vector_batch(target_quat,
            torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(num_envs, 3))
        
        p_roll_target = target_pos + d * x_axis_target
        p_pitch_target = target_pos + d * y_axis_target
        p_yaw_target = target_pos + d * z_axis_target
        
        # 计算三个轴的距离
        d_roll = torch.norm(p_roll_ee - p_roll_target, dim=1)
        d_pitch = torch.norm(p_pitch_ee - p_pitch_target, dim=1)
        d_yaw = torch.norm(p_yaw_ee - p_yaw_target, dim=1)
        
        position_error = d_roll + d_pitch + d_yaw
        
        # 方向误差（四元数角度差）
        dot_product = (ee_quat * target_quat).sum(dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle_error = 2 * torch.acos(torch.abs(dot_product))
        
        pose_error = position_error + angle_error
    else:
        # 只有位置目标：使用简化的三点法
        d_roll = torch.norm(p_roll_ee - target_pos, dim=1)
        d_pitch = torch.norm(p_pitch_ee - target_pos, dim=1)
        d_yaw = torch.norm(p_yaw_ee - target_pos, dim=1)
        
        pose_error = d_roll + d_pitch + d_yaw
    
    return pose_error


def compute_point_to_segment_distance(point, seg_start, seg_end):
    """
    计算点到线段的最短距离
    
    Args:
        point: [num_envs, 3] 点位置
        seg_start: [num_envs, 3] 线段起点
        seg_end: [num_envs, 3] 线段终点
        
    Returns:
        distance: [num_envs] 最短距离
        
    数学原理：
        线段参数化：P(t) = P₁ + t·v, t ∈ [0,1]
        最优参数：t = (v·(Q-P₁)) / (v·v)
        最短距离：||Q - P(t)||
    """
    # 线段方向向量
    v = seg_end - seg_start  # [num_envs, 3]
    v_dot_v = (v * v).sum(dim=1, keepdim=True)  # [num_envs, 1]
    
    # 计算最优参数 t
    point_minus_start = point - seg_start  # [num_envs, 3]
    t = (v * point_minus_start).sum(dim=1, keepdim=True) / (v_dot_v + 1e-8)  # [num_envs, 1]
    
    # 限制 t 在 [0, 1] 范围内（线段而非直线）
    t = torch.clamp(t, 0.0, 1.0)
    
    # 计算线段上最近点
    closest_point = seg_start + t * v  # [num_envs, 3]
    
    # 计算距离
    distance = torch.norm(point - closest_point, dim=1)  # [num_envs]
    
    return distance


def quat_conjugate(quat):
    """
    四元数共轭
    
    Args:
        quat: [num_envs, 4] - 四元数 [x, y, z, w]
        
    Returns:
        [num_envs, 4] - 共轭四元数 [-x, -y, -z, w]
    """
    return torch.cat([-quat[:, :3], quat[:, 3:4]], dim=1)


def quat_multiply(q1, q2):
    """
    四元数乘法
    
    Args:
        q1, q2: [num_envs, 4] - 四元数 [x, y, z, w]
        
    Returns:
        [num_envs, 4] - 乘积四元数
    """
    w1, x1, y1, z1 = q1[:, 3], q1[:, 0], q1[:, 1], q1[:, 2]
    w2, x2, y2, z2 = q2[:, 3], q2[:, 0], q2[:, 1], q2[:, 2]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=1)


def euler_to_quat(roll, pitch, yaw):
    """
    欧拉角转四元数（ZYX顺序）
    
    Args:
        roll, pitch, yaw: [num_envs] - 欧拉角（弧度）
        
    Returns:
        [num_envs, 4] - 四元数 [x, y, z, w]
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([x, y, z, w], dim=1)

