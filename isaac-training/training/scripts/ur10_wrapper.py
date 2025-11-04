"""
UR10e Manipulator Wrapper for Isaac Sim
为 Isaac Sim 的 Articulation 提供高层次的机械臂接口
"""
import torch
import numpy as np
from omni.isaac.orbit.assets.articulation import Articulation
from typing import Optional, Tuple
from pxr import Gf, UsdPhysics


class UR10ManipulatorWrapper:
    """
    将 Articulation API 封装为更易用的接口
    提供类似于 SingleArmManipulator 的方法
    """
    
    def __init__(self, articulation: Articulation):
        """
        初始化包装器
        
        Args:
            articulation: Articulation 实例
        """
        self.articulation = articulation
        self.device = articulation.device
        
        # 末端执行器 link 名称（根据 UR10 的实际模型）
        self.ee_link_name = "ee_link"  # 默认名称
        
        # 检查是否存在 ee_link
        try:
            self._find_ee_link()
        except Exception as e:
            print(f"[Warning] Could not find ee_link: {e}")
            print(f"[Info] Using default link for end effector")
    
    def _find_ee_link(self):
        """尝试找到末端执行器 link"""
        # UR10 常见的末端执行器名称
        possible_names = ["ee_link", "ee_link_0", "wrist_3_link", "tool0"]
        
        for name in possible_names:
            try:
                self.articulation.find_bodies(name)
                self.ee_link_name = name
                return
            except:
                continue
    
    def initialize(self):
        """初始化机械臂"""
        if hasattr(self.articulation, 'initialize'):
            self.articulation.initialize()
    
    def set_joint_positions(self, positions: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """
        设置关节位置
        
        Args:
            positions: (num_envs, num_joints) 关节位置
            env_ids: 环境 ID（可选）
        """
        # 使用 Articulation API
        self.articulation.set_joint_position_target(positions)
        self.articulation.write_data_to_sim()
    
    def set_joint_velocities(self, velocities: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        """
        设置关节速度
        
        Args:
            velocities: (num_envs, num_joints) 关节速度
            env_ids: 环境 ID（可选）
        """
        self.articulation.set_joint_velocity_target(velocities)
        self.articulation.write_data_to_sim()
    
    def get_joint_positions(self) -> torch.Tensor:
        """
        获取关节位置
        
        Returns:
            (num_envs, num_joints) 关节位置
        """
        return self.articulation.data.joint_pos
    
    def get_ee_state(self) -> torch.Tensor:
        """
        获取末端执行器状态
        
        Returns:
            (num_envs, 13) 张量: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
        """
        # 尝试获取末端执行器 link 的状态
        try:
            # 获取 link 的世界坐标系状态
            link_state = self.articulation.data.body_state_w
            
            # 找到 ee_link 的索引
            try:
                ee_body_ids, ee_body_names = self.articulation.find_bodies(self.ee_link_name)
                if len(ee_body_ids) > 0:
                    ee_idx = ee_body_ids[0]
                    ee_state_w = link_state[:, ee_idx]
                else:
                    # 使用最后一个 link 作为末端
                    ee_state_w = link_state[:, -1]
            except:
                # 如果找不到，使用最后一个 body
                ee_state_w = link_state[:, -1]
            
            # 提取位置和四元数
            ee_pos = ee_state_w[:, :3]  # 位置
            ee_quat = ee_state_w[:, 3:7]  # 四元数 (w, x, y, z)
            
        except Exception as e:
            # 如果获取失败，使用基于关节位置的前向运动学
            print(f"[Warning] Failed to get link state: {e}")
            return self._compute_ee_state_from_joints()
        
        # 末端速度（简化：使用关节速度计算）
        # 实际应该使用雅可比矩阵
        joint_vel = self.articulation.data.joint_vel
        
        # 简化的速度计算
        ee_vel = self._compute_ee_velocity(joint_vel)
        
        # 组合状态
        ee_state = torch.cat([
            ee_pos,      # 3
            ee_quat,     # 4
            ee_vel       # 6
        ], dim=-1)  # 总共 13 维
        
        return ee_state
    
    def _compute_ee_state_from_joints(self) -> torch.Tensor:
        """基于关节位置计算末端状态（简化版前向运动学）"""
        joint_pos = self.get_joint_positions()
        
        # UR10 DH 参数
        l1, l2, l3 = 0.089159, 0.425, 0.39225  # 前三个连杆
        
        # 简化的前向运动学（只考虑前 3 个关节）
        pos_x = torch.sin(joint_pos[:, 0]) * (l2 * torch.cos(joint_pos[:, 1]) + l3 * torch.cos(joint_pos[:, 1] + joint_pos[:, 2]))
        pos_y = -torch.cos(joint_pos[:, 0]) * (l2 * torch.cos(joint_pos[:, 1]) + l3 * torch.cos(joint_pos[:, 1] + joint_pos[:, 2]))
        pos_z = l1 + l2 * torch.sin(joint_pos[:, 1]) + l3 * torch.sin(joint_pos[:, 1] + joint_pos[:, 2])
        
        ee_pos = torch.stack([pos_x, pos_y, pos_z], dim=-1)
        
        # 简化的四元数（基于末端关节）
        ee_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(ee_pos.shape[0], 4)
        
        # 末端速度（简化为零）
        ee_vel = torch.zeros(ee_pos.shape[0], 6, device=self.device)
        
        ee_state = torch.cat([ee_pos, ee_quat, ee_vel], dim=-1)
        return ee_state
    
    def _compute_ee_velocity(self, joint_vel: torch.Tensor) -> torch.Tensor:
        """计算末端执行器速度"""
        # 简化的速度计算
        # 实际应该使用雅可比矩阵: ee_vel = J * joint_vel
        
        # 这里使用简化的线性映射
        # 实际实现需要完整的雅可比矩阵
        ee_vel = torch.zeros(joint_vel.shape[0], 6, device=joint_vel.device)
        
        # 简化的线性速度（比例缩放）
        ee_vel[:, 0] = joint_vel[:, 0] * 0.3  # X
        ee_vel[:, 1] = joint_vel[:, 1] * 0.3  # Y
        ee_vel[:, 2] = joint_vel[:, 2] * 0.3  # Z
        
        # 简化的角速度
        ee_vel[:, 3] = joint_vel[:, 3] * 0.5  # wx
        ee_vel[:, 4] = joint_vel[:, 4] * 0.5  # wy
        ee_vel[:, 5] = joint_vel[:, 5] * 0.5  # wz
        
        return ee_vel
    
    def apply_ee_velocity(self, ee_velocity: torch.Tensor):
        """
        应用末端执行器速度控制
        
        Args:
            ee_velocity: (num_envs, 6) 末端速度 [vx, vy, vz, wx, wy, wz]
        """
        # 这里需要实现速度逆运动学
        # 使用雅可比矩阵伪逆: joint_vel = J^+ * ee_vel
        
        # 简化实现：直接映射到关节空间
        joint_vel = self._ee_velocity_to_joint_velocity(ee_velocity)
        
        # 设置关节速度目标
        self.articulation.set_joint_velocity_target(joint_vel)
        self.articulation.write_data_to_sim()
    
    def _ee_velocity_to_joint_velocity(self, ee_vel: torch.Tensor) -> torch.Tensor:
        """
        将末端速度转换为关节速度
        
        使用简化的逆雅可比矩阵
        """
        num_envs = ee_vel.shape[0]
        num_joints = 6
        
        # 简化的逆映射（实际应该使用雅可比矩阵的伪逆）
        joint_vel = torch.zeros(num_envs, num_joints, device=ee_vel.device)
        
        # 线性速度到关节速度的映射（简化）
        joint_vel[:, 0] = ee_vel[:, 0] * 3.0  # X
        joint_vel[:, 1] = ee_vel[:, 1] * 3.0  # Y
        joint_vel[:, 2] = ee_vel[:, 2] * 3.0  # Z
        
        # 角速度到关节速度的映射
        joint_vel[:, 3] = ee_vel[:, 3] * 2.0  # wx
        joint_vel[:, 4] = ee_vel[:, 4] * 2.0  # wy
        joint_vel[:, 5] = ee_vel[:, 5] * 2.0  # wz
        
        return joint_vel
