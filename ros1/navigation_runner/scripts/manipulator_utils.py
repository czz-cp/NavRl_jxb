"""
Utility Functions for Manipulator Navigation
机械臂导航的工具函数
"""
import numpy as np
import torch


def ur10e_forward_kinematics(joint_angles):
    """
    UR10e正运动学
    
    Args:
        joint_angles: [6] 关节角度 (弧度)
    
    Returns:
        ee_pos: [3] 末端位置
        ee_rot: [3, 3] 末端旋转矩阵
    """
    # UR10e DH参数
    d = np.array([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
    a = np.array([0, -0.6127, -0.57155, 0, 0, 0])
    alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
    
    # DH变换
    def dh_transform(theta, d, a, alpha):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
    
    # 连乘所有变换
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(joint_angles[i], d[i], a[i], alpha[i])
    
    ee_pos = T[:3, 3]
    ee_rot = T[:3, :3]
    
    return ee_pos, ee_rot


def ur10e_jacobian(joint_angles):
    """
    UR10e雅可比矩阵
    
    Args:
        joint_angles: [6] 关节角度
    
    Returns:
        J: [6, 6] 雅可比矩阵
    """
    # 简化版本：数值微分
    # 完整版本需要解析求导
    
    epsilon = 1e-6
    J = np.zeros((6, 6))
    
    # 当前末端位置
    ee_pos_0, ee_rot_0 = ur10e_forward_kinematics(joint_angles)
    
    for i in range(6):
        # 扰动第i个关节
        joint_angles_perturbed = joint_angles.copy()
        joint_angles_perturbed[i] += epsilon
        
        # 计算扰动后的末端位置
        ee_pos_1, ee_rot_1 = ur10e_forward_kinematics(joint_angles_perturbed)
        
        # 线速度雅可比
        J[:3, i] = (ee_pos_1 - ee_pos_0) / epsilon
        
        # 角速度雅可比（简化）
        # 完整版本需要旋转矩阵的对数映射
        J[3:, i] = 0  # 简化处理
    
    return J


def joint_to_ee_velocity(joint_angles, joint_velocities):
    """
    关节速度 → 末端速度
    
    Args:
        joint_angles: [6] 关节角度
        joint_velocities: [6] 关节速度
    
    Returns:
        ee_velocity: [6] 末端速度 [vx, vy, vz, wx, wy, wz]
    """
    J = ur10e_jacobian(joint_angles)
    ee_velocity = J @ joint_velocities
    return ee_velocity


def ee_to_joint_velocity(joint_angles, ee_velocity):
    """
    末端速度 → 关节速度 (伪逆)
    
    Args:
        joint_angles: [6] 关节角度
        ee_velocity: [6] 末端速度
    
    Returns:
        joint_velocities: [6] 关节速度
    """
    J = ur10e_jacobian(joint_angles)
    
    # 伪逆
    J_pinv = np.linalg.pinv(J)
    joint_velocities = J_pinv @ ee_velocity
    
    return joint_velocities


def check_workspace_limit(ee_pos, workspace_min, workspace_max):
    """
    检查工作空间限制
    
    Args:
        ee_pos: [3] 末端位置
        workspace_min: [3] 工作空间下限
        workspace_max: [3] 工作空间上限
    
    Returns:
        in_workspace: bool
    """
    return np.all(ee_pos >= workspace_min) and np.all(ee_pos <= workspace_max)


def quaternion_to_rotation_matrix(quat):
    """
    四元数转旋转矩阵
    
    Args:
        quat: [4] 四元数 [w, x, y, z]
    
    Returns:
        R: [3, 3] 旋转矩阵
    """
    w, x, y, z = quat
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def rotation_matrix_to_quaternion(R):
    """
    旋转矩阵转四元数
    
    Args:
        R: [3, 3] 旋转矩阵
    
    Returns:
        quat: [4] 四元数 [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        # 其他情况...
        w, x, y, z = 1, 0, 0, 0
    
    return np.array([w, x, y, z])


def depth_image_to_pointcloud(depth_image, intrinsics, depth_scale=1000.0):
    """
    深度图转点云
    
    Args:
        depth_image: [H, W] 深度图
        intrinsics: [4] [fx, fy, cx, cy]
        depth_scale: 深度缩放因子
    
    Returns:
        points: [N, 3] 点云
    """
    fx, fy, cx, cy = intrinsics
    h, w = depth_image.shape
    
    points = []
    for v in range(h):
        for u in range(w):
            depth = depth_image[v, u] / depth_scale
            if depth > 0.1 and depth < 3.0:  # 有效深度
                x = (u - cx) * depth / fx
                y = (v - cy) * depth / fy
                z = depth
                points.append([x, y, z])
    
    return np.array(points)


class LowPassFilter:
    """
    低通滤波器 - 平滑速度命令
    """
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: 平滑系数 (0-1)，越小越平滑
        """
        self.alpha = alpha
        self.prev_value = None
    
    def filter(self, value):
        """
        Args:
            value: 当前值
        
        Returns:
            filtered_value: 滤波后的值
        """
        if self.prev_value is None:
            self.prev_value = value
            return value
        
        filtered = self.alpha * value + (1 - self.alpha) * self.prev_value
        self.prev_value = filtered
        return filtered
    
    def reset(self):
        """重置滤波器"""
        self.prev_value = None


class SafetyMonitor:
    """
    安全监控器
    """
    def __init__(self, config):
        self.min_obstacle_dist = config.get('min_obstacle_distance', 0.15)
        self.emergency_stop_dist = config.get('emergency_stop_distance', 0.08)
        self.max_linear_vel = config.get('max_linear_vel', 0.3)
        self.max_angular_vel = config.get('max_angular_vel', 0.5)
    
    def check_safety(self, ee_pos, depth_points, joint_pos):
        """
        检查安全性
        
        Returns:
            is_safe: bool
            warning_msg: str
        """
        # 1. 深度检查
        if len(depth_points) > 0:
            min_depth = np.min(np.linalg.norm(depth_points - ee_pos, axis=1))
            
            if min_depth < self.emergency_stop_dist:
                return False, f"EMERGENCY: obstacle at {min_depth:.3f}m"
            
            if min_depth < self.min_obstacle_dist:
                return True, f"WARNING: obstacle at {min_depth:.3f}m"
        
        # 2. 关节限位（简化检查）
        if np.any(np.abs(joint_pos) > 6.0):
            return False, "EMERGENCY: joint limit"
        
        return True, "OK"
    
    def limit_velocity(self, ee_vel):
        """
        限制速度
        
        Args:
            ee_vel: [6] 末端速度
        
        Returns:
            ee_vel_limited: [6] 限制后的速度
        """
        ee_vel_limited = ee_vel.copy()
        
        # 线速度限制
        linear_vel_norm = np.linalg.norm(ee_vel[:3])
        if linear_vel_norm > self.max_linear_vel:
            ee_vel_limited[:3] = ee_vel[:3] / linear_vel_norm * self.max_linear_vel
        
        # 角速度限制
        angular_vel_norm = np.linalg.norm(ee_vel[3:])
        if angular_vel_norm > self.max_angular_vel:
            ee_vel_limited[3:] = ee_vel[3:] / angular_vel_norm * self.max_angular_vel
        
        return ee_vel_limited


def vec_to_new_frame_np(vec, goal_direction):
    """
    将向量转换到新坐标系 (NumPy版本)
    
    Args:
        vec: [3] 或 [N, 3]
        goal_direction: [3] 新坐标系的x方向
    
    Returns:
        vec_new: 转换后的向量
    """
    if vec.ndim == 1:
        vec = vec.reshape(1, 3)
    
    # 归一化目标方向
    goal_x = goal_direction / np.linalg.norm(goal_direction)
    
    # z方向
    z_direction = np.array([0, 0, 1])
    
    # y方向（叉乘）
    goal_y = np.cross(z_direction, goal_x)
    goal_y = goal_y / np.linalg.norm(goal_y)
    
    # z方向（叉乘）
    goal_z = np.cross(goal_x, goal_y)
    goal_z = goal_z / np.linalg.norm(goal_z)
    
    # 投影
    vec_x_new = vec @ goal_x
    vec_y_new = vec @ goal_y
    vec_z_new = vec @ goal_z
    
    vec_new = np.stack([vec_x_new, vec_y_new, vec_z_new], axis=-1)
    
    return vec_new


def compute_path_length(waypoints):
    """
    计算路径长度
    
    Args:
        waypoints: [N, 3] 路径点
    
    Returns:
        length: float 总长度
    """
    if len(waypoints) < 2:
        return 0.0
    
    diffs = np.diff(waypoints, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(lengths)


class TrajectoryBuffer:
    """
    轨迹缓冲器 - 用于可视化和分析
    """
    def __init__(self, max_length=500):
        self.max_length = max_length
        self.positions = []
        self.timestamps = []
    
    def add(self, position, timestamp):
        """添加位置点"""
        self.positions.append(position.copy())
        self.timestamps.append(timestamp)
        
        if len(self.positions) > self.max_length:
            self.positions.pop(0)
            self.timestamps.pop(0)
    
    def get_trajectory(self):
        """获取完整轨迹"""
        return np.array(self.positions), np.array(self.timestamps)
    
    def get_length(self):
        """获取轨迹长度"""
        return compute_path_length(np.array(self.positions))
    
    def clear(self):
        """清空缓冲"""
        self.positions = []
        self.timestamps = []


def interpolate_pose(pose_start, pose_end, alpha):
    """
    位姿插值（用于轨迹生成）
    
    Args:
        pose_start: [7] [x, y, z, qw, qx, qy, qz]
        pose_end: [7] [x, y, z, qw, qx, qy, qz]
        alpha: float [0, 1] 插值比例
    
    Returns:
        pose_interp: [7] 插值后的位姿
    """
    # 位置线性插值
    pos_interp = (1 - alpha) * pose_start[:3] + alpha * pose_end[:3]
    
    # 四元数球面插值 (SLERP)
    quat_start = pose_start[3:]
    quat_end = pose_end[3:]
    
    # 简化版本：归一化线性插值 (NLERP)
    quat_interp = (1 - alpha) * quat_start + alpha * quat_end
    quat_interp = quat_interp / np.linalg.norm(quat_interp)
    
    return np.concatenate([pos_interp, quat_interp])


# ========== PyTorch版本（用于训练） ==========

def compute_manipulability(joint_pos):
    """
    计算可操作度（用于奖励）
    
    Args:
        joint_pos: torch.Tensor [N, 6]
    
    Returns:
        manipulability: torch.Tensor [N]
    """
    # 简化版本：距离奇异配置的距离
    # 完整版本需要雅可比行列式
    
    # 避免完全伸直（奇异配置）
    extended = (joint_pos[:, 1].abs() < 0.1) & (joint_pos[:, 2].abs() < 0.1)
    manipulability = torch.ones(joint_pos.size(0), device=joint_pos.device)
    manipulability[extended] = 0.1
    
    return manipulability


def collision_check_point_to_box(point, box_center, box_size):
    """
    点到包围盒的碰撞检测
    
    Args:
        point: [3] 或 [N, 3]
        box_center: [3]
        box_size: [3]
    
    Returns:
        collision: bool 或 [N]
    """
    if isinstance(point, torch.Tensor):
        half_size = box_size / 2
        lower = box_center - half_size
        upper = box_center + half_size
        
        in_box = (
            (point[..., 0] >= lower[0]) & (point[..., 0] <= upper[0]) &
            (point[..., 1] >= lower[1]) & (point[..., 1] <= upper[1]) &
            (point[..., 2] >= lower[2]) & (point[..., 2] <= upper[2])
        )
        return in_box
    else:
        point = np.array(point)
        box_center = np.array(box_center)
        box_size = np.array(box_size)
        
        half_size = box_size / 2
        lower = box_center - half_size
        upper = box_center + half_size
        
        return np.all(point >= lower) and np.all(point <= upper)


def sample_collision_free_position(workspace_min, workspace_max, obstacles, 
                                   min_distance=0.2, max_attempts=100):
    """
    采样无碰撞位置
    
    Args:
        workspace_min: [3]
        workspace_max: [3]
        obstacles: List of (center, size)
        min_distance: 最小距离
        max_attempts: 最大尝试次数
    
    Returns:
        position: [3] 或 None (如果失败)
    """
    for _ in range(max_attempts):
        # 随机采样
        pos = workspace_min + np.random.rand(3) * (workspace_max - workspace_min)
        
        # 检查与所有障碍物的距离
        collision_free = True
        for obs_center, obs_size in obstacles:
            dist = np.linalg.norm(pos - obs_center)
            if dist < (np.max(obs_size) / 2 + min_distance):
                collision_free = False
                break
        
        if collision_free:
            return pos
    
    return None


def create_depth_image_from_points(points, ee_pos, intrinsics, image_size):
    """
    从点云创建深度图（用于可视化）
    
    Args:
        points: [N, 3] 点云（世界坐标）
        ee_pos: [3] 末端位置
        intrinsics: [4] [fx, fy, cx, cy]
        image_size: [2] [width, height]
    
    Returns:
        depth_image: [H, W] 深度图
    """
    fx, fy, cx, cy = intrinsics
    w, h = image_size
    
    depth_image = np.zeros((h, w), dtype=np.float32)
    
    # 转换到相机坐标系（简化：假设相机朝向+z）
    points_cam = points - ee_pos
    
    for point in points_cam:
        if point[2] > 0.1:  # 有效深度
            # 投影到图像平面
            u = int(fx * point[0] / point[2] + cx)
            v = int(fy * point[1] / point[2] + cy)
            
            if 0 <= u < w and 0 <= v < h:
                depth_image[v, u] = point[2]
    
    return depth_image

