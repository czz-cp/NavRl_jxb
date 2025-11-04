"""
UR10e Manipulator Navigation Environment for Isaac Gym
Isaac Gym 版本的 UR10e 机械臂导航环境
"""
# IMPORTANT: Isaac Gym must be imported before PyTorch
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *

import torch
import torch.nn as nn
import math
import numpy as np

# 导入工具模块
from utils import (
    # 数学工具
    quat_rotate_vector_batch,
    quat_rotate_vector,
    compute_pose_error_three_point,
    # 碰撞检测
    compute_link_obstacle_distances,
    detect_collision_multisource,
    compute_collision_penalty,
    # 奖励函数
    compute_downward_search_reward,
    compute_systematic_scan_reward,
    compute_active_exploration_reward,
    compute_static_penalty,
    compute_z_exploration_reward,
    # 传感器
    check_target_in_camera_view,
)


class ManipulatorEnvGym:
    """
    UR10e 机械臂导航环境 - Isaac Gym 版本
    
    特性:
    - 6自由度UR10e机械臂
    - 末端速度控制 [vx, vy, vz, wx, wy, wz]
    - 深度相机观测（模拟Realsense D435i）
    - 静态和动态障碍物
    - 动态ArUco目标
    - 两阶段任务：探索 + 导航
    - 基于视觉的目标检测
    """
    
    def __init__(self, cfg):
        # 保存配置
        self.cfg = cfg
        self.num_envs = cfg.env.num_envs
        self.max_episode_length = cfg.env.max_episode_length
        
        # 设备
        device_str = cfg.device if hasattr(cfg, 'device') else "cuda:0"
        self.device = torch.device(device_str)
        
        # 图形设备：如果禁用可视化，必须设为 -1
        if hasattr(cfg, 'visualization') and hasattr(cfg.visualization, 'enable') and cfg.visualization.enable:
            self.graphics_device = cfg.graphics.graphics_device_id if hasattr(cfg, 'graphics') else 0
        else:
            self.graphics_device = -1
        
        # 可视化
        self.enable_viewer = cfg.visualization.enable if hasattr(cfg, 'visualization') else False
        self.viewer = None
        self.draw_lidar = cfg.visualization.draw_lidar if hasattr(cfg, 'visualization') and hasattr(cfg.visualization, 'draw_lidar') else False
        self.draw_camera_fov = cfg.visualization.draw_camera_fov if hasattr(cfg, 'visualization') and hasattr(cfg.visualization, 'draw_camera_fov') else False
        
        # LiDAR 传感器参数 (模仿无人机)
        if hasattr(cfg, 'sensor'):
            self.lidar_range = cfg.sensor.lidar_range
            self.lidar_vfov = tuple(cfg.sensor.lidar_vfov)  # (min_angle, max_angle)
            self.lidar_vbeams = cfg.sensor.lidar_vbeams
            self.lidar_hres = cfg.sensor.lidar_hres
            self.lidar_hbeams = int(360 / self.lidar_hres)  # 水平光束数
            self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)
            print(f"[LiDAR] 配置: range={self.lidar_range}m, "
                  f"h_beams={self.lidar_hbeams}, v_beams={self.lidar_vbeams}")
        else:
            # 默认 LiDAR 参数
            self.lidar_range = 2.0
            self.lidar_vfov = (-30.0, 30.0)
            self.lidar_vbeams = 4
            self.lidar_hres = 10.0
            self.lidar_hbeams = 36
            self.lidar_resolution = (36, 4)
            print("[LiDAR] 使用默认配置")
        
        # 障碍物生成参数（在 _create_envs 之前设置）
        self.obstacle_min_spacing = cfg.env.obstacle_min_spacing if hasattr(cfg.env, 'obstacle_min_spacing') else 0.4
        self.obstacle_size = cfg.env.obstacle_size if hasattr(cfg.env, 'obstacle_size') else 0.2
        self.obstacle_size_min = cfg.env.obstacle_size_min if hasattr(cfg.env, 'obstacle_size_min') else 0.15
        self.obstacle_size_max = cfg.env.obstacle_size_max if hasattr(cfg.env, 'obstacle_size_max') else 0.35
        self.obstacle_height_min = cfg.env.obstacle_height_min if hasattr(cfg.env, 'obstacle_height_min') else 0.0
        self.obstacle_height_max = cfg.env.obstacle_height_max if hasattr(cfg.env, 'obstacle_height_max') else 0.8
        self.obstacle_radius_min = cfg.env.obstacle_radius_min if hasattr(cfg.env, 'obstacle_radius_min') else 0.6
        self.obstacle_radius_max = cfg.env.obstacle_radius_max if hasattr(cfg.env, 'obstacle_radius_max') else 1.2
        
        # 机械臂安全区域参数
        self.arm_safe_radius = cfg.env.arm_safe_radius if hasattr(cfg.env, 'arm_safe_radius') else 0.6
        self.arm_safe_height = cfg.env.arm_safe_height if hasattr(cfg.env, 'arm_safe_height') else 0.8
        self.arm_safety_buffer = cfg.env.arm_safety_buffer if hasattr(cfg.env, 'arm_safety_buffer') else 0.1
        
        # 终止条件配置
        self.terminate_on_collision = cfg.env.terminate_on_collision if hasattr(cfg.env, 'terminate_on_collision') else False
        self.terminate_on_out_of_bounds = cfg.env.terminate_on_out_of_bounds if hasattr(cfg.env, 'terminate_on_out_of_bounds') else False
        self.terminate_on_joint_limits = cfg.env.terminate_on_joint_limits if hasattr(cfg.env, 'terminate_on_joint_limits') else False
        self.collision_distance = cfg.env.collision_distance if hasattr(cfg.env, 'collision_distance') else 0.15
        self.workspace_limit = cfg.env.workspace_limit if hasattr(cfg.env, 'workspace_limit') else 2.0
        self.success_distance = cfg.env.success_distance if hasattr(cfg.env, 'success_distance') else 0.1
        
        print(f"[Obstacles] 间距: {self.obstacle_min_spacing}m, 尺寸: {self.obstacle_size_min}-{self.obstacle_size_max}m")
        print(f"[Obstacles] 半径范围: {self.obstacle_radius_min}-{self.obstacle_radius_max}m, 高度: {self.obstacle_height_min}-{self.obstacle_height_max}m")
        print(f"[Arm Safety] 安全半径: {self.arm_safe_radius}m, 安全高度: {self.arm_safe_height}m, 缓冲区: {self.arm_safety_buffer}m")
        print(f"[Termination] 碰撞: {self.terminate_on_collision}, 越界: {self.terminate_on_out_of_bounds}, "
              f"关节限位: {self.terminate_on_joint_limits}")
        
        # 创建环境
        self.gym = gymapi.acquire_gym()
        
        # 创建模拟
        self._create_sim()
        
        # 创建环境（此时只收集数据到 obstacle_sizes_temp）
        self._create_envs()
        
        # 先初始化基础缓冲区（不包含障碍物）
        self._init_basic_buffers()
        
        # 设置机械臂初始位置
        print("[Obstacles] 初始化机械臂姿态...")
        self._set_initial_arm_pose()
        
        # 创建障碍物
        print("[Obstacles] 根据机械臂实际位置创建障碍物...")
        self._create_obstacles_after_init()
        
        # 重新初始化完整的缓冲区（包含障碍物）
        print("[Buffers] 重新初始化状态缓冲区...")
        self._refresh_all_buffers()
        
        # 创建视口 (如果启用可视化)
        if self.enable_viewer:
            self._create_viewer()
        
        # 初始化调试绘制
        if self.enable_viewer and (self.draw_lidar or self.draw_camera_fov):
            self._init_debug_viz()
        
        print(f"[Isaac Gym] 环境创建完成: {self.num_envs} 个环境")
    
    def _create_sim(self):
        """创建仿真"""
        # 创建仿真参数
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81
        
        # 使用 PhysX 引擎
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False
        
        # 计算环境空间
        env_lower = gymapi.Vec3(-2.0, -2.0, -0.5)
        env_upper = gymapi.Vec3(2.0, 2.0, 1.5)
        
        # 创建仿真
        sim_device_id = self.cfg.sim.device_id if hasattr(self.cfg, 'sim') else 0
        self.sim = self.gym.create_sim(
            sim_device_id,
            self.graphics_device,
            gymapi.SIM_PHYSX,
            sim_params
        )
        
        if self.sim is None:
            raise RuntimeError("Failed to create simulation")
        
        # 添加地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self):
        """创建所有环境"""
        # 导入 UR10 资产
        asset_root = self.cfg.assets.asset_root
        ur10_asset_file = self.cfg.assets.ur10_asset_file
        
        # 加载 UR10 资产
        ur10_options = gymapi.AssetOptions()
        ur10_options.fix_base_link = True
        ur10_options.flip_visual_attachments = True  # 启用以正确显示 mesh
        ur10_options.use_mesh_materials = True  # 启用材质
        ur10_options.override_com = True
        ur10_options.override_inertia = True
        ur10_options.vhacd_enabled = True  # 启用 VHACD 以处理凸包碰撞
        ur10_options.vhacd_params = gymapi.VhacdParams()
        ur10_options.vhacd_params.resolution = 300000
        ur10_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
        
        try:
            self.ur10_asset = self.gym.load_asset(
                self.sim, asset_root, ur10_asset_file, ur10_options
            )
        except Exception as e:
            print(f"[Error] 无法加载资产: {e}")
            print(f"[Info] 资产路径: {asset_root}/{ur10_asset_file}")
            raise
        
        # 获取DOF信息
        self.num_dofs = self.gym.get_asset_dof_count(self.ur10_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.ur10_asset)
        
        if self.num_dofs == 0:
            raise RuntimeError(f"加载的资产没有 DOF: {asset_root}/{ur10_asset_file}")
        
        # 查找末端执行器 body 索引
        self.ee_body_idx = None
        for i in range(self.num_bodies):
            body_name = self.gym.get_asset_rigid_body_name(self.ur10_asset, i)
            if "wrist" in body_name.lower() or "ee" in body_name.lower() or "tool" in body_name.lower():
                self.ee_body_idx = i
                print(f"[Camera] 找到末端执行器 body: {body_name} (索引: {i})")
                break
        
        if self.ee_body_idx is None:
            self.ee_body_idx = self.num_bodies - 1  # 使用最后一个 body
            print(f"[Camera] 使用最后一个 body 作为末端执行器")
        
        print(f"[Manipulator] DOF数量: {self.num_dofs}, 刚体数量: {self.num_bodies}")
        
        # 创建环境
        env_lower = gymapi.Vec3(-1.5, -1.5, 0.0)
        env_upper = gymapi.Vec3(1.5, 1.5, 0.0)
        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        
        self.envs = []
        self.ur10_handles = []
        self.aruco_handles = []  # ArUco 标记句柄
        self.obstacle_handles = []  # 障碍物句柄 (用于 LiDAR 检测)
        
        for i in range(self.num_envs):
            # 创建环境
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            
            # 创建UR10
            pos = gymapi.Vec3(0.0, 0.0, 0.0)
            pose = gymapi.Transform(pos, gymapi.Quat(0.0, 0.0, 0.0, 1.0))
            
            ur10_handle = self.gym.create_actor(
                env, self.ur10_asset, pose, "UR10", i, 0, 0
            )
            self.ur10_handles.append(ur10_handle)
            
            # 配置DOF属性
            dof_props = self.gym.get_actor_dof_properties(env, ur10_handle)
            
            # 在第一个环境中保存关节限位信息
            if i == 0:
                self.dof_lower_limits = torch.zeros(self.num_dofs, device=self.device)
                self.dof_upper_limits = torch.zeros(self.num_dofs, device=self.device)
                for j in range(self.num_dofs):
                    self.dof_lower_limits[j] = float(dof_props['lower'][j])
                    self.dof_upper_limits[j] = float(dof_props['upper'][j])
                print(f"[DOF Limits] Lower: {self.dof_lower_limits}")
                print(f"[DOF Limits] Upper: {self.dof_upper_limits}")
            
            for j in range(self.num_dofs):
                dof_props['stiffness'][j] = 1000.0
                dof_props['damping'][j] = 100.0
                dof_props['upper'][j] = math.pi
                dof_props['lower'][j] = -math.pi
            self.gym.set_actor_dof_properties(env, ur10_handle, dof_props)
            
            # 添加 ArUco 标记 (作为目标)
            # ArUco 标记是一个正方形，0.1m x 0.1m
            aruco_options = gymapi.AssetOptions()
            aruco_options.fix_base_link = True
            aruco_options.disable_gravity = True
            # 创建 ArUco 标记资产
            aruco_asset = self.gym.create_box(self.sim, 0.1, 0.1, 0.001, aruco_options)
            
            # ArUco 位置将在 reset 时设置
            aruco_pos = gymapi.Vec3(0.0, 0.5, 0.3)  # 临时位置
            aruco_rot = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            aruco_pose = gymapi.Transform(aruco_pos, aruco_rot)
            
            aruco_handle = self.gym.create_actor(
                env, aruco_asset, aruco_pose, f"aruco_{i}", i, 0
            )
            self.aruco_handles.append(aruco_handle)
            
            # 设置 ArUco 颜色 (黑色方块代表 ArUco，黑色边框白色内部)
            self.gym.set_rigid_body_color(env, aruco_handle, 0, gymapi.MESH_VISUAL,
                                        gymapi.Vec3(0.2, 0.2, 0.2))  # 深色
            
        # 注意：障碍物将在机械臂初始化后再创建
        # 这样可以根据机械臂的实际初始位置设置安全区域
        self.obstacle_handles = []  # 先初始化为空列表
        
        # 添加相机（如果启用可视化）
        for i in range(self.num_envs):
            env = self.envs[i]
            
            # 添加相机 (眼在手上配置)
            if self.enable_viewer and i == 0:  # 只为第一个环境添加相机
                # 创建相机属性
                camera_props = gymapi.CameraProperties()
                camera_props.width = 1280
                camera_props.height = 720
                camera_props.horizontal_fov = 60.0
                
                # 在机械臂末端添加相机
                camera_handle = self.gym.create_camera_sensor(env, camera_props)
                
                # 相机偏移 (相对于末端执行器)
                camera_offset = gymapi.Vec3(0.08, 0.0, 0.0)  # 在末端前
                # 相机旋转: 绕 Y 轴旋转 90 度，使相机朝前
                angle = np.deg2rad(90)
                camera_rot = gymapi.Quat(
                    np.cos(angle/2),
                    0,
                    np.sin(angle/2),
                    0
                )
                
                # 获取末端执行器 body
                ee_body_handle = self.gym.get_actor_rigid_body_handle(env, ur10_handle, self.ee_body_idx)
                
                self.gym.attach_camera_to_body(
                    camera_handle, env, ee_body_handle,
                    gymapi.Transform(camera_offset, camera_rot),
                    gymapi.FOLLOW_TRANSFORM
                )
                print(f"[Camera] 相机已附加到末端执行器 body {self.ee_body_idx}")
    
    def _set_initial_arm_pose(self):
        """设置机械臂初始姿态"""
        import random
        import torch
        
        # 为每个环境设置初始关节位置
        for i in range(self.num_envs):
            # 随机初始关节位置
            init_pos = torch.zeros(self.num_dofs, device=self.device)
            init_pos[0] = (torch.rand(1, device=self.device) - 0.5) * 0.5
            init_pos[1] = -1.0 + torch.rand(1, device=self.device) * 1.0
            init_pos[2] = (torch.rand(1, device=self.device) - 0.5) * 1.0
            
            # 设置到 DOF 状态
            start_idx = i * self.num_dofs
            end_idx = (i + 1) * self.num_dofs
            self.dof_states[start_idx:end_idx, 0] = init_pos
            self.dof_states[start_idx:end_idx, 1] = 0.0  # 零速度
        
        # 应用到仿真
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
        
        # 运行几步让机械臂稳定在初始位置
        for _ in range(10):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        
        # 刷新状态
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
    
    def _create_obstacles_after_init(self):
        """根据机械臂实际位置创建障碍物"""
        import random
        
        # 获取所有机械臂刚体的位置作为安全区域
        # rigid_body_states shape: [num_envs * num_bodies, 13]
        arm_body_positions = []
        
        for i in range(self.num_envs):
            env_positions = []
            for j in range(self.num_bodies):
                body_idx = i * self.num_bodies + j
                pos = self.rigid_body_states[body_idx, :3].cpu().numpy()
                env_positions.append(pos)
            arm_body_positions.append(env_positions)
        
        # 为每个环境创建障碍物
        for i in range(self.num_envs):
            env = self.envs[i]
            env_obstacles = []
            obstacle_positions = []
            
            box_options = gymapi.AssetOptions()
            box_options.fix_base_link = True
            
            num_obstacles = self.cfg.env.num_static_obstacles if hasattr(self.cfg.env, 'num_static_obstacles') else 3
            max_attempts = 100
            
            for j in range(num_obstacles):
                placed = False
                box_size = random.uniform(self.obstacle_size_min, self.obstacle_size_max)
                box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size * 2, box_options)
                
                for attempt in range(max_attempts):
                    # 随机生成位置
                    radius = random.uniform(self.obstacle_radius_min, self.obstacle_radius_max)
                    angle = random.uniform(0, 2 * np.pi)
                    
                    obs_x = radius * np.cos(angle)
                    obs_y = radius * np.sin(angle)
                    obs_z = random.uniform(self.obstacle_height_min, self.obstacle_height_max)
                    
                    obs_pos_np = np.array([obs_x, obs_y, obs_z])
                    valid = True
                    
                    # 检查与其他障碍物的距离
                    for existing_pos in obstacle_positions:
                        distance = np.linalg.norm(obs_pos_np - existing_pos)
                        if distance < self.obstacle_min_spacing:
                            valid = False
                            break
                    
                    # 检查与机械臂所有刚体的距离（基于实际位置）
                    if valid:
                        for body_pos in arm_body_positions[i]:
                            distance_to_body = np.linalg.norm(obs_pos_np - body_pos)
                            # 安全距离 = 配置的安全缓冲区 + 障碍物半径
                            safe_distance = self.arm_safety_buffer + box_size/2 + 0.2  # 额外20cm余量
                            if distance_to_body < safe_distance:
                                valid = False
                                break
                    
                    if valid:
                        # 创建障碍物
                        obs_pos = gymapi.Vec3(obs_x, obs_y, obs_z)
                        obs_pose = gymapi.Transform(obs_pos, gymapi.Quat(0.0, 0.0, 0.0, 1.0))
                        obs_handle = self.gym.create_actor(env, box_asset, obs_pose, f"obstacle_{i}_{j}", i, 0)
                        
                        # 设置颜色
                        size_ratio = (box_size - self.obstacle_size_min) / (self.obstacle_size_max - self.obstacle_size_min)
                        color_intensity = 0.5 + 0.4 * size_ratio
                        self.gym.set_rigid_body_color(env, obs_handle, 0, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(color_intensity, 0.1, 0.1))
                        
                        env_obstacles.append(obs_handle)
                        obstacle_positions.append(obs_pos_np)
                        placed = True
                        break
                
                if not placed:
                    print(f"[Warning] 环境 {i} 无法放置第 {j+1} 个障碍物（尝试了 {max_attempts} 次）")
            
            self.obstacle_handles.append(env_obstacles)
            
            if i == 0:
                print(f"[Obstacles] 环境 {i} 成功放置 {len(env_obstacles)} 个障碍物")
    
    def _refresh_all_buffers(self):
        """在障碍物创建后，重新初始化所有缓冲区"""
        # 重新获取状态张量（因为障碍物已经创建）
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        
        # 更新访问器
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        
        # 计算每个环境的 actors 数量
        # UR10 + ArUco + num_obstacles
        num_obstacles = self.cfg.env.num_static_obstacles if hasattr(self.cfg.env, 'num_static_obstacles') else 3
        actors_per_env = 2 + num_obstacles  # UR10 + ArUco + 障碍物
        
        # 重新计算索引
        self.ur10_indices = torch.arange(self.num_envs, device=self.device) * actors_per_env
        self.aruco_indices = torch.arange(self.num_envs, device=self.device) * actors_per_env + 1
        
        # 更新 root_states 相关数据
        ur10_indices_cpu = self.ur10_indices.cpu().long()
        self.root_positions = self.root_states[ur10_indices_cpu, 0:3].to(self.device)
        self.root_orientations = self.root_states[ur10_indices_cpu, 3:7].to(self.device)
        self.root_linvels = self.root_states[ur10_indices_cpu, 7:10].to(self.device)
        self.root_angvels = self.root_states[ur10_indices_cpu, 10:13].to(self.device)
        
        # 更新 ArUco 位置
        aruco_indices_cpu = self.aruco_indices.cpu().long()
        self.aruco_positions = self.root_states[aruco_indices_cpu, 0:3].to(self.device)
        
        # 目标位置（与 ArUco 相同）- 仅在初始化时设置，之后由 _reset_envs 管理
        if not hasattr(self, 'target_pos'):
            self.target_pos = self.aruco_positions.clone()
        
        print(f"[Buffers] 缓冲区已更新: {actors_per_env} actors/env (UR10 + ArUco + {num_obstacles} obstacles)")
    
    def _create_viewer(self):
        """创建视口"""
        if not self.enable_viewer:
            return
        
        camera_pos = self.cfg.visualization.camera_pos if hasattr(self.cfg, 'visualization') else [2.0, 2.0, 2.0]
        camera_target = self.cfg.visualization.camera_target if hasattr(self.cfg, 'visualization') else [0.0, 0.0, 0.5]
        
        # 创建视口
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("[Warning] 无法创建视口")
            return
        
        # 设置相机
        cam_pos = gymapi.Vec3(camera_pos[0], camera_pos[1], camera_pos[2])
        cam_target = gymapi.Vec3(camera_target[0], camera_target[1], camera_target[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        print("[Visualization] 视口已创建")
    
    def _init_debug_viz(self):
        """初始化调试可视化"""
        print("[Debug Viz] 初始化传感器可视化")
        if self.draw_lidar:
            print("  - ✅ LiDAR 射线绘制已启用")
        if self.draw_camera_fov:
            print("  - ✅ 摄像头视野绘制已启用")
    
    def _draw_sensor_visualization(self):
        """绘制传感器可视化（LiDAR 和摄像头）"""
        if not self.enable_viewer or self.viewer is None:
            return
        
        # 只可视化第一个环境
        env_id = 0
        
        # 获取末端执行器状态（位置和姿态）
        ee_state = self._get_ee_state()
        ee_pos = ee_state[env_id, :3].cpu().numpy()
        ee_quat = ee_state[env_id, 3:7].cpu().numpy()  # [w, x, y, z] 四元数
        
        # 绘制 LiDAR 射线
        if self.draw_lidar:
            self._draw_lidar_rays(env_id, ee_pos, ee_quat)
        
        # 绘制摄像头视野
        if self.draw_camera_fov:
            self._draw_camera_fov(env_id, ee_pos, ee_quat)
    
    def _draw_lidar_rays(self, env_id, ee_pos, ee_quat):
        """绘制 LiDAR 射线"""
        # 采样部分射线进行可视化（全部144条太多）
        sample_h = 4  # 每40度一条
        sample_v = 2  # 中间2条
        
        for h in range(0, self.lidar_hbeams, sample_h):
            for v in range(0, self.lidar_vbeams, sample_v):
                # 获取射线方向（局部坐标系）
                ray_dir_local = self.ray_directions[h, v].cpu().numpy()
                
                # 将射线方向从局部坐标系转换到世界坐标系 - 工具模块
                ray_dir_world = quat_rotate_vector(ee_quat, ray_dir_local)
                
                # 获取该射线的检测距离
                lidar_distance = self.lidar_scan[env_id, 0, h, v].item()
                actual_distance = self.lidar_range - lidar_distance
                
                # 计算射线终点
                ray_end = ee_pos + ray_dir_world * actual_distance
                
                # 绘制射线（绿色=检测到，红色=未检测到）
                color = gymapi.Vec3(0.0, 1.0, 0.0) if actual_distance < self.lidar_range * 0.9 else gymapi.Vec3(1.0, 0.0, 0.0)
                
                self.gym.add_lines(
                    self.viewer,
                    self.envs[env_id],
                    1,
                    [ee_pos[0], ee_pos[1], ee_pos[2], ray_end[0], ray_end[1], ray_end[2]],
                    [color.x, color.y, color.z]
                )
    
    # _quat_rotate_vector → utils.math_utils.quat_rotate_vector (NumPy版本)
    
    def _draw_camera_fov(self, env_id, ee_pos, ee_quat):
        """绘制摄像头视野锥（跟随末端执行器姿态）"""
        # 摄像头在末端执行器坐标系中的朝向（+X 方向，即末端正前方）
        # 注意：摄像头安装在末端正上方（位置），但朝向是正前方
        cam_forward_local = np.array([1.0, 0.0, 0.0])
        
        # 转换到世界坐标系 - 工具模块
        cam_forward_world = quat_rotate_vector(ee_quat, cam_forward_local)
        cam_forward_world = cam_forward_world / np.linalg.norm(cam_forward_world)
        
        # 计算视野锥的4个角点（在局部坐标系中）
        # 摄像头朝向 +X（末端正前方），视野锥围绕 X 轴展开
        half_fov_rad = np.deg2rad(self.camera_fov / 2.0)
        sin_half_fov = np.sin(half_fov_rad)
        cos_half_fov = np.cos(half_fov_rad)
        
        # 4个角方向（相对于 +X 轴的圆锥）
        # 使用球面坐标：[cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)]
        # theta = half_fov_rad, phi = 方位角
        fov_directions_local = [
            np.array([cos_half_fov, sin_half_fov, sin_half_fov]),    # 右上（+Y, +Z）
            np.array([cos_half_fov, -sin_half_fov, sin_half_fov]),   # 左上（-Y, +Z）
            np.array([cos_half_fov, -sin_half_fov, -sin_half_fov]),  # 左下（-Y, -Z）
            np.array([cos_half_fov, sin_half_fov, -sin_half_fov]),   # 右下（+Y, -Z）
        ]
        
        # 绘制视野边界线
        fov_points_world = []
        for direction_local in fov_directions_local:
            # 归一化
            direction_local = direction_local / np.linalg.norm(direction_local)
            
            # 转换到世界坐标系 - 工具模块
            direction_world = quat_rotate_vector(ee_quat, direction_local)
            
            # 计算终点
            ray_end = ee_pos + direction_world * self.camera_max_distance
            fov_points_world.append(ray_end)
            
            # 绘制从末端到视野边界的线
            color = gymapi.Vec3(0.0, 0.5, 1.0)  # 蓝色
            self.gym.add_lines(
                self.viewer,
                self.envs[env_id],
                1,
                [ee_pos[0], ee_pos[1], ee_pos[2], ray_end[0], ray_end[1], ray_end[2]],
                [color.x, color.y, color.z]
            )
        
        # 绘制视野锥的边框（连接4个角点）
        for i in range(4):
            p1 = fov_points_world[i]
            p2 = fov_points_world[(i + 1) % 4]
            color = gymapi.Vec3(0.0, 0.5, 1.0)  # 蓝色
            self.gym.add_lines(
                self.viewer,
                self.envs[env_id],
                1,
                [p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]],
                [color.x, color.y, color.z]
            )
        
        # 绘制中心线（摄像头主轴）
        center_end = ee_pos + cam_forward_world * self.camera_max_distance
        color = gymapi.Vec3(0.0, 0.8, 1.0)  # 亮蓝色
        self.gym.add_lines(
            self.viewer,
            self.envs[env_id],
            1,
            [ee_pos[0], ee_pos[1], ee_pos[2], center_end[0], center_end[1], center_end[2]],
            [color.x, color.y, color.z]
        )
        
        # 如果目标在视野内，绘制到目标的连线
        if self.target_discovered[env_id]:
            target_pos = self.target_pos[env_id].cpu().numpy()
            color = gymapi.Vec3(1.0, 1.0, 0.0)  # 黄色
            self.gym.add_lines(
                self.viewer,
                self.envs[env_id],
                1,
                [ee_pos[0], ee_pos[1], ee_pos[2], target_pos[0], target_pos[1], target_pos[2]],
                [color.x, color.y, color.z]
            )
    
    def _init_basic_buffers(self):
        """初始化基础状态缓冲区（障碍物创建之前）"""
        # 获取状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 创建访问器
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        
        # 此时每个环境只有 2 个 actors（UR10 + ArUco）
        # 障碍物将在后面创建
        actors_per_env = 2  # UR10 + ArUco (障碍物还未创建)
        
        # 为每个环境提取 UR10 的 root_state
        # UR10 是每个环境的第一个 actor (索引 0)
        self.ur10_indices = torch.arange(self.num_envs, device=self.device) * actors_per_env
        
        # ArUco 是每个环境的第二个 actor (索引 1)
        self.aruco_indices = torch.arange(self.num_envs, device=self.device) * actors_per_env + 1
        
        # 提取 UR10 的状态 (确保在正确的设备上)
        ur10_indices_cpu = self.ur10_indices.cpu().long()
        self.root_positions = self.root_states[ur10_indices_cpu, 0:3].to(self.device)
        self.root_orientations = self.root_states[ur10_indices_cpu, 3:7].to(self.device)
        self.root_linvels = self.root_states[ur10_indices_cpu, 7:10].to(self.device)
        self.root_angvels = self.root_states[ur10_indices_cpu, 10:13].to(self.device)
        
        # 提取 ArUco 的位置作为目标
        aruco_indices_cpu = self.aruco_indices.cpu().long()
        self.aruco_positions = self.root_states[aruco_indices_cpu, 0:3].to(self.device)
        
        # DOF 状态形状: [num_envs * num_dofs, 2]
        # reshape 为 [num_envs, num_dofs] 并确保在正确的设备上
        self.dof_positions = self.dof_states[:, 0].reshape(self.num_envs, self.num_dofs).to(self.device)
        self.dof_velocities = self.dof_states[:, 1].reshape(self.num_envs, self.num_dofs).to(self.device)
        
        # 重置缓冲区
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # 目标和障碍物状态
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        
        # 两阶段任务状态
        self.target_discovered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 是否发现目标
        self.exploration_phase = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)  # 是否在探索阶段
        self.discovery_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 发现目标的步数
        
        # 连续发现计数器（需要连续多步看到目标才确认发现）
        self.target_confirm_steps = self.cfg.env.target_confirm_steps if hasattr(self.cfg.env, 'target_confirm_steps') else 5
        self.target_view_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 连续看到目标的步数
        
        # 探索历史（记录访问过的区域）
        self.visited_positions = torch.zeros(self.num_envs, 0, 3, device=self.device)  # 访问过的位置历史
        
        # 搜索模式状态（用于奖励系统性搜索）
        self.prev_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)  # 上一步的末端位置
        self.prev_ee_orient = torch.zeros(self.num_envs, 4, device=self.device)  # 上一步的末端姿态
        self.prev_ee_vel_linear = torch.zeros(self.num_envs, 3, device=self.device)  # 上一步的速度
        self.scan_direction_history = torch.zeros(self.num_envs, 5, 3, device=self.device)  # 最近5步的移动方向
        self.scan_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 扫描步数计数
        
        # 摄像头参数（用于视觉检测）
        self.camera_fov = self.cfg.env.camera_fov if hasattr(self.cfg.env, 'camera_fov') else 60.0
        self.camera_max_distance = self.cfg.env.camera_max_distance if hasattr(self.cfg.env, 'camera_max_distance') else 2.0
        
        # ArUco 生成范围
        self.aruco_min_distance = self.cfg.env.aruco_min_distance if hasattr(self.cfg.env, 'aruco_min_distance') else 0.5
        self.aruco_max_distance = self.cfg.env.aruco_max_distance if hasattr(self.cfg.env, 'aruco_max_distance') else 1.5
        self.aruco_height_range = self.cfg.env.aruco_height_range if hasattr(self.cfg.env, 'aruco_height_range') else [0.3, 0.8]
        
        print(f"[Two-Phase Task] 探索 + 导航模式已启用")
        print(f"[Camera] FOV={self.camera_fov}°, max_dist={self.camera_max_distance}m")
        print(f"[ArUco] 距离范围: {self.aruco_min_distance}m - {self.aruco_max_distance}m")
        
        # LiDAR 缓冲区
        total_rays = self.lidar_hbeams * self.lidar_vbeams
        self.lidar_scan = torch.zeros(
            self.num_envs, 1, self.lidar_hbeams, self.lidar_vbeams, 
            device=self.device
        )  # 存储 LiDAR 扫描数据
        
        # 障碍物位置缓冲区 (每个环境3个障碍物)
        self.obstacle_positions = torch.zeros(self.num_envs, 3, 3, device=self.device)
        
        # 延迟到第一次使用时才计算射线方向
        self.ray_directions = None
    
    def _check_target_in_view(self):
        """检查 ArUco 是否在摄像头视野内（使用真实的末端姿态）"""
        # 获取末端执行器状态
        ee_state = self._get_ee_state()
        ee_pos = ee_state[:, :3].to(self.device)  # [num_envs, 3]
        ee_quat = ee_state[:, 3:7].to(self.device)  # [num_envs, 4]
        
        # 计算相对位置
        rel_pos = self.target_pos - ee_pos  # [num_envs, 3]
        distance = torch.norm(rel_pos, dim=1)  # [num_envs]
        
        # 检查距离是否在范围内
        in_range = distance < self.camera_max_distance
        
        # 归一化相对位置向量
        rel_dir = rel_pos / (distance.unsqueeze(1) + 1e-6)
        
        # 摄像头朝向（局部坐标系中的 +X 方向，即末端正前方）
        camera_dir_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, 3)
        
        # 使用四元数旋转到世界坐标系 - 工具模块
        camera_dir_world = quat_rotate_vector_batch(ee_quat, camera_dir_local)
        
        # 计算目标方向与摄像头方向的夹角
        dot_product = (rel_dir * camera_dir_world).sum(dim=1)  # [num_envs]
        angle_rad = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        angle_deg = torch.rad2deg(angle_rad)
        
        # 检查是否在视场角内
        half_fov = self.camera_fov / 2.0
        in_fov = angle_deg < half_fov
        
        # 同时满足距离和视场角条件
        in_view = in_range & in_fov
        
        return in_view
    
    def _compute_ray_directions(self):
        """预计算 LiDAR 射线方向 (在末端执行器坐标系中)"""
        # 水平角度: 0 到 360 度
        h_angles = torch.linspace(0, 360 - self.lidar_hres, self.lidar_hbeams, device=self.device)
        h_angles_rad = torch.deg2rad(h_angles)
        
        # 垂直角度: vfov[0] 到 vfov[1]
        v_angles = torch.linspace(self.lidar_vfov[0], self.lidar_vfov[1], self.lidar_vbeams, device=self.device)
        v_angles_rad = torch.deg2rad(v_angles)
        
        # 创建网格
        h_grid, v_grid = torch.meshgrid(h_angles_rad, v_angles_rad, indexing='ij')
        
        # 计算射线方向 (球坐标转笛卡尔坐标)
        # X: 前方, Y: 左方, Z: 上方
        x = torch.cos(v_grid) * torch.cos(h_grid)
        y = torch.cos(v_grid) * torch.sin(h_grid)
        z = torch.sin(v_grid)
        
        # [h_beams, v_beams, 3]
        self.ray_directions = torch.stack([x, y, z], dim=-1)
        
        print(f"[LiDAR] 射线方向已预计算: {self.ray_directions.shape}")
    
    def _update_lidar(self):
        """更新 LiDAR 扫描数据 - GPU 优化版本"""
        # 🔧 第一次调用时初始化射线方向
        if self.ray_directions is None:
            self._compute_ray_directions()
        
        # 获取末端执行器状态
        ee_state = self._get_ee_state()
        ee_pos = ee_state[:, :3].to(self.device)  # [num_envs, 3]
        ee_quat = ee_state[:, 3:7].to(self.device)  # [num_envs, 4]
        
        # 更新障碍物位置（完全向量化，无 Python 循环）
        actors_per_env = 5  # 每个环境有5个实体
        num_obstacles = 3   # 其中3个是障碍物
        
        # 生成所有障碍物索引（向量化，在 CPU 上因为 root_states 在 CPU）
        # env_ids: [0,0,0, 1,1,1, 2,2,2, ...]
        # obs_ids: [0,1,2, 0,1,2, 0,1,2, ...]
        env_ids = torch.arange(self.num_envs).unsqueeze(1).expand(-1, num_obstacles).flatten()
        obs_ids = torch.arange(num_obstacles).unsqueeze(0).expand(self.num_envs, -1).flatten()
        
        # 计算障碍物在 root_states 中的索引（CPU）
        obstacle_indices = env_ids * actors_per_env + 2 + obs_ids
        
        # 批量提取障碍物位置（从 CPU tensor 索引，然后移到 GPU）
        all_obs_pos = self.root_states[obstacle_indices.long(), 0:3].to(self.device)
        self.obstacle_positions = all_obs_pos.reshape(self.num_envs, num_obstacles, 3)
        
        # 批量旋转所有射线方向（完全 GPU 并行，无 Python 循环）
        num_rays = self.lidar_hbeams * self.lidar_vbeams
        
        # 预扩展射线方向 [h_beams, v_beams, 3] → [num_envs, h_beams, v_beams, 3]
        ray_dirs_local = self.ray_directions.unsqueeze(0).expand(self.num_envs, -1, -1, -1)
        
        # Reshape 为 [num_envs * num_rays, 3] 用于批量旋转
        ray_dirs_local_flat = ray_dirs_local.reshape(self.num_envs * num_rays, 3)
        
        # 扩展四元数：[num_envs, 4] → [num_envs * num_rays, 4]
        # 每个环境的四元数复制 num_rays 次
        ee_quat_expanded = ee_quat.unsqueeze(1).expand(-1, num_rays, -1).reshape(self.num_envs * num_rays, 4)
        
        # 一次性旋转所有射线（完全并行）- 工具模块
        ray_dirs_world_flat = quat_rotate_vector_batch(ee_quat_expanded, ray_dirs_local_flat)
        
        # Reshape 回 [num_envs, h_beams, v_beams, 3]
        ray_dirs_world = ray_dirs_world_flat.reshape(self.num_envs, self.lidar_hbeams, self.lidar_vbeams, 3)
        
        # GPU 并行计算所有射线与障碍物的碰撞（射线-球体相交，更准确）
        # ray_dirs_world: [num_envs, h_beams, v_beams, 3]
        # obstacle_positions: [num_envs, 3, 3]
        
        # 射线-球体相交检测（向量化，GPU 并行）
        # 扩展维度用于广播
        ee_pos_2d = ee_pos.unsqueeze(1).unsqueeze(1)  # [num_envs, 1, 1, 3]
        obs_pos_3d = self.obstacle_positions.unsqueeze(1).unsqueeze(1)  # [num_envs, 1, 1, 3, 3]
        ray_dirs_4d = ray_dirs_world.unsqueeze(3)  # [num_envs, h_beams, v_beams, 1, 3]
        
        # 向量: 射线原点到球心 [num_envs, 1, 1, 3, 3] - [num_envs, 1, 1, 1, 3]
        oc = obs_pos_3d - ee_pos_2d.unsqueeze(3)  # [num_envs, 1, 1, 3, 3]
        
        # 射线方向与 oc 的点积（沿射线的投影）
        # [num_envs, h, v, 1, 3] * [num_envs, 1, 1, 3, 3] -> [num_envs, h, v, 3, 3]
        ray_to_center_proj = (ray_dirs_4d * oc).sum(dim=-1)  # [num_envs, h, v, 3]
        
        # 计算射线到球心的最短距离的平方
        oc_norm_sq = (oc ** 2).sum(dim=-1)  # [num_envs, 1, 1, 3]
        closest_dist_sq = oc_norm_sq - ray_to_center_proj ** 2  # [num_envs, h, v, 3]
        
        # 障碍物半径（使用固定大小）
        obstacle_radius = self.obstacle_size / 2.0
        
        # 击中条件：最短距离小于半径，且投影为正（在射线前方）
        hit_mask = (closest_dist_sq <= obstacle_radius ** 2) & (ray_to_center_proj > 0)
        
        # 计算精确的击中距离（使用射线-球体相交公式）
        # 如果 hit_mask 为真，计算实际交点距离
        # d = proj - sqrt(r^2 - closest_dist_sq)
        discriminant = obstacle_radius ** 2 - closest_dist_sq
        hit_distances = torch.where(
            hit_mask,
            ray_to_center_proj - torch.sqrt(discriminant.clamp(min=0)),
            torch.full_like(ray_to_center_proj, self.lidar_range)
        )
        
        # 确保距离为正
        hit_distances = hit_distances.clamp(min=0, max=self.lidar_range)
        
        # 对每条射线，取所有障碍物中的最小距离
        min_distance = hit_distances.min(dim=-1)[0]  # [num_envs, h_beams, v_beams]
        
        # 存储 LiDAR 扫描数据 (模仿无人机: range - distance)
        self.lidar_scan = (self.lidar_range - min_distance).unsqueeze(1)  # [num_envs, 1, h_beams, v_beams]
        
        # 限制在 [0, lidar_range]
        self.lidar_scan = torch.clamp(self.lidar_scan, 0, self.lidar_range)
    
    def reset(self):
        """重置环境"""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_envs(env_ids)
        
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        self.progress_buf.zero_()
        self.reset_buf.zero_()
        
        # 调试：检查初始状态
        if self.num_envs == 1:
            ee_state = self._get_ee_state()
            ee_pos = ee_state[0, :3]
            ee_quat = ee_state[0, 3:7]
            target = self.target_pos[0]
            
            distance = torch.norm(target - ee_pos).item()
            
            from utils.math_utils import quat_rotate_vector_batch
            camera_dir_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0)
            camera_dir_world = quat_rotate_vector_batch(ee_quat.unsqueeze(0), camera_dir_local)
            
            rel_dir = (target - ee_pos) / (distance + 1e-6)
            dot = (rel_dir * camera_dir_world[0]).sum().item()
            angle = torch.rad2deg(torch.acos(torch.clamp(torch.tensor(dot), -1.0, 1.0))).item()
            
            in_view_initial = distance < self.camera_max_distance and angle < self.camera_fov/2
            
            print(f"\n{'='*80}")
            print(f"🔄 环境重置")
            print(f"{'='*80}")
            print(f"  末端位置: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]")
            print(f"  目标位置: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
            print(f"  初始距离: {distance:.2f}m")
            print(f"  初始角度: {angle:.1f}°")
            print(f"  初始在视野内: {in_view_initial}")
            if in_view_initial:
                print(f"  ⚠️  警告：初始时就在视野内，会立即触发发现！")
            print(f"{'='*80}\n")
        
        return self._compute_obs()
    
    def _reset_envs(self, env_ids):
        """重置指定环境"""
        # 随机初始关节位置（增加随机范围，避免初始时就对准目标）
        init_pos = torch.zeros((len(env_ids), self.num_dofs), device=self.device)
        # Base rotation: 更大的随机范围
        init_pos[:, 0] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 3.14  # ±90度
        # Shoulder: 向下看的初始姿态
        init_pos[:, 1] = -1.5 + torch.rand(len(env_ids), device=self.device) * 0.5  # [-1.5, -1.0]
        # Elbow: 随机弯曲
        init_pos[:, 2] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 2.0  # [-1.0, 1.0]
        # Wrist1: 额外随机化
        init_pos[:, 3] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 1.0  # [-0.5, 0.5]
        # Wrist2: 额外随机化
        init_pos[:, 4] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 1.0  # [-0.5, 0.5]
        # Wrist3: 额外随机化
        init_pos[:, 5] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 1.0  # [-0.5, 0.5]
        
        # 设置DOF状态
        for i, env_id in enumerate(env_ids):
            self.set_joint_positions(env_id, init_pos[i])
        
        # 设置 ArUco 位置 (作为目标) - 基于基座的随机半径
        import random
        import numpy as np
        for i, env_id in enumerate(env_ids):
            # 在指定半径范围内随机生成 ArUco 位置（以基座为圆心）
            # 基座位置假设为 (0, 0, 0)
            radius = random.uniform(self.aruco_min_distance, self.aruco_max_distance)
            angle = random.uniform(0, 2 * np.pi)  # 随机角度
            
            aruco_x = radius * np.cos(angle)
            aruco_y = radius * np.sin(angle)
            aruco_z = random.uniform(self.aruco_height_range[0], self.aruco_height_range[1])
            
            aruco_pos_np = np.array([aruco_x, aruco_y, aruco_z])
            
            # 构造新的 root state
            aruco_idx = self.aruco_indices[env_id].item()
            new_root_state = self.root_states[aruco_idx].clone()
            new_root_state[0:3] = torch.tensor(aruco_pos_np, device=self.device)
            new_root_state[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            new_root_state[7:13] = 0.0  # 速度设为0
            
            # 更新 root_states
            self.root_states[aruco_idx] = new_root_state
            
            # 存储 ArUco 位置作为目标（但初始状态下不让智能体知道）
            self.target_pos[env_id] = torch.tensor(aruco_pos_np, device=self.device)
            
            # 重置两阶段任务状态
            self.target_discovered[env_id] = False
            self.exploration_phase[env_id] = True
            self.discovery_step[env_id] = 0
            
            # 重置连续发现计数器
            if hasattr(self, 'target_view_counter'):
                self.target_view_counter[env_id] = 0
            
            # 重置导航奖励的历史误差（将在首次发现目标时自动初始化）
            if hasattr(self, 'prev_pose_error'):
                self.prev_pose_error[env_id] = 0.0
            
            # 重置角度历史（首次不计角度奖励）
            if hasattr(self, 'prev_angle_deg'):
                self.prev_angle_deg[env_id] = float('nan')
            
            # 重置搜索历史
            self.scan_step_counter[env_id] = 0
        
        # 重置方向历史和速度历史
        self.scan_direction_history[env_ids] = 0.0
        self.prev_ee_vel_linear[env_ids] = 0.0
        
        self.target_quat[env_ids] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32).unsqueeze(0).expand(len(env_ids), 4)
        
        # 应用状态更新到仿真（需要 CPU 索引）
        env_ids_int32 = env_ids.to(dtype=torch.int32).cpu()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
        
        # 应用 DOF 状态
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
    
    def set_joint_positions(self, env_id, positions):
        """设置关节位置"""
        # env_id 是张量，需要转换为 Python int
        env_id_int = int(env_id.item()) if isinstance(env_id, torch.Tensor) else int(env_id)
        
        # dof_states 形状: [num_envs * num_dofs, 2]
        # 直接修改对应环境的数据
        start_idx = env_id_int * self.num_dofs
        end_idx = start_idx + self.num_dofs
        self.dof_states[start_idx:end_idx, 0] = positions
    
    def step(self, actions):
        """执行一步"""
        # 应用动作（末端速度）
        self._apply_actions(actions)
        
        # 仿真一步
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # 更新缓冲区
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 更新 UR10 和 ArUco 的状态
        ur10_indices_cpu = self.ur10_indices.cpu().long()
        self.root_positions = self.root_states[ur10_indices_cpu, 0:3].to(self.device)
        self.root_orientations = self.root_states[ur10_indices_cpu, 3:7].to(self.device)
        self.root_linvels = self.root_states[ur10_indices_cpu, 7:10].to(self.device)
        self.root_angvels = self.root_states[ur10_indices_cpu, 10:13].to(self.device)
        
        # 更新 ArUco 位置（用于可视化等）
        aruco_indices_cpu = self.aruco_indices.cpu().long()
        self.aruco_positions = self.root_states[aruco_indices_cpu, 0:3].to(self.device)
        # 注意：target_pos 在 _reset_envs 中设置，不应在 step 中更新
        
        # 更新 LiDAR 数据 (模仿无人机)
        self._update_lidar()
        
        # 检查是否发现目标（通过摄像头视觉检测 - 需要连续多步确认）
        in_view = self._check_target_in_view()
        
        # 更新连续看到目标的计数器
        self.target_view_counter[in_view] += 1  # 能看到：计数+1
        self.target_view_counter[~in_view] = 0  # 看不到：重置为0
        
        # 只有连续看到足够多步才确认发现
        confirmed_discovery = self.target_view_counter >= self.target_confirm_steps
        
        # 更新目标发现状态（只针对未发现的环境）
        newly_discovered = confirmed_discovery & (~self.target_discovered)
        self.target_discovered = self.target_discovered | newly_discovered
        
        # 从探索阶段转到导航阶段
        self.exploration_phase = ~self.target_discovered
        
        # 记录发现目标的步数
        self.discovery_step[newly_discovered] = self.progress_buf[newly_discovered]
        
        # 调试：打印发现目标的瞬间
        if newly_discovered.any() and self.num_envs == 1:
            ee_state = self._get_ee_state()
            ee_pos = ee_state[0, :3]
            ee_quat = ee_state[0, 3:7]
            target = self.target_pos[0]
            distance = torch.norm(target - ee_pos).item()
            
            from utils.math_utils import quat_rotate_vector_batch
            camera_dir_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0)
            camera_dir_world = quat_rotate_vector_batch(ee_quat.unsqueeze(0), camera_dir_local)
            rel_dir = (target - ee_pos) / (distance + 1e-6)
            dot = (rel_dir * camera_dir_world[0]).sum().item()
            angle = torch.rad2deg(torch.acos(torch.clamp(torch.tensor(dot), -1.0, 1.0))).item()
            
            print(f"\n{'='*80}")
            print(f"🎉 目标发现！(Step {self.progress_buf[0].item()}) - 连续确认 {self.target_confirm_steps} 步")
            print(f"{'='*80}")
            print(f"  末端位置: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]")
            print(f"  目标位置: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
            print(f"  距离: {distance:.2f}m")
            print(f"  角度: {angle:.1f}°")
            print(f"  连续确认次数: {self.target_view_counter[0].item()}/{self.target_confirm_steps}")
            print(f"  发现奖励: {self.cfg.env.reward_discovery if hasattr(self.cfg.env, 'reward_discovery') else 10.0}")
            print(f"{'='*80}\n")
        
        # 当环境刚确认发现目标时，同步角度历史，避免阶段切换造成的角度奖励尖峰
        if newly_discovered.any():
            ee_state_sync = self._get_ee_state()
            ee_pos_sync = ee_state_sync[:, :3].to(self.device)
            ee_quat_sync = ee_state_sync[:, 3:7].to(self.device)
            from utils.math_utils import quat_rotate_vector_batch
            cam_dir_local_sync = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, 3)
            cam_dir_world_sync = quat_rotate_vector_batch(ee_quat_sync, cam_dir_local_sync)
            rel_vec_sync = (self.target_pos - ee_pos_sync)
            rel_dist_sync = torch.norm(rel_vec_sync, dim=1, keepdim=True)
            rel_dir_sync = rel_vec_sync / (rel_dist_sync + 1e-6)
            dot_sync = torch.clamp((rel_dir_sync * cam_dir_world_sync).sum(dim=1), -1.0, 1.0)
            angle_deg_sync = torch.rad2deg(torch.acos(dot_sync))
            if not hasattr(self, 'prev_angle_deg'):
                self.prev_angle_deg = torch.full((self.num_envs,), float('nan'), device=self.device)
            self.prev_angle_deg[newly_discovered] = angle_deg_sync[newly_discovered].detach()
        
        # 计算奖励和终止
        rewards = self._compute_reward(newly_discovered)
        done = self._check_done()
        
        self.progress_buf += 1
        self.reset_buf[:] = done
        
        # 自动重置完成的环境
        if done.any():
            env_ids_to_reset = done.nonzero(as_tuple=False).squeeze(-1)
            self._reset_envs(env_ids_to_reset)
            # 重置 progress_buf
            self.progress_buf[env_ids_to_reset] = 0
            
            # 刷新状态以确保重置生效
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 渲染 (如果启用)
        if self.enable_viewer and self.viewer is not None:
            # 清除之前的线条
            self.gym.clear_lines(self.viewer)
            
            # 绘制传感器可视化
            if self.draw_lidar or self.draw_camera_fov:
                self._draw_sensor_visualization()
            
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        
        return self._compute_obs(), rewards, done, {}
    
    def _apply_actions(self, actions):
        """应用动作（末端速度控制）"""
        # actions: [num_envs, 6] - [vx, vy, vz, wx, wy, wz]
        # 需要转换为关节速度
        
        # 简化实现：直接使用雅可比映射
        # TODO: 实现完整的雅可比矩阵
        
        ee_vel = actions  # [num_envs, 6]
        joint_vel = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        
        # 简化的线性映射
        joint_vel[:, 0] = ee_vel[:, 0] * 3.0  # X
        joint_vel[:, 1] = ee_vel[:, 1] * 3.0  # Y
        joint_vel[:, 2] = ee_vel[:, 2] * 3.0  # Z
        joint_vel[:, 3] = ee_vel[:, 3] * 2.0  # wx
        joint_vel[:, 4] = ee_vel[:, 4] * 2.0  # wy
        joint_vel[:, 5] = ee_vel[:, 5] * 2.0  # wz
        
        # 限制关节速度
        joint_vel = torch.clamp(joint_vel, -2.0, 2.0)
        
        # 设置关节速度目标
        for env_id in range(self.num_envs):
            dof_states = self.gym.get_actor_dof_states(self.envs[env_id], self.ur10_handles[env_id], gymapi.STATE_VEL)
            dof_states['vel'] = joint_vel[env_id].cpu().numpy()
            self.gym.set_actor_dof_velocity_targets(self.envs[env_id], self.ur10_handles[env_id], dof_states['vel'])
    
    def _compute_obs(self):
        """计算观测（包含 LiDAR 数据和目标发现标志）"""
        # 获取末端执行器状态
        ee_state = self._get_ee_state()
        
        # 相对目标位置
        ee_pos = ee_state[:, :3]
        
        # 确保所有张量在同一设备上
        ee_pos = ee_pos.to(self.device)
        ee_vel = ee_state[:, 7:13].to(self.device)
        joint_pos = self.dof_positions.to(self.device)
        
        # 如果目标已发现，提供真实的相对位置；否则提供零向量
        rpos = torch.zeros(self.num_envs, 3, device=self.device)
        distance = torch.zeros(self.num_envs, 1, device=self.device)
        
        discovered_mask = self.target_discovered.unsqueeze(1)  # [num_envs, 1]
        
        # 只有发现目标后才提供真实的目标信息
        real_rpos = self.target_pos - ee_pos
        real_distance = torch.norm(real_rpos, dim=1, keepdim=True)
        
        rpos = torch.where(discovered_mask, real_rpos, rpos)
        distance = torch.where(discovered_mask, real_distance, distance)
        
        # LiDAR 数据展平 (模仿无人机)
        # lidar_scan: [num_envs, 1, h_beams, v_beams] -> [num_envs, h_beams * v_beams]
        lidar_flat = self.lidar_scan.reshape(self.num_envs, -1).to(self.device)
        
        # 归一化 LiDAR 数据到 [0, 1]
        lidar_normalized = lidar_flat / self.lidar_range
        
        # 目标发现标志（0 或 1）
        target_found = self.target_discovered.float().unsqueeze(1)  # [num_envs, 1]
        
        # 组装观测（添加 LiDAR 数据和目标发现标志）
        obs = torch.cat([
            rpos / (distance + 1e-6),  # 归一化相对位置（仅在发现后有效） [num_envs, 3]
            distance,  # 距离（仅在发现后有效） [num_envs, 1]
            target_found,  # 目标发现标志 [num_envs, 1]
            ee_vel,  # 末端速度 [num_envs, 6]
            joint_pos,  # 关节位置 [num_envs, 6]
            lidar_normalized,  # LiDAR 扫描 [num_envs, h_beams * v_beams]
        ], dim=1)
        
        return obs
    
    def _get_ee_state(self):
        """获取末端执行器状态"""
        # 获取末端执行器刚体的索引
        # rigid_body_states 格式: [num_envs * num_bodies, 13]
        # 每个环境的刚体索引: env_id * num_bodies + body_idx
        
        # 创建 CPU 索引（因为 rigid_body_states 在 CPU 上）
        ee_indices = torch.arange(self.num_envs) * self.num_bodies + self.ee_body_idx
        ee_indices = ee_indices.long()
        
        # 提取末端执行器的状态
        # rigid_body_states: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        ee_states = self.rigid_body_states[ee_indices].to(self.device)
        
        ee_pos = ee_states[:, 0:3]
        ee_quat = ee_states[:, 3:7]
        ee_linvel = ee_states[:, 7:10]
        ee_angvel = ee_states[:, 10:13]
        
        ee_vel = torch.cat([ee_linvel, ee_angvel], dim=1)
        
        # 组合状态 [pos(3), quat(4), vel(6)] = 13
        ee_state = torch.cat([ee_pos, ee_quat, ee_vel], dim=1)
        
        return ee_state
    
    def _compute_reward(self, newly_discovered):
        """计算奖励（分为探索阶段和导航阶段）- 增强版"""
        ee_state = self._get_ee_state()
        ee_pos = ee_state[:, :3].to(self.device)
        ee_quat = ee_state[:, 3:7].to(self.device)
        ee_vel_linear = ee_state[:, 7:10].to(self.device)
        ee_vel_angular = ee_state[:, 10:13].to(self.device)
        
        # 角度塑形系数与限幅（度）
        angle_shaping_coef = self.cfg.env.angle_shaping_coef if hasattr(self.cfg.env, 'angle_shaping_coef') else 0.3
        angle_shaping_cap = self.cfg.env.angle_shaping_cap if hasattr(self.cfg.env, 'angle_shaping_cap') else 10.0
        
        # 批量计算相机朝向与目标方向的夹角（度）
        from utils.math_utils import quat_rotate_vector_batch
        camera_dir_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, 3)
        camera_dir_world = quat_rotate_vector_batch(ee_quat, camera_dir_local)  # [N,3]
        rel_vec = (self.target_pos - ee_pos)  # [N,3]
        rel_dist = torch.norm(rel_vec, dim=1, keepdim=True)
        rel_dir = rel_vec / (rel_dist + 1e-6)
        dot_vals = torch.clamp((rel_dir * camera_dir_world).sum(dim=1), -1.0, 1.0)
        angle_deg = torch.rad2deg(torch.acos(dot_vals))  # [N]
        
        # 初始化/更新历史角度，首次不计角度奖励
        if not hasattr(self, 'prev_angle_deg'):
            self.prev_angle_deg = torch.full((self.num_envs,), float('nan'), device=self.device)
        
        # 获取奖励权重
        reward_exploration_weight = self.cfg.env.reward_exploration if hasattr(self.cfg.env, 'reward_exploration') else 0.1
        reward_discovery_bonus = self.cfg.env.reward_discovery if hasattr(self.cfg.env, 'reward_discovery') else 10.0
        reward_approach_weight = self.cfg.env.reward_approach if hasattr(self.cfg.env, 'reward_approach') else 1.0
        reward_reach_bonus = self.cfg.env.reward_reach if hasattr(self.cfg.env, 'reward_reach') else 50.0
        
        # 探索模式的细粒度奖励权重
        reward_z_exploration = self.cfg.env.reward_z_exploration if hasattr(self.cfg.env, 'reward_z_exploration') else 0.15
        reward_downward_search = self.cfg.env.reward_downward_search if hasattr(self.cfg.env, 'reward_downward_search') else 0.2
        reward_systematic_scan = self.cfg.env.reward_systematic_scan if hasattr(self.cfg.env, 'reward_systematic_scan') else 0.3
        reward_active_exploration = self.cfg.env.reward_active_exploration if hasattr(self.cfg.env, 'reward_active_exploration') else 0.1
        penalty_static_behavior = self.cfg.env.penalty_static_behavior if hasattr(self.cfg.env, 'penalty_static_behavior') else 0.05
        
        # 初始化奖励
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 1. 探索阶段奖励（增强版）
        exploration_reward = torch.zeros(self.num_envs, device=self.device)
        exploration_mask = self.exploration_phase
        
        if exploration_mask.any():
            # 1.1 基础移动奖励
            velocity_magnitude = torch.norm(ee_vel_linear, dim=1)
            exploration_reward[exploration_mask] = velocity_magnitude[exploration_mask] * reward_exploration_weight
            
            # 1.2 Z轴探索奖励（只奖励向上移动）- 使用工具模块
            z_exploration_bonus = compute_z_exploration_reward(ee_vel_linear, reward_z_exploration)
            exploration_reward[exploration_mask] += z_exploration_bonus[exploration_mask]
            
            # 1.3 向下搜索奖励（摄像头朝下）- 使用工具模块
            downward_bonus = compute_downward_search_reward(ee_quat, reward_downward_search, self.device)
            exploration_reward[exploration_mask] += downward_bonus[exploration_mask]
            
            # 1.4 系统性扫描奖励（左右扫描、有规律的移动）- 使用工具模块
            systematic_bonus = compute_systematic_scan_reward(
                ee_vel_linear, reward_systematic_scan, 
                prev_direction=self.prev_move_direction if hasattr(self, 'prev_move_direction') else None
            )
            exploration_reward[exploration_mask] += systematic_bonus[exploration_mask]
            
            # 1.5 奖励主动探索移动（鼓励在探索阶段移动）- 使用工具模块
            active_reward = compute_active_exploration_reward(
                ee_vel_linear, ee_vel_angular, reward_active_exploration
            )
            exploration_reward[exploration_mask] += active_reward[exploration_mask]
            
            # 1.6 惩罚静止不动（探索时应该移动）- 使用工具模块
            static_pen = compute_static_penalty(
                ee_vel_linear, ee_vel_angular, penalty_static_behavior
            )
            exploration_reward[exploration_mask] -= static_pen[exploration_mask]
            
            # 1.7 角度减小即奖励（探索阶段）
            has_prev = torch.isfinite(self.prev_angle_deg)
            delta_angle = torch.zeros(self.num_envs, device=self.device)
            delta_angle[has_prev] = self.prev_angle_deg[has_prev] - angle_deg[has_prev]
            improve = torch.clamp(delta_angle, min=0.0, max=angle_shaping_cap)
            exploration_reward[exploration_mask] += angle_shaping_coef * improve[exploration_mask]
            
            # 更新历史状态
            self._update_search_history(ee_pos, ee_quat, ee_vel_linear)
        
        # 2. 发现目标奖励（一次性）
        discovery_reward = torch.zeros(self.num_envs, device=self.device)
        discovery_reward[newly_discovered] = reward_discovery_bonus
        
        # 3. 导航阶段奖励（发现目标后）- 使用三点位姿误差
        navigation_reward = torch.zeros(self.num_envs, device=self.device)
        navigation_mask = self.target_discovered
        
        if navigation_mask.any():
            # 使用三点表示法计算精确的位姿误差 - 工具模块
            pose_error = compute_pose_error_three_point(
                ee_pos, ee_quat, self.target_pos, target_quat=None, d=0.1
            )
            
            # 初始化上一步误差（如果不存在）
            if not hasattr(self, 'prev_pose_error'):
                self.prev_pose_error = torch.zeros(self.num_envs, device=self.device)
            
            # 对于刚发现目标的环境，将prev_pose_error设为当前误差（避免异常的delta）
            self.prev_pose_error[newly_discovered] = pose_error[newly_discovered]
            
            # 奖励误差减小（delta_error > 0 表示在接近目标）
            delta_error = self.prev_pose_error - pose_error
            navigation_reward[navigation_mask] = delta_error[navigation_mask] * reward_approach_weight * 10.0  # 放大系数
            
            # 更新历史误差
            self.prev_pose_error[navigation_mask] = pose_error[navigation_mask]
            
            # 到达目标奖励（使用简单欧几里得距离判断）
        distance = torch.norm(ee_pos - self.target_pos, dim=1)
            reached_mask = navigation_mask & (distance < 0.1)
            navigation_reward[reached_mask] += reward_reach_bonus
            
            # 3.1 导航阶段的视野管理奖励（智能导航）
            navigation_reward[navigation_mask] += self._compute_view_management_reward(
                ee_pos[navigation_mask], 
                ee_quat[navigation_mask],
                self.target_pos[navigation_mask]
            )
            
            # 3.2 角度减小即奖励（导航阶段）
            has_prev = torch.isfinite(self.prev_angle_deg)
            delta_angle = torch.zeros(self.num_envs, device=self.device)
            delta_angle[has_prev] = self.prev_angle_deg[has_prev] - angle_deg[has_prev]
            improve = torch.clamp(delta_angle, min=0.0, max=angle_shaping_cap)
            navigation_reward[navigation_mask] += angle_shaping_coef * improve[navigation_mask]
        
        # 4. 惩罚项
        penalty = torch.zeros(self.num_envs, device=self.device)
        
        # 碰撞惩罚（双重检测：LiDAR + 连杆距离）- 使用工具模块
        real_min_distance, lidar_min_distance, link_min_distance = detect_collision_multisource(
            self.lidar_scan,
            self.lidar_range,
            self.rigid_body_states,
            self.obstacle_positions,
            self.num_envs,
            self.num_bodies,
            self.device
        )
        
        # 分级惩罚 - 使用工具模块
        collision_threshold = self.collision_distance if hasattr(self, 'collision_distance') else 0.15
        warning_threshold = self.cfg.env.warning_distance if hasattr(self.cfg.env, 'warning_distance') else 0.2
        collision_pen = compute_collision_penalty(real_min_distance, collision_threshold, warning_threshold)
        penalty += collision_pen
        
        # 超出工作空间惩罚
        workspace_limit = 2.0
        out_of_workspace = torch.norm(ee_pos, dim=1) > workspace_limit
        penalty[out_of_workspace] -= 2.0
        
        # 更新历史角度（最后进行，避免本步读取被污染）
        self.prev_angle_deg = angle_deg.detach()
        
        # 总奖励
        rewards = exploration_reward + discovery_reward + navigation_reward + penalty
        
        # 调试信息：记录奖励分解（每100步打印一次）
        if not hasattr(self, 'debug_step_counter'):
            self.debug_step_counter = 0
        self.debug_step_counter += 1
        
        if self.debug_step_counter % 100 == 0 and self.num_envs == 1:
            # 获取末端和目标的详细信息
            ee_state = self._get_ee_state()
            ee_pos = ee_state[0, :3]
            ee_quat = ee_state[0, 3:7]
            target = self.target_pos[0]
            
            # 计算距离和角度
            distance = torch.norm(target - ee_pos).item()
            
            # 计算摄像头方向
            from utils.math_utils import quat_rotate_vector_batch
            camera_dir_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0)
            camera_dir_world = quat_rotate_vector_batch(ee_quat.unsqueeze(0), camera_dir_local)
            
            # 计算目标相对方向
            rel_dir = (target - ee_pos) / (distance + 1e-6)
            dot = (rel_dir * camera_dir_world[0]).sum().item()
            angle = torch.rad2deg(torch.acos(torch.clamp(torch.tensor(dot), -1.0, 1.0))).item()
            
            print(f"\n[Debug] 奖励分解 (Step {self.debug_step_counter}):")
            print(f"  探索奖励: {exploration_reward[0].item():7.3f}")
            print(f"  发现奖励: {discovery_reward[0].item():7.3f}")
            print(f"  导航奖励: {navigation_reward[0].item():7.3f}")
            print(f"  惩罚: {penalty[0].item():7.3f}")
            print(f"  总奖励: {rewards[0].item():7.3f}")
            print(f"  目标状态: {'已发现' if self.target_discovered[0] else '探索中'}")
            print(f"  → 距离: {distance:.2f}m (限制: {self.camera_max_distance:.2f}m)")
            print(f"  → 角度: {angle:.1f}° (限制: {self.camera_fov/2:.1f}°)")
            print(f"  → 在视野内: {distance < self.camera_max_distance and angle < self.camera_fov/2}")
        
        return rewards
    
    def _compute_view_management_reward(self, ee_pos, ee_quat, target_pos):
        """
        计算导航阶段的视野管理奖励
        
        Args:
            ee_pos: 末端位置 [N, 3]
            ee_quat: 末端姿态 [N, 4]
            target_pos: 目标位置 [N, 3]
            
        Returns:
            视野管理奖励 [N]
        """
        from utils.math_utils import quat_rotate_vector_batch
        
        num_envs = ee_pos.shape[0]
        reward = torch.zeros(num_envs, device=self.device)
        
        # 获取配置参数
        reward_target_centered = self.cfg.env.reward_target_centered if hasattr(self.cfg.env, 'reward_target_centered') else 3.0
        reward_aware_navigation = self.cfg.env.reward_aware_navigation if hasattr(self.cfg.env, 'reward_aware_navigation') else 5.0
        penalty_narrow_focus = self.cfg.env.penalty_narrow_focus if hasattr(self.cfg.env, 'penalty_narrow_focus') else 2.0
        camera_center_region = self.cfg.env.camera_center_region if hasattr(self.cfg.env, 'camera_center_region') else 15.0
        
        # 1. 计算目标相对于摄像头的角度
        rel_pos = target_pos - ee_pos  # [N, 3]
        distance = torch.norm(rel_pos, dim=1, keepdim=True)  # [N, 1]
        rel_dir = rel_pos / (distance + 1e-6)  # [N, 3]
        
        # 摄像头朝向（+X方向）
        camera_dir_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(num_envs, 3)
        camera_dir_world = quat_rotate_vector_batch(ee_quat, camera_dir_local)
        
        # 计算夹角
        dot_product = (rel_dir * camera_dir_world).sum(dim=1)  # [N]
        angle_rad = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        angle_deg = torch.rad2deg(angle_rad)
        
        # 2. 奖励保持目标在视野中心区域（安全布尔掩码应用）
        in_center = (angle_deg < camera_center_region)
        if in_center.any():
            reward[in_center] += reward_target_centered * (1.0 - angle_deg[in_center] / camera_center_region)
        
        # 3. 奖励能同时看到目标和障碍物的视角（提升环境感知）
        # 检查LiDAR是否检测到障碍物
        if hasattr(self, 'lidar_scan'):
            # lidar_scan: [num_envs, h*v] - 数值越大代表更近（实现内定义）
            lidar_max = self.lidar_scan.max(dim=1)[0]  # [num_envs]
            has_obstacles_nearby = lidar_max > (self.lidar_range * 0.3)
            
            # 如果能看到目标（角度较小）且LiDAR检测到障碍物，说明视角良好
            good_awareness = in_center & has_obstacles_nearby
            if good_awareness.any():
                reward[good_awareness] += reward_aware_navigation
        
        # 4. 惩罚"盯着目标不看路"（只看目标，LiDAR没检测到任何障碍）
        if hasattr(self, 'lidar_scan'):
            # 目标在视野中心，但没检测到任何障碍物（可能是视野太窄或忽略环境）
            lidar_max = self.lidar_scan.max(dim=1)[0]
            no_obstacle_detected = lidar_max < (self.lidar_range * 0.1)
            
            narrow_focus = in_center & no_obstacle_detected
            if narrow_focus.any():
                reward[narrow_focus] -= penalty_narrow_focus
        
        return reward
    
    # ====================================================================
    # 以下方法已移至 utils/ 模块，此处保留为注释以便参考
    # ====================================================================
    # _compute_downward_search_reward → utils.rewards.compute_downward_search_reward
    # _compute_systematic_scan_reward → utils.rewards.compute_systematic_scan_reward
    # _compute_active_exploration_reward → utils.rewards.compute_active_exploration_reward
    # _compute_static_penalty → utils.rewards.compute_static_penalty
    # ====================================================================
    
    def _update_search_history(self, ee_pos, ee_quat, ee_vel_linear):
        """更新搜索历史"""
        # 存储当前状态作为下一步的"上一步"
        self.prev_ee_pos = ee_pos.clone()
        self.prev_ee_orient = ee_quat.clone()
        
        # 增加扫描步数
        self.scan_step_counter += 1
    
    # ====================================================================
    # 数学方法已移至 utils/ 模块
    # ====================================================================
    # _compute_pose_error_three_point → utils.math_utils.compute_pose_error_three_point
    # _compute_link_obstacle_distances → utils.collision.compute_link_obstacle_distances
    # ====================================================================
    
    # _quat_rotate_vector_batch → utils.math_utils.quat_rotate_vector_batch (批量版本)
    
    def _check_done(self):
        """检查终止条件（增强版）"""
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 初始化终止原因记录（用于统计）
        # 0: 运行中, 1: 超时, 2: 成功, 3: 碰撞, 4: 越界, 5: 关节限位
        if not hasattr(self, 'termination_reason'):
            self.termination_reason = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # 1. 超时终止
        timeout = self.progress_buf >= self.max_episode_length
        done = done | timeout
        self.termination_reason[timeout] = 1
        
        # 2. 成功到达目标
        ee_state = self._get_ee_state()
        ee_pos = ee_state[:, :3].to(self.device)
        distance = torch.norm(ee_pos - self.target_pos, dim=1)
        success = distance < self.success_distance
        done = done | success
        self.termination_reason[success] = 2
        
        # 3. 碰撞终止（可选）- 使用工具模块
        if self.terminate_on_collision and hasattr(self, 'obstacle_positions'):
            # 使用连杆-障碍物距离（比 LiDAR 更精确）
            link_min_distance = compute_link_obstacle_distances(
                self.rigid_body_states,
                self.obstacle_positions,
                self.num_envs,
                self.num_bodies,
                self.device
            )
            collision = link_min_distance < self.collision_distance
            done = done | collision
            self.termination_reason[collision] = 3
        
        # 4. 超出工作空间终止（可选）
        if self.terminate_on_out_of_bounds:
            distance_from_base = torch.norm(ee_pos[:, :2], dim=1)  # XY平面距离
            out_of_bounds_xy = distance_from_base > self.workspace_limit
            out_of_bounds_z = (ee_pos[:, 2] < -0.2) | (ee_pos[:, 2] > 2.0)  # Z轴范围
            out_of_bounds = out_of_bounds_xy | out_of_bounds_z
            done = done | out_of_bounds
            self.termination_reason[out_of_bounds] = 4
        
        # 5. 关节限位终止（可选）
        if self.terminate_on_joint_limits and hasattr(self, 'dof_lower_limits'):
            # 获取关节位置（需要先刷新）
            dof_pos = self.dof_positions.to(self.device)
            # 检查是否超出关节限位（不添加余量，直接使用限位）
            lower_limit_violation = dof_pos < self.dof_lower_limits.unsqueeze(0)
            upper_limit_violation = dof_pos > self.dof_upper_limits.unsqueeze(0)
            joint_limit_violation = (lower_limit_violation | upper_limit_violation).any(dim=1)
            done = done | joint_limit_violation
            self.termination_reason[joint_limit_violation] = 5
        
        # 重置未终止环境的原因
        self.termination_reason[~done] = 0
        
        return done
    
    def close(self):
        """关闭环境"""
        self.gym.destroy_sim(self.sim)
