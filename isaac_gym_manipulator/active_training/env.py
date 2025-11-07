import os
import math
import numpy as np
from typing import Optional, Union

# IMPORTANT: Isaac Gym must be imported before PyTorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch

# 导入数学工具函数（用于LiDAR射线旋转）
try:
    from ..utils.math_utils import quat_rotate_vector_batch
except ImportError:
    try:
        from isaac_gym_manipulator.utils.math_utils import quat_rotate_vector_batch
    except ImportError:
        # 如果导入失败，定义本地版本
        def quat_rotate_vector_batch(quat, vec):
            quat_w = quat[:, 3:4]
            quat_xyz = quat[:, :3]
            uv = torch.cross(quat_xyz, vec, dim=1)
            uuv = torch.cross(quat_xyz, uv, dim=1)
            return vec + 2.0 * (quat_w * uv + uuv)


class ArmNavEnv:

    """
    Isaac Gym 原生环境（不依赖 manipulator_env_gym）：
    - 加载 UR10e URDF
    - 将相机（D435i 等效）固定到 wrist_3_link 上方，朝 +X（末端局部前向）
    - 动作空间：action_dim=6，表示关节角度增量 [dq1, dq2, dq3, dq4, dq5, dq6]（弧度）
      - 关节顺序：shoulder_pan → shoulder_lift → elbow → wrist_1 → wrist_2 → wrist_3
      - 动作范围：[-0.0189, 0.0189] rad/step（参考 ur5e_DDPG_trajectory_planning_template）
      - 坐标系：关节空间（joint space），不需要坐标系转换
    - 提供 render_depth()/camera_pose()/intrinsics 供体素构建
    - 观测 aux 与现有训练脚本对齐
    """
    def __init__(self, num_envs: int = 1, max_steps: int = 800, img_w: int = 160, img_h: int = 120, fov_deg: float = 60.0, enable_viewer: bool = False, camera_pos: Optional[list] = None, camera_target: Optional[list] = None, cfg: Optional[dict] = None):
        # 🎯 GPU设备配置（支持多GPU服务器训练）
        # 从config.yaml读取gpu_device_id，如果没有配置则默认为0
        gpu_device_id = 0
        if cfg is not None and 'env' in cfg:
            gpu_device_id = cfg['env'].get('gpu_device_id', 0)
        
        # 设置PyTorch设备
        if torch.cuda.is_available() and gpu_device_id >= 0:
            # 检查GPU设备是否可用
            if gpu_device_id < torch.cuda.device_count():
                self._device = torch.device(f'cuda:{gpu_device_id}')
                # 设置当前CUDA设备（确保所有操作都在指定的GPU上）
                torch.cuda.set_device(gpu_device_id)
            else:
                print(f"[Warning] GPU {gpu_device_id} not available, only {torch.cuda.device_count()} GPUs found. Using GPU 0.")
                self._device = torch.device('cuda:0')
                torch.cuda.set_device(0)
                gpu_device_id = 0  # 更新为实际使用的设备ID
        else:
            self._device = torch.device('cpu')
            gpu_device_id = -1  # CPU模式
        
        # 保存GPU设备ID（用于Isaac Gym）
        self.gpu_device_id = gpu_device_id
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.img_w = img_w
        self.img_h = img_h
        self.fov = math.radians(fov_deg)
        
        # 可视化设置
        self.enable_viewer = enable_viewer
        self.camera_pos = camera_pos if camera_pos is not None else [2.0, 2.0, 2.0]
        self.camera_target = camera_target if camera_target is not None else [0.0, 0.0, 0.5]
        self.viewer = None
        self.render_step_count = 0  # 用于跟踪渲染频率
        self.visualize_env_index = 0  # 默认只可视化第一个环境，可由外部配置覆盖
        # ArUco 连续确认参数（默认值，可由外部 cfg 覆盖）
        self.target_confirm_steps = 5
        self.target_view_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.target_discovered = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.steps_since_last_detection = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.last_detection_step = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        
        # 多角度观察奖励相关状态（每个环境独立）
        self.observation_angles = [[] for _ in range(self.num_envs)]  # 每个环境存储已观察过的角度
        self.multiview_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)  # 每个环境发现目标后的观察步数
        
        # 探索覆盖率相关（体素地图统计）
        self.explored_voxels = set()  # 探索过的体素索引（用于覆盖率计算）
        self.total_voxels = None  # 总体素数（初始化时计算）

        # 随机球形障碍参数
        self.num_obstacles = 5
        self.sphere_radius_min = 0.10
        self.sphere_radius_max = 0.30
        self.obstacle_min_spacing = 0.40
        self.aruco_safe_radius = 0.30
        self.los_clearance = 0.25
        self.workspace_radius = 1.4
        self.workspace_z = [0.05, 1.2]
        self.keepout_base_radius = 0.60
        self.keepout_target_margin = 0.05
        self.obstacle_handles = []  # 每个环境的静态障碍物handle列表
        
        # 动态障碍物参数（默认值，会被config覆盖，应与config.yaml保持一致）
        self.num_dynamic_obstacles = 0  # 动态障碍物数量（每个环境）
        self.dynamic_obstacle_radius_min = 0.08  # 与config.yaml一致
        self.dynamic_obstacle_radius_max = 0.15  # 与config.yaml一致
        self.dynamic_obstacle_vel_range = [0.1, 0.2]  # 速度范围（m/s），与config.yaml一致
        self.dynamic_obstacle_local_range = [0.4, 0.4, 0.3]  # 局部移动范围 [x, y, z]，与config.yaml一致
        self.dynamic_obstacle_goal_threshold = 0.2  # 到达目标的距离阈值（m），与config.yaml一致
        self.dynamic_obstacle_vel_update_interval = 2.0  # 速度更新间隔（秒），与config.yaml一致
        self.num_closest_dyn_obs = 1  # 检测最近的N个动态障碍物，与config.yaml一致
        
        # 动态障碍物状态（将在创建后初始化）
        self.dynamic_obstacle_handles = []  # 动态障碍物handle列表
        self.dyn_obs_state = None  # [num_envs * num_dynamic_obstacles, 13] 位置+旋转+速度
        
        # 🎯 关键修复：竞争条件检测标志
        self._dynamic_obstacles_updated = False  # 标记动态障碍物是否已更新（用于检测竞争条件）
        self._render_depth_call_count = 0  # render_depth()调用计数器（用于定期清理GPU缓存）
        self.dyn_obs_goal = None  # [num_envs * num_dynamic_obstacles, 3] 目标位置
        self.dyn_obs_origin = None  # [num_envs * num_dynamic_obstacles, 3] 原点位置
        self.dyn_obs_vel = None  # [num_envs * num_dynamic_obstacles, 3] 速度
        self.dyn_obs_vel_norm = None  # [num_envs * num_dynamic_obstacles, 1] 速度大小
        self.dyn_obs_step_count = 0  # 步数计数器
        self.dyn_obs_radii = None  # [num_envs * num_dynamic_obstacles] 半径
        
        # ArUco位置范围（默认值，会被config覆盖）
        self.target_pos_range_x = [0.5, 1.2]
        self.target_pos_range_y = [-0.5, 0.5]
        self.target_pos_range_z = [0.25, 0.6]
        self.target_rotation_range = [-180, 180]  # 绕Z轴旋转范围（度）

        # 相机内参（针孔）
        fx = (img_w / 2) / math.tan(self.fov / 2)
        fy = (img_h / 2) / math.tan(self.fov / 2)
        cx = (img_w - 1) / 2
        cy = (img_h - 1) / 2
        self.intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        # 相机安装外参（相机坐标系采用z前、x右、y下的常规针孔模型）
        # 默认采用你ROS工程 ur_visual_servo.launch 的手眼标定（tool0->camera_link）
        # T_tc: t = [-0.08263826, -0.0410523, -0.00994019], q = [-8.18e-04, -1.89e-03, -3.86455e-01, 9.22306e-01]
        self.camera_mount_offset = torch.tensor([-0.08263826, -0.0410523, -0.00994019], dtype=torch.float32, device=self.device)
        self.camera_mount_quat = torch.tensor([-8.18298121e-04, -1.89235780e-03, -3.86455281e-01, 9.22305841e-01], dtype=torch.float32, device=self.device)
        # 若需要使用RPY而非四元数，可设置 self.camera_mount_quat=None 并改用下两行：
        self.camera_mount_rpy_deg = [0.0, -90.0, 0.0]
        self.camera_mount_rpy = torch.deg2rad(torch.tensor(self.camera_mount_rpy_deg, dtype=torch.float32, device=self.device))
        
        # 深度相机最大范围（用于体素构建）
        # 默认值 5.0 米，可以在 train.py 中通过配置覆盖
        self.max_range = 5.0  # 5米，D435i 的典型范围

        # Isaac Gym 初始化（参考manipulator_env_gym.py）
        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # 🎯 参考ur5e模板：dt=0.001，但Isaac Gym通常使用1/60.0
        # 对于每次RL step，执行约16个仿真步骤（对应1/60秒，约16.67ms）
        sim_params.dt = 1.0 / 60.0  # 每个仿真步骤的时间间隔（约16.67ms）
        sim_params.substeps = 2      # 子步骤数，提高仿真稳定性
        
        # PhysX 设置（参考manipulator_env_gym.py）
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False
        
        # 创建sim：第一个参数是计算设备ID，第二个参数是图形设备ID
        # 🎯 修复：即使不使用viewer，相机传感器也需要图形设备支持
        # 相机传感器用于深度图渲染和体素构建，需要图形设备，但不一定需要viewer窗口
        # 🎯 GPU设备配置：使用配置的GPU设备ID
        compute_device_id = self.gpu_device_id if self.gpu_device_id >= 0 else 0  # Isaac Gym的计算设备ID
        graphics_device_id = self.gpu_device_id if self.gpu_device_id >= 0 else 0  # Isaac Gym的图形设备ID（相机渲染需要）
        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        print(f"[Env] GPU设备配置: PyTorch={self._device}, Isaac Gym计算设备={compute_device_id}, Isaac Gym图形设备={graphics_device_id}")
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")

        # 地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # 创建多个环境（参考manipulator_env_gym.py）
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        self.envs = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)
        
        # 为了向后兼容，保留self.env指向第一个环境
        self.env = self.envs[0] if self.num_envs > 0 else None

        # 加载 UR10e URDF
        asset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        asset_file = 'ur10e.urdf'
        asset_path = os.path.join(asset_root, asset_file)
        
        # 检查URDF文件是否存在
        if not os.path.exists(asset_path):
            raise FileNotFoundError(f"URDF文件不存在: {asset_path}")
        
        # 配置AssetOptions以正确加载和显示URDF（参考manipulator_env_gym.py）
        asset_opts = gymapi.AssetOptions()
        asset_opts.fix_base_link = True
        asset_opts.flip_visual_attachments = True  # 启用以正确显示mesh
        asset_opts.use_mesh_materials = True  # 启用材质以正确显示颜色
        asset_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS  # 改为位置控制模式，用于关节角度增量控制
        asset_opts.override_com = True  # 覆盖质心计算
        asset_opts.override_inertia = True  # 覆盖惯性计算
        asset_opts.vhacd_enabled = True  # 启用VHACD以处理凸包碰撞（参考manipulator_env_gym.py）
        asset_opts.vhacd_params = gymapi.VhacdParams()
        asset_opts.vhacd_params.resolution = 300000  # 与manipulator_env_gym.py一致
        
        # 确保visual shapes被启用（关键设置）
        asset_opts.disable_gravity = False  # 物理重力
        asset_opts.collapse_fixed_joints = False  # 不合并固定关节
        asset_opts.replace_cylinder_with_capsule = False  # 不替换圆柱体
        # 注意：default_dof_drive_mode已在第128行设置，这里不再重复
        
        print(f"[URDF] AssetOptions设置: flip_visual={asset_opts.flip_visual_attachments}, use_materials={asset_opts.use_mesh_materials}")
        
        try:
            print(f"[URDF] 正在加载: {asset_path}")
            self.ur10_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_opts)
            if self.ur10_asset is None:
                raise RuntimeError(f"Failed to load URDF: {asset_path}")
            
            # 验证加载的资产
            num_dofs = self.gym.get_asset_dof_count(self.ur10_asset)
            num_bodies = self.gym.get_asset_rigid_body_count(self.ur10_asset)
            print(f"[URDF] ✅ 成功加载! DOF数量: {num_dofs}, 刚体数量: {num_bodies}")
            
            if num_dofs == 0:
                raise RuntimeError(f"加载的URDF没有DOF（自由度）: {asset_path}")
                
        except Exception as e:
            print(f"[URDF] ❌ 加载失败: {e}")
            print(f"[URDF] 资产根目录: {asset_root}")
            print(f"[URDF] URDF文件: {asset_file}")
            print(f"[URDF] 完整路径: {asset_path}")
            raise

        # 获取关键连杆的刚体索引（从 asset 获取，不是从 actor）
        num_bodies = self.gym.get_asset_rigid_body_count(self.ur10_asset)
        
        # 查找所有重要连杆的索引
        # UR10e主要链路：base_link -> shoulder_link -> upper_arm_link -> forearm_link -> wrist_1_link
        link_names_to_find = ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link']
        self.link_indices = {}  # 存储每个连杆的索引
        
        # 首先打印所有刚体名称（调试用）
        print(f"[URDF] 所有刚体 ({num_bodies}个):")
        for i in range(num_bodies):
            name = self.gym.get_asset_rigid_body_name(self.ur10_asset, i)
            print(f"  {i}: {name}")
        
        # 查找目标连杆
        for link_name in link_names_to_find:
            found = False
            for i in range(num_bodies):
                name = self.gym.get_asset_rigid_body_name(self.ur10_asset, i)
                if link_name in name:
                    self.link_indices[link_name] = i
                    print(f"[URDF] ✅ 找到 {link_name}: 索引 {i}")
                    found = True
                    break
            if not found:
                print(f"[URDF] ⚠️  未找到 {link_name}")
        
        # 获取wrist_3_link作为末端执行器（向后兼容）
        self.wrist_body_index = self.link_indices.get('wrist_1_link', None)
        if self.wrist_body_index is None:
            # 回退：尝试查找任何wrist连杆
            for i in range(num_bodies):
                name = self.gym.get_asset_rigid_body_name(self.ur10_asset, i)
                if 'wrist' in name.lower() or 'tool' in name.lower():
                    self.wrist_body_index = i
                    print(f"[URDF] ⚠️  回退：使用 {name} (索引 {i})")
                    break

        # 为每个环境创建机械臂actor（参考manipulator_env_gym.py）
        self.robot_handles = []
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        for i in range(self.num_envs):
            # 创建actor（collision_group=i，每个环境独立）
            actor_handle = self.gym.create_actor(self.envs[i], self.ur10_asset, robot_pose, f"UR10_{i}", i, 0, 0)
            if actor_handle is None:
                raise RuntimeError(f"Failed to create UR10e actor for env {i}")
            self.robot_handles.append(actor_handle)
        
        # 为了向后兼容，保留self.robot_handle指向第一个环境的handle
        self.robot_handle = self.robot_handles[0] if self.num_envs > 0 else None
        print(f"[URDF] 机械臂actor创建成功: {self.num_envs}个环境")
        
        # 获取DOF数量（在设置DOF属性之前，所有环境使用相同的asset，所以DOF数量相同）
        self.dof_count = self.gym.get_actor_dof_count(self.envs[0], self.robot_handles[0])
        
        # 为每个环境配置DOF属性（位置控制模式：关节角度增量控制）
        # 参考 ur5e_DDPG_trajectory_planning_template: 每个关节有独立的限位范围和PD控制器参数
        # 注意：UR10e的质量和惯性比UR5e大约1.5-2倍，需要相应调整PD控制器参数
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.robot_handles[0])
        
        # 从配置文件读取PD控制器参数（如果提供），否则使用默认值
        if cfg is not None and 'env' in cfg:
            env_cfg = cfg['env']
            stiffness_base = env_cfg.get('stiffness_base', 8000.0)
            damping_base = env_cfg.get('damping_base', 200.0)
            stiffness_wrist = env_cfg.get('stiffness_wrist', 7000.0)
            damping_wrist = env_cfg.get('damping_wrist', 200.0)
        else:
            # 默认值（参考 ur5e_DDPG_1_3.py: UR5e参数前3个关节 Kp=3500, Kd=100; 后3个关节 Kp=3000, Kd=100）
            # UR10e调整: 考虑到UR10e的质量和惯性更大（1.5-2倍），Kp增加25%，Kd增加20%
            stiffness_base = 8000.0
            damping_base = 200.0
            stiffness_wrist = 7000.0
            damping_wrist = 200.0
        
        # 前3个关节（shoulder_pan, shoulder_lift, elbow）
        dof_props['stiffness'][0] = stiffness_base
        dof_props['damping'][0] = damping_base
        dof_props['stiffness'][1] = stiffness_base
        dof_props['damping'][1] = damping_base
        dof_props['stiffness'][2] = stiffness_base
        dof_props['damping'][2] = damping_base
        # 后3个关节（wrist_1-3）
        for j in range(3, self.dof_count):
            dof_props['stiffness'][j] = stiffness_wrist
            dof_props['damping'][j] = damping_wrist
        
        # 使用URDF定义的实际关节限位（UR10e）
        # 参考 ur10e.urdf: 每个关节的实际限位范围
        # 索引0: shoulder_pan_joint
        dof_props['upper'][0] = 2 * math.pi  # [-2π, 2π]
        dof_props['lower'][0] = -2 * math.pi
        # 索引1: shoulder_lift_joint
        dof_props['upper'][1] = 2 * math.pi  # [-2π, 2π]（URDF定义，不是UR5e的[-π, 0]）
        dof_props['lower'][1] = -2 * math.pi
        # 索引2: elbow_joint
        dof_props['upper'][2] = math.pi  # [-π, π]（URDF定义，不是UR5e的[-π/2, π/2]）
        dof_props['lower'][2] = -math.pi
        # 索引3-5: wrist_1, wrist_2, wrist_3_joint
        for j in range(3, self.dof_count):
            dof_props['upper'][j] = 2 * math.pi  # [-2π, 2π]
            dof_props['lower'][j] = -2 * math.pi
        
        # 🎯 关键：确保DOF模式是位置控制
        dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
        
        # 🎯 重要：确保所有DOF都启用了驱动器（参考ur5e模板）
        # ur5e模板中通过set_torque_servo启用驱动器
        # Isaac Gym中通过设置stiffness和damping启用PD控制器
        # 如果stiffness=0，PD控制器不会工作
        for j in range(self.dof_count):
            if dof_props['stiffness'][j] == 0:
                print(f"[WARNING] DOF {j} stiffness is 0, setting default values")
                if j < 3:
                    dof_props['stiffness'][j] = stiffness_base  # 使用配置文件的值
                    dof_props['damping'][j] = damping_base
                else:
                    dof_props['stiffness'][j] = stiffness_wrist  # 使用配置文件的值
                    dof_props['damping'][j] = damping_wrist
        
        # 为每个环境设置DOF属性
        for i in range(self.num_envs):
            self.gym.set_actor_dof_properties(self.envs[i], self.robot_handles[i], dof_props)
        
        print(f"[URDF] DOF属性已设置: mode=POS ({self.num_envs}个环境)")
        print(f"[URDF] PD控制器参数（UR10e优化，参考ur5e模板）: shoulder_pan/lift/elbow Kp={stiffness_base} Kd={damping_base}, wrist_1-3 Kp={stiffness_wrist} Kd={damping_wrist}")
        print(f"[URDF] 说明: 匹配ur5e模板的速度（delta_time=0.034s，动作范围0.1 rad/step，16仿真步骤）")
        print(f"[URDF] 关节限位（URDF定义）: shoulder_pan=[-2π,2π], shoulder_lift=[-2π,2π], elbow=[-π,π], wrist=[-2π,2π]")
        
        # 🎯 验证DOF属性设置（调试用，只验证第一个环境）
        verify_props = self.gym.get_actor_dof_properties(self.envs[0], self.robot_handles[0])
        print(f"[URDF] 验证 - driveMode: {verify_props['driveMode']}")
        print(f"[URDF] 验证 - stiffness: {verify_props['stiffness']}")
        print(f"[URDF] 验证 - damping: {verify_props['damping']}")
        
        # 关键：使用tensor方式设置初始关节位置（完全参考manipulator_env_gym.py的_set_initial_arm_pose）
        # 获取DOF state tensor（如果还没有）
        if not hasattr(self, 'dof_states') or self.dof_states is None:
            dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
            self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        
        # 为每个环境设置初始关节位置为0，零速度（参考manipulator_env_gym.py）
        # 注意：初始位置0对所有关节都是有效的（在限位范围内）
        for i in range(self.num_envs):
            start_idx = i * self.dof_count
            end_idx = (i + 1) * self.dof_count
            self.dof_states[start_idx:end_idx, 0] = 0.0  # 位置
            self.dof_states[start_idx:end_idx, 1] = 0.0  # 速度
        
        # 使用tensor方式应用到仿真（与manipulator_env_gym.py一致）
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
        print(f"[URDF] 使用tensor方式设置初始关节位置为0，速度为0 ({self.num_envs}个环境)")
        
        # 运行几步simulation让机械臂稳定（参考manipulator_env_gym.py）
        for _ in range(10):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        
        # 刷新状态（与manipulator_env_gym.py一致）
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        print(f"[URDF] 运行10步simulation让机械臂稳定，状态已刷新")
        
        num_bodies = self.gym.get_asset_rigid_body_count(self.ur10_asset)
        print(f"[URDF] ✅ 机械臂actor已创建: {self.num_envs}个环境, 每个环境{num_bodies}个刚体")
        
        # 强制刷新状态
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # 为每个环境创建 ArUco 标记 (作为目标) - 参考manipulator_env_gym.py
        # ArUco 标记是一个正方形薄板，0.1m x 0.1m x 0.001m
        # 注意：将其设置得更明显以便观察（使用较亮的颜色）
        aruco_options = gymapi.AssetOptions()
        aruco_options.fix_base_link = True
        aruco_options.disable_gravity = True
        aruco_asset = self.gym.create_box(self.sim, 0.1, 0.1, 0.001, aruco_options)
        
        self.target_handles = []
        # ArUco 位置将在 reset 时设置，这里先创建一个临时位置
        aruco_pos = gymapi.Vec3(0.8, 0.0, 0.3)  # 临时位置
        aruco_rot = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        target_pose = gymapi.Transform(aruco_pos, aruco_rot)
        
        for i in range(self.num_envs):
            target_handle = self.gym.create_actor(
                self.envs[i], aruco_asset, target_pose, f"aruco_{i}", i, 0
            )
            self.target_handles.append(target_handle)
            
            # 设置 ArUco 颜色 (使用白色或浅色使其更明显)
            self.gym.set_rigid_body_color(
                self.envs[i], target_handle, 0, 
                gymapi.MESH_VISUAL, 
                gymapi.Vec3(1.0, 1.0, 1.0)  # 白色，更容易看到
            )
        
        # 为了向后兼容，保留self.target_handle指向第一个环境的handle
        self.target_handle = self.target_handles[0] if self.num_envs > 0 else None
        
        # 为每个环境创建相机并附加到 wrist_3_link
        cam_props = gymapi.CameraProperties()
        cam_props.width = img_w
        cam_props.height = img_h
        cam_props.horizontal_fov = math.degrees(self.fov)
        # 注意：当 use_gpu_pipeline=False 时，不要启用 enable_tensors
        
        self.camera_handles = []
        self.wrist_body_handles = []
        
        # 🎯 从配置读取相机参数
        if cfg is not None and 'env' in cfg:
            env_cfg = cfg['env']
            self.aruco_camera_enabled = env_cfg.get('aruco_camera_enabled', True)  # 默认启用
            self.aruco_render_freq = env_cfg.get('aruco_render_freq', 5)  # 默认每5步渲染一次
        else:
            self.aruco_camera_enabled = True
            self.aruco_render_freq = 5
        
        # 🎯 关键修复：根据aruco_camera_enabled决定是否创建相机传感器
        # 这样可以避免 render_all_camera_sensors() 导致的段错误
        if not self.aruco_camera_enabled:
            print(f"[Camera] aruco_camera_enabled=False，跳过相机传感器创建")
            # 不创建相机传感器，设置handles为None
            self.camera_handles = [None] * self.num_envs
            self.camera_handle = None
            self.wrist_body_handles = [None] * self.num_envs
            self.wrist_body_handle = None
        elif self.enable_viewer:
            print(f"[Camera] Viewer已启用，跳过相机传感器创建（仅用于可视化机械臂和障碍物）")
            # 不创建相机传感器，设置handles为None
            self.camera_handles = [None] * self.num_envs
            self.camera_handle = None
            self.wrist_body_handles = [None] * self.num_envs
            self.wrist_body_handle = None
        else:
            # 不启用viewer时，正常创建相机传感器用于深度渲染
            for i in range(self.num_envs):
                # 🎯 修复：相机传感器是必需的，如果创建失败应该抛出错误
                try:
                    camera_handle = self.gym.create_camera_sensor(self.envs[i], cam_props)
                    if camera_handle is None or camera_handle == -1:
                        raise ValueError(f"Failed to create camera sensor for environment {i}: got invalid handle {camera_handle}. Camera sensors are required for depth rendering and voxel mapping.")
                    self.camera_handles.append(camera_handle)
                except Exception as e:
                    print(f"[Error] Failed to create camera sensor for environment {i}: {e}")
                    raise RuntimeError(f"Cannot continue training without camera sensors. Please check GPU/graphics support and ensure graphics_device_id is set correctly.")
            
                # 获取 wrist body handle（用于附加相机）
                wrist_body_handle = self.gym.get_actor_rigid_body_handle(self.envs[i], self.robot_handles[i], self.wrist_body_index)
                if wrist_body_handle is None or wrist_body_handle == -1:
                    raise ValueError(f"Failed to get wrist body handle for environment {i}: got invalid handle {wrist_body_handle}")
                self.wrist_body_handles.append(wrist_body_handle)
                
                # 局部相机位姿（末端坐标系），朝 +X，略微抬高
                local_T = gymapi.Transform()
                local_T.p = gymapi.Vec3(0.0, 0.0, 0.05)
                local_T.r = gymapi.Quat(0, 0, 0, 1)
                
                # 🎯 修复：相机附加是必需的，如果失败应该抛出错误
                try:
                    self.gym.attach_camera_to_body(
                        camera_handle, self.envs[i], wrist_body_handle,
                        local_T, gymapi.FOLLOW_TRANSFORM
                    )
                except Exception as e:
                    print(f"[Error] Failed to attach camera to body for environment {i}: {e}")
                    print(f"[Error] camera_handle={camera_handle}, env={self.envs[i]}, wrist_body_handle={wrist_body_handle}")
                    raise RuntimeError(f"Cannot continue training without properly attached camera sensors.")
            
            # 为了向后兼容，保留self.camera_handle指向第一个环境的handle
            self.camera_handle = self.camera_handles[0] if self.num_envs > 0 else None
            self.wrist_body_handle = self.wrist_body_handles[0] if self.num_envs > 0 else None

        # 获取刚体状态 tensor（用于高效读取）
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        
        # 获取 DOF state tensor（用于高效读取关节状态）
        # 注意：如果已经在前面获取过（设置初始姿态时），这里检查是否需要重新获取
        if not hasattr(self, 'dof_states') or self.dof_states is None:
            dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
            self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        
        # 获取 actor root state tensor（用于设置 actor 变换）
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        
        # 计算每个环境的 actors 索引（参考manipulator_env_gym.py）
        # 多环境下，每个环境的actors顺序：robot (0), target (1), static_obstacles (2, 3, ...), dynamic_obstacles (...)
        # 每个环境的actors数量：1 (robot) + 1 (target) + num_obstacles (static) + num_dynamic_obstacles (dynamic)
        self.actors_per_env = 2 + self.num_obstacles + self.num_dynamic_obstacles  # robot + target + static obstacles + dynamic obstacles
        
        # 计算每个环境的robot和target在root_states中的索引
        self.robot_root_indices = torch.arange(self.num_envs, device=self.device) * self.actors_per_env
        self.target_root_indices = torch.arange(self.num_envs, device=self.device) * self.actors_per_env + 1
        
        # 计算 wrist 在 rigid_body_states 中的索引
        # 多环境下：rigid_body_states shape为 [num_envs * num_bodies, 13]
        # 每个环境的wrist body索引 = env_idx * num_bodies + wrist_body_index
        num_bodies = self.gym.get_asset_rigid_body_count(self.ur10_asset)
        self.wrist_state_indices = torch.arange(self.num_envs, device=self.device) * num_bodies + self.wrist_body_index

        # torch 状态 - 支持多环境
        self.ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat = torch.zeros((self.num_envs, 4), device=self.device)  # 末端姿态四元数
        self.ee_quat[:, 3] = 1.0  # 初始化为单位四元数 [0,0,0,1]
        
        # 用于存储最后有效的位姿值（当读取到 NaN 时使用）
        self._last_valid_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_valid_ee_pos[:, 2] = 0.5  # 默认z=0.5
        self._last_valid_ee_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._last_valid_ee_quat[:, 3] = 1.0  # 默认单位四元数
        
        # 临时target位置，将在reset时更新
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pos[:, 0] = target_pose.p.x
        self.target_pos[:, 1] = target_pose.p.y
        self.target_pos[:, 2] = target_pose.p.z
        
        self.progress = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        
        # 渐进式安全奖励相关状态（每个环境独立）
        self._should_terminate = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._termination_reason = [None] * self.num_envs
        self.danger_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        
        # 障碍物位置和半径
        # 注意：真实位置用于碰撞检测和reset，但不应该直接用于观测
        self.obstacle_positions = torch.zeros((self.num_obstacles, 3), device=self.device)
        self.obstacle_radii = torch.zeros((self.num_obstacles,), device=self.device)
        self.obstacle_actual_radii = []  # 存储实际创建的障碍物半径
        self.obstacle_positions_np = []  # 存储障碍物位置（numpy格式，用于跟踪）
        
        # LiDAR参数（从配置读取，参考无人机环境）
        if cfg is not None and 'env' in cfg:
            env_cfg = cfg['env']
            self.use_lidar_input = env_cfg.get('use_lidar_input', True)  # 默认启用LiDAR
            self.lidar_range = env_cfg.get('lidar_range', 3.0)  # LiDAR最大量程（米）
            self.lidar_vfov = env_cfg.get('lidar_vfov', [-10.0, 10.0])  # 垂直视场角（度）
            self.lidar_vbeams = env_cfg.get('lidar_vbeams', 4)  # 垂直光束数
            self.lidar_hres = env_cfg.get('lidar_hres', 10.0)  # 水平分辨率（度）
        else:
            self.use_lidar_input = True
            self.lidar_range = 3.0
            self.lidar_vfov = [-10.0, 10.0]
            self.lidar_vbeams = 4
            self.lidar_hres = 10.0
        
        # 计算水平光束数
        self.lidar_hbeams = int(360 / self.lidar_hres)  # 例如：360/10=36
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)  # (h_beams, v_beams)
        
        # LiDAR射线方向（将在首次调用时初始化）
        self.ray_directions = None  # [h_beams, v_beams, 3] 局部坐标系中的射线方向
        self.lidar_scan = None  # [num_envs, 1, h_beams, v_beams] LiDAR扫描数据
        
        # 体素地图相关（已废弃，保留用于向后兼容）
        self.voxel_grid_origin = None  # 体素网格原点
        self.voxel_res = None  # 体素分辨率
        self.current_voxel = None  # 当前体素地图（用于障碍物距离计算）

        # 奖励函数相关的状态变量（每个环境独立）
        self.prev_action = None  # 将在step中处理batch
        self.prev_discovery_state = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.prev_dist = None  # 将在step中处理batch
        self.prev_visit_key = None  # 将在step中处理batch
        # 末端转动奖励相关：用于跟踪上一次的末端四元数
        self.prev_ee_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.prev_ee_quat[:, 3] = 1.0  # 初始化为单位四元数 [x=0, y=0, z=0, w=1]
        self.wrist_rotation_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)  # 连续转动步数计数器
        
        # Z轴向上探索奖励相关：用于跟踪上一次的末端位置（z坐标）
        self._prev_ee_pos_upward = None  # 将在首次调用时初始化

        # 生成随机球形障碍（静态）
        self._spawn_or_reset_obstacles()
        
        # 🎯 注意：动态障碍物不在__init__中创建，因为此时num_dynamic_obstacles可能还是默认值0
        # 动态障碍物将在配置设置完成后由train.py调用_spawn_dynamic_obstacles()创建

        # 创建 viewer（如果启用）- 必须在所有actor创建之后
        if self.enable_viewer:
            self._create_viewer()

        self.reset()
        
        # 如果启用了可视化，立即渲染一次以确保能看到场景
        if self.enable_viewer and self.viewer is not None:
            # 重要：必须调用 poll_viewer_events 来处理窗口事件
            self.gym.poll_viewer_events(self.viewer)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            
            # 打印应该看到的物体信息
            print(f"[Visualization] 初始渲染完成")
            print(f"[Visualization] 应该能看到以下物体:")
            print(f"  - ArUco标记 (深灰色薄板，位置: 约 {self.target_pos[0,0]:.2f}, {self.target_pos[0,1]:.2f}, {self.target_pos[0,2]:.2f})")
            print(f"  - 蓝色障碍物球体 (共 {len(self.obstacle_handles)} 个)")
            # 🎯 注意：动态障碍物信息在_spawn_dynamic_obstacles()中打印
            if self.num_dynamic_obstacles > 0 and len(self.dynamic_obstacle_handles) > 0:
                print(f"  - 红色动态障碍物球体 (共 {len(self.dynamic_obstacle_handles)} 个)")
            print(f"[Visualization] 提示：如果看不到窗口，请检查是否有新的窗口弹出，或者尝试Alt+Tab切换窗口")
            print(f"[Visualization] 窗口操作：鼠标左键拖动旋转视角，右键拖动平移，滚轮缩放")

    @property
    def device(self):
        return self._device

    def _create_viewer(self):
        """创建 Isaac Gym 可视化窗口"""
        if not self.enable_viewer:
            return
        
        try:
            # 创建 viewer
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("[Warning] 无法创建 Isaac Gym 视口，可能是无头模式或缺少图形支持")
                self.enable_viewer = False
                return
            
            # 设置相机位置和目标（确保能清楚地看到机械臂）
            # 机械臂在 (0, 0, 0)，相机应该从侧面/上方观看
            cam_pos = gymapi.Vec3(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2])
            cam_target = gymapi.Vec3(self.camera_target[0], self.camera_target[1], self.camera_target[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            
            print(f"[Visualization] ✅ Isaac Gym 视口已创建")
            
            # 立即处理一次事件以确保窗口显示
            self.gym.poll_viewer_events(self.viewer)
        except Exception as e:
            print(f"[Warning] 创建 viewer 时出错: {e}")
            self.enable_viewer = False
            self.viewer = None
    
    def _update_viewer_camera(self):
        """更新viewer相机视角（在reset后调用，确保视角正确）"""
        if not self.enable_viewer or self.viewer is None:
            return
        
        try:
            cam_pos = gymapi.Vec3(self.camera_pos[0], self.camera_pos[1], self.camera_pos[2])
            cam_target = gymapi.Vec3(self.camera_target[0], self.camera_target[1], self.camera_target[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        except Exception as e:
            print(f"[Warning] 更新相机视角时出错: {e}")

    def render(self, render_freq: int = 1):
        """更新 Isaac Gym 可视化窗口
        
        Args:
            render_freq: 渲染频率（每N步渲染一次，1=每步都渲染）
        """
        if not self.enable_viewer or self.viewer is None:
            return
        
        # 检查是否需要渲染（根据频率）
        self.render_step_count += 1
        if self.render_step_count % render_freq != 0:
            return
        
        try:
            # 🎯 参考 robotic-powder-weighing-main 的调用顺序
            # 检查 viewer 是否关闭（在事件处理之前）
            try:
                if self.gym.query_viewer_has_closed(self.viewer):
                    print("[Visualization] Viewer 窗口已关闭")
                    self.enable_viewer = False
                    return
            except Exception as e:
                print(f"[Error] query_viewer_has_closed() failed: {e}")
                self.enable_viewer = False
                return
            
            # 更新图形（参考代码顺序：simulate -> fetch_results -> step_graphics -> draw_viewer）
            # 注意：simulate 和 fetch_results 已经在 _simulate_and_fetch() 中完成
            # 注意：render_depth()可能已经调用了step_graphics()，但为了确保viewer正确显示，这里也调用一次
            # step_graphics()可以安全地多次调用，不会造成问题
            try:
                self.gym.step_graphics(self.sim)
            except Exception as e:
                print(f"[Error] step_graphics() failed: {e}")
                self.enable_viewer = False
                return
            
            # 绘制到viewer（参考代码使用 True 阻塞模式，但在循环中调用）
            # 🎯 修复：参考 robotic-powder-weighing-main 使用 True（阻塞模式）
            try:
                # 验证 viewer 和 sim 是否有效
                if self.viewer is None or self.sim is None:
                    print(f"[Warning] viewer or sim is None, skipping draw_viewer")
                    self.enable_viewer = False
                    return
                
                # 🎯 修复：使用 False（非阻塞模式），避免阻塞训练循环
                # 第一次渲染时使用 True 确保窗口显示
                blocking = (self.render_step_count == render_freq)  # 第一次渲染时阻塞
                self.gym.draw_viewer(self.viewer, self.sim, blocking)
                
                # 处理viewer事件（必须在draw_viewer之后调用）
                self.gym.poll_viewer_events(self.viewer)
                    
            except Exception as e:
                # 如果draw_viewer失败，禁用可视化但继续训练
                print(f"[Warning] draw_viewer失败，禁用可视化: {e}")
                self.enable_viewer = False
                return
            
            # 可视化相机视锥（红色线框）- 只绘制指定环境的视锥
            # 🎯 临时禁用以调试段错误
            if True:  # 暂时禁用以调试
                try:
                    self._draw_camera_frustum(env_idx=self.visualize_env_index)
                except Exception as e:
                    # 静默失败，不影响训练
                    pass
            
            # 🎯 参考代码在 draw_viewer 后调用 sync_frame_time（可选，用于同步帧时间）
            # 注意：在训练循环中通常不需要，因为我们已经通过 render_freq 控制频率
            # 如果启用，可能导致训练变慢，所以注释掉
            # try:
            #     self.gym.sync_frame_time(self.sim)
            # except Exception as e:
            #     pass  # sync_frame_time 失败不影响训练
            
        except Exception as e:
            # 如果渲染出错，禁用可视化但继续训练
            print(f"[Warning] 渲染时出错: {e}")
            self.enable_viewer = False

    def _draw_camera_frustum(self, near: float = 0.1, far: float = 0.6, env_idx: int = None):
        """在viewer中绘制相机视锥线框（红色）。
        说明：只绘制指定环境的相机视锥（默认使用 visualize_env_index）。
        """
        if not self.enable_viewer or self.viewer is None:
            return
        
        # 🎯 只绘制指定环境的相机视锥
        if env_idx is None:
            env_idx = self.visualize_env_index
        
        # 确保环境索引有效
        if env_idx < 0 or env_idx >= self.num_envs:
            env_idx = 0
        
        # 获取指定环境的 env 对象
        env = self.envs[env_idx] if hasattr(self, 'envs') and self.envs is not None and env_idx < len(self.envs) else self.env
        if env is None:
            return
        
        # 🎯 修复：安全地获取指定环境的相机位姿
        try:
            cam_T = self.camera_pose(env_idx=env_idx)  # 4x4
            if cam_T is None or torch.isnan(cam_T).any() or torch.isinf(cam_T).any():
                return
            R = cam_T[:3, :3]
            t = cam_T[:3, 3]
        except Exception as e:
            print(f"[Warning] Failed to get camera pose in _draw_camera_frustum: {e}")
            return
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']
        W = self.img_w
        H = self.img_h

        def pixel_to_dir(u: float, v: float):
            # 针孔模型：Z前、X右、Y下
            z = 1.0
            x = (u - cx) / fx
            y = (v - cy) / fy
            d = torch.tensor([x, y, z], device=self.device, dtype=torch.float32)
            d = d / (torch.norm(d) + 1e-9)
            return d

        # 四个角像素（左上、右上、右下、左下）
        corners_px = [
            (0.0, 0.0),
            (W - 1.0, 0.0),
            (W - 1.0, H - 1.0),
            (0.0, H - 1.0),
        ]

        near_pts = []
        far_pts = []
        for (u, v) in corners_px:
            d = pixel_to_dir(u, v)
            p_near_cam = d * near
            p_far_cam = d * far
            p_near_world = R @ p_near_cam + t
            p_far_world = R @ p_far_cam + t
            near_pts.append(p_near_world)
            far_pts.append(p_far_world)

        cam_pos = t
        # 线段顶点列表（每条线2个点）
        lines = []
        # 从相机到近截面四角
        for i in range(4):
            lines.append(cam_pos)
            lines.append(near_pts[i])
        # 近截面边框
        for i in range(4):
            lines.append(near_pts[i])
            lines.append(near_pts[(i + 1) % 4])
        # 远截面边框
        for i in range(4):
            lines.append(far_pts[i])
            lines.append(far_pts[(i + 1) % 4])
        # 近远连接线
        for i in range(4):
            lines.append(near_pts[i])
            lines.append(far_pts[i])

        num_lines = len(lines) // 2
        if num_lines == 0:
            return
        
        # 🎯 修复：安全地堆叠张量和转换为numpy
        try:
            # 检查所有线条点是否有效
            for line_pt in lines:
                if not isinstance(line_pt, torch.Tensor) or torch.isnan(line_pt).any() or torch.isinf(line_pt).any():
                    print(f"[Warning] Invalid line point in _draw_camera_frustum")
                    return
            
            # 转numpy float32
            verts = torch.stack(lines).detach().cpu().numpy().astype('float32')
            
            # 检查 verts 是否有效
            if verts.shape[0] != 2 * num_lines or verts.shape[1] != 3:
                print(f"[Warning] Invalid verts shape in _draw_camera_frustum: {verts.shape}")
                return
            
            # 统一红色
            colors = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).repeat(2 * num_lines).view(-1, 3).cpu().numpy().astype('float32')
            
            if colors.shape[0] != 2 * num_lines or colors.shape[1] != 3:
                print(f"[Warning] Invalid colors shape in _draw_camera_frustum: {colors.shape}")
                return
        except Exception as e:
            print(f"[Warning] Failed to prepare frustum data: {e}")
            return

        # 清理旧线并添加
        try:
            self.gym.clear_lines(self.viewer)
        except Exception as e:
            print(f"[Warning] clear_lines failed: {e}")
            pass
        # 使用gymutil绘制或底层API添加
        try:
            self.gym.add_lines(self.viewer, env, num_lines, verts, colors)
        except Exception as e:
            print(f"[Warning] add_lines failed: {e}")
            # 回退到gymutil接口（若可用）
            try:
                import gymutil
                gymutil.draw_lines(verts, colors, self.gym, self.viewer, env)
            except Exception as e2:
                print(f"[Warning] gymutil.draw_lines also failed: {e2}")
                pass

    def _simulate_and_fetch(self):
        """
        执行仿真步骤并获取结果
        
        参考 ur5e_DDPG_trajectory_planning_template (ur5e_DDPG_1_3.py, line 323-326):
        - MuJoCo: while (time - time_prev < 1.0 / 60.0): mj.mj_step(model, data)
        - 每次RL step执行约16-17个仿真步骤（dt=0.001，累积到1/60秒）
        - 这确保PD控制器有足够时间响应目标位置变化
        
        Isaac Gym对应实现：
        - sim_params.dt = 1.0 / 60.0（每个仿真步骤约16.67ms）
        - 每次RL step执行1个仿真步骤（对应1/60秒）即可，因为每个仿真步骤已经足够大
        - 但为了确保PD控制器响应，可以执行多个步骤
        """
        # 🎯 关键：参考ur5e模板的仿真循环逻辑
        # ur5e模板：dt=0.001，循环直到累积时间达到1/60秒（约16.67ms）
        # Isaac Gym：dt=1/60秒，所以1次仿真步骤 = 1/60秒
        # 但为了确保PD控制器充分响应，执行多个步骤（类似ur5e模板的多次迭代）
        
        # 方法1：执行1个步骤（对应1/60秒，与ur5e模板的时间对应）
        # num_sim_steps = 1
        
        # 方法2：执行多个步骤以确保PD控制器响应（更安全）
        # 参考ur5e模板：约16步（dt=0.001，累积到1/60秒）
        # 但Isaac Gym的dt已经是1/60，所以这里执行1步即可
        # 🎯 参考ur5e模板：delta_time=0.034秒，每个RL step执行约16步仿真（dt=0.001，累积到1/60秒）
        # 为了匹配ur5e模板的速度，需要增加仿真步骤数
        # Isaac Gym: dt=1/60≈0.0167秒，执行16步≈0.267秒（比ur5e的0.034秒更长，但能确保PD控制器充分响应）
        num_sim_steps = 16  # 每次RL step执行16个仿真步骤（对应约0.267秒，参考ur5e模板的执行方式）
        
        # 🎯 重要：在执行仿真前，确保DOF状态已刷新
        # 这样可以确保读取到最新的关节状态（虽然已经在step中设置目标）
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # 🎯 执行仿真步骤（参考ur5e模板的仿真循环）
        # 在ur5e模板中：每次循环执行 mj.mj_forward() 和 mj.mj_step()
        # 在Isaac Gym中：gym.simulate() 内部已经包含了前向动力学计算和仿真步骤
        # 🎯 关键修复：在每个仿真子步骤中更新动态障碍物位置
        # 这样可以确保障碍物位置不会被物理模拟重置
        for i in range(num_sim_steps):
            # 🎯 在每个仿真步骤之前更新动态障碍物位置
            if self.num_dynamic_obstacles > 0 and len(self.dynamic_obstacle_handles) > 0:
                try:
                    # 快速更新位置（不刷新，直接使用之前计算的位置）
                    self._update_dynamic_obstacle_positions_to_sim_quick()
                except Exception as e:
                    # 如果快速更新失败，使用完整更新
                    try:
                        self._update_dynamic_obstacle_positions_to_sim()
                    except:
                        pass  # 如果更新失败，继续仿真
            
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # 🎯 关键修复：在每个仿真步骤之后也更新动态障碍物位置
            # 这样可以覆盖物理模拟可能重置的位置
            if self.num_dynamic_obstacles > 0 and len(self.dynamic_obstacle_handles) > 0:
                try:
                    # 快速更新位置（覆盖物理模拟的结果）
                    self._update_dynamic_obstacle_positions_to_sim_quick()
                except:
                    pass  # 如果更新失败，继续仿真
        
        # 🎯 刷新状态（确保获取最新的仿真结果）
        # 参考ur5e模板：每次仿真后需要读取最新状态
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 🎯 注意：step_graphics 应该在 render() 或 render_depth() 中调用，而不是在这里
        # 参考 robotic-powder-weighing-main：只在 render loop 中调用 step_graphics

    def _read_wrist_pose(self):
        # 读取所有环境的 wrist_3_link 位姿作为末端位姿（位置+旋转）
        # 刷新 rigid body states
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 从 tensor 中提取所有环境的位置和四元数
        # rigid_body_states: [num_envs * num_bodies, 13] where 13 = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        # 注意：rigid_body_states 通常在 CPU 上，需要将索引转换为 CPU
        wrist_indices_cpu = self.wrist_state_indices.cpu().long()
        wrist_states = self.rigid_body_states[wrist_indices_cpu]  # [num_envs, 13]
        ee_pos_raw = wrist_states[:, :3].to(self.device)  # [num_envs, 3]
        ee_quat_raw = wrist_states[:, 3:7].to(self.device)  # [num_envs, 4], quat is [x, y, z, w] in Isaac Gym
        
        # 检查并修复位置数据（批量处理）
        valid_mask = ~(torch.isnan(ee_pos_raw).any(dim=1) | torch.isinf(ee_pos_raw).any(dim=1))
        self.ee_pos[valid_mask] = ee_pos_raw[valid_mask]
        self._last_valid_ee_pos[valid_mask] = ee_pos_raw[valid_mask].clone()
        
        # 对于无效的位置，使用最后有效的值或默认值
        invalid_mask = ~valid_mask
        if invalid_mask.any():
            # 使用最后有效的值
            self.ee_pos[invalid_mask] = self._last_valid_ee_pos[invalid_mask]
        
        # 检查并修复四元数数据（批量处理）
        valid_mask_quat = ~(torch.isnan(ee_quat_raw).any(dim=1) | torch.isinf(ee_quat_raw).any(dim=1))
        
        # 归一化有效的四元数
        quat_norms = torch.norm(ee_quat_raw[valid_mask_quat], dim=1, keepdim=True)
        normalized_quat = ee_quat_raw[valid_mask_quat].clone()
        # 避免除零
        nonzero_mask = quat_norms.squeeze() > 1e-6
        if nonzero_mask.any():
            normalized_quat[nonzero_mask] = normalized_quat[nonzero_mask] / quat_norms[nonzero_mask]
        # 对于零范数的四元数，使用单位四元数
        zero_mask = ~nonzero_mask
        if zero_mask.any():
            normalized_quat[zero_mask] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        
        self.ee_quat[valid_mask_quat] = normalized_quat
        self._last_valid_ee_quat[valid_mask_quat] = normalized_quat.clone()
        
        # 对于无效的四元数，使用最后有效的值
        invalid_mask_quat = ~valid_mask_quat
        if invalid_mask_quat.any():
            self.ee_quat[invalid_mask_quat] = self._last_valid_ee_quat[invalid_mask_quat]
    
    def _get_joint_positions(self) -> torch.Tensor:
        """
        获取当前关节位置（关节空间）
        
        Returns:
            joint_pos: [num_envs, 6] 关节角度（弧度）
                      - 顺序：shoulder_pan → shoulder_lift → elbow → wrist_1 → wrist_2 → wrist_3
                      - 坐标系：关节空间（Joint Space），Isaac Gym DOF顺序与URDF一致
        """
        # 刷新 DOF state tensor
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # 多环境下，dof_states shape: [num_envs * num_dofs, 2] where 2 = [pos, vel]
        joint_pos = self.dof_states[:, 0].to(self.device).view(self.num_envs, self.dof_count)  # [num_envs, dof_count]
        
        return joint_pos
    
    def _forward_kinematics(self, joint_angles: torch.Tensor) -> tuple:
        """
        正运动学：根据关节角度计算末端位置和姿态
        
        Args:
            joint_angles: [6] 关节角度（弧度）
        
        Returns:
            ee_pos: [3] 末端位置（世界坐标系）
            ee_rot: [3, 3] 末端旋转矩阵（世界坐标系）
            T_end: [4, 4] 末端变换矩阵
        """
        # UR10e DH 参数
        d = torch.tensor([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655], device=self.device)
        a = torch.tensor([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0], device=self.device)
        alpha = torch.tensor([math.pi/2, 0.0, 0.0, math.pi/2, -math.pi/2, 0.0], device=self.device)
        
        # DH变换
        def dh_transform(theta, d_i, a_i, alpha_i):
            ct = torch.cos(theta)
            st = torch.sin(theta)
            ca = torch.cos(alpha_i)
            sa = torch.sin(alpha_i)
            T = torch.zeros(4, 4, device=self.device)
            T[0, 0] = ct
            T[0, 1] = -st * ca
            T[0, 2] = st * sa
            T[0, 3] = a_i * ct
            T[1, 0] = st
            T[1, 1] = ct * ca
            T[1, 2] = -ct * sa
            T[1, 3] = a_i * st
            T[2, 1] = sa
            T[2, 2] = ca
            T[2, 3] = d_i
            T[3, 3] = 1.0
            return T
        
        # 累积变换（正运动学）
        T_cum = torch.eye(4, device=self.device)
        for i in range(6):
            T_i = dh_transform(joint_angles[i], d[i], a[i], alpha[i])
            T_cum = T_cum @ T_i
        
        T_end = T_cum
        ee_pos = T_end[:3, 3]  # 末端位置
        ee_rot = T_end[:3, :3]  # 末端旋转矩阵
        
        return ee_pos, ee_rot, T_end
    
    def _compute_ee_velocity_from_joints(self, joint_angles: torch.Tensor, joint_velocities: torch.Tensor) -> torch.Tensor:
        """
        根据关节角度和关节速度计算末端速度（使用雅可比矩阵）
        
        注意：这是在关节空间控制中，如果需要末端速度时的计算方法
        对于关节角度增量控制，通常不需要计算末端速度
        
        Args:
            joint_angles: [6] 关节角度（弧度）
            joint_velocities: [6] 关节速度（rad/s）
        
        Returns:
            ee_velocity: [6] 末端速度 [vx, vy, vz, wx, wy, wz]（基座坐标系）
        """
        J = self._compute_jacobian(joint_angles)
        ee_velocity = J @ joint_velocities
        return ee_velocity
    
    def _compute_jacobian(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        计算 UR10e 的几何雅可比矩阵（6x6）
        参考: visual_servo_py 中的速度转换逻辑，以及 ros1/navigation_runner/scripts/manipulator_utils.py
        
        注意：基于DH参数计算的雅可比矩阵，返回的是**基座坐标系**下的末端速度。
        这与 visual_servo_py 中发布的基座坐标系速度一致。
        
        Args:
            joint_angles: [6] 关节角度（弧度）
        
        Returns:
            J: [6, 6] 雅可比矩阵，将关节速度映射到基座坐标系下的末端速度 [vx, vy, vz, wx, wy, wz]
        """
        # UR10e DH 参数
        d = torch.tensor([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655], device=self.device)
        a = torch.tensor([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0], device=self.device)
        alpha = torch.tensor([math.pi/2, 0.0, 0.0, math.pi/2, -math.pi/2, 0.0], device=self.device)
        
        # 计算每个关节的变换矩阵
        def dh_transform(theta, d_i, a_i, alpha_i):
            ct = torch.cos(theta)
            st = torch.sin(theta)
            ca = torch.cos(alpha_i)
            sa = torch.sin(alpha_i)
            T = torch.zeros(4, 4, device=self.device)
            T[0, 0] = ct
            T[0, 1] = -st * ca
            T[0, 2] = st * sa
            T[0, 3] = a_i * ct
            T[1, 0] = st
            T[1, 1] = ct * ca
            T[1, 2] = -ct * sa
            T[1, 3] = a_i * st
            T[2, 1] = sa
            T[2, 2] = ca
            T[2, 3] = d_i
            T[3, 3] = 1.0
            return T
        
        # 累积变换
        T_list = []
        T_cum = torch.eye(4, device=self.device)
        for i in range(6):
            T_i = dh_transform(joint_angles[i], d[i], a[i], alpha[i])
            T_cum = T_cum @ T_i
            T_list.append(T_cum.clone())
        
        # 计算雅可比矩阵（解析法）
        J = torch.zeros(6, 6, device=self.device)
        T_end = T_list[-1]
        p_end = T_end[:3, 3]  # 末端位置
        
        for i in range(6):
            T_i = T_list[i]
            p_i = T_i[:3, 3]  # 第i个关节的位置
            z_i = T_i[:3, 2]   # 第i个关节的z轴方向
            
            # 线速度雅可比：v = z_i × (p_end - p_i) （旋转关节）
            if i < 6:  # 所有关节都是旋转关节
                # 🎯 修复：使用torch.cross并显式指定dim参数（PyTorch 2.4+要求）
                # z_i和(p_end - p_i)都是[3]形状的1D向量，需要先扩展为[1, 3]
                vec1 = z_i.unsqueeze(0)  # [1, 3]
                vec2 = (p_end - p_i).unsqueeze(0)  # [1, 3]
                cross_result = torch.cross(vec1, vec2, dim=1)  # [1, 3]
                J[:3, i] = cross_result.squeeze(0)  # [3]
            else:
                J[:3, i] = z_i  # 如果是移动关节
            
            # 角速度雅可比：w = z_i （旋转关节）
            J[3:, i] = z_i
        
        return J
    
    def _ee_velocity_to_joint_velocity(self, ee_velocity: torch.Tensor) -> torch.Tensor:
        """
        将末端速度转换为关节速度（使用雅可比矩阵伪逆）
        
        注意：此方法已废弃，因为现在使用关节角度增量控制，不需要从末端速度转换
        
        Args:
            ee_velocity: [6] 末端速度 [vx, vy, vz, wx, wy, wz]（世界坐标系）
        
        Returns:
            joint_velocity: [6] 关节速度
        """
        # 已废弃：关节角度增量控制不需要此方法
        # 保留此方法仅用于向后兼容或特殊用途
        # 🎯 修复：处理多环境情况，只使用第一个环境的关节角度
        joint_angles = self._get_joint_positions()  # [num_envs, 6]
        if joint_angles.ndim > 1:
            joint_angles = joint_angles[0]  # 只使用第一个环境 [6]
        J = self._compute_jacobian(joint_angles)
        
        # 使用伪逆计算关节速度：q_dot = J^+ @ v_ee
        J_pinv = torch.linalg.pinv(J)
        joint_velocity = J_pinv @ ee_velocity
        
        # 速度限幅（防止关节速度过大）
        max_joint_vel = 2.0  # rad/s
        joint_velocity = torch.clamp(joint_velocity, -max_joint_vel, max_joint_vel)
        
        return joint_velocity

    def reset(self):
        # 🎯 修复：为每个环境生成有效的目标位置（避开障碍物、基座、工作空间内）
        # 先刷新以获取最新状态
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        
        # 🎯 先重置障碍物位置（在生成目标位置之前）
        self._spawn_or_reset_obstacles(reset_only=True)
        # 更新障碍物位置和半径（必须先生成/重置障碍物）
        self._update_obstacle_positions()
        
        # 为每个环境生成有效的目标位置和角度
        for i in range(self.num_envs):
            # 🎯 使用验证函数生成安全的目标位置
            target_pos, yaw_deg = self._generate_valid_target_position(i, max_attempts=100)
            
            # 计算四元数（绕Z轴旋转）
            yaw_rad = torch.deg2rad(torch.tensor(yaw_deg, device=self.device))
            qx = 0.0
            qy = 0.0
            qz = torch.sin(yaw_rad / 2).item()
            qw = torch.cos(yaw_rad / 2).item()
            
            # 使用 root_state tensor 设置 target 位置和旋转
            target_idx = self.target_root_indices[i].cpu().item()  # 转换为CPU索引
            if target_idx < self.root_states.shape[0]:
                # 参考 manipulator_env_gym.py：先克隆，然后使用 torch.tensor 转换 numpy 值
                new_root_state = self.root_states[target_idx].clone()
                new_root_state[0:3] = target_pos
                new_root_state[3:7] = torch.tensor([qx, qy, qz, qw], device=self.device, dtype=torch.float32)
                new_root_state[7:13] = 0.0  # 速度设为0
                
                # 更新 root_states
                self.root_states[target_idx] = new_root_state
            
            # 更新target_pos
            self.target_pos[i] = target_pos

        # 应用所有更改（批量更新所有环境）
        # 🎯 使用统一的 _set_root_states_indexed 函数
        target_indices_cpu = self.target_root_indices.to(device='cpu', dtype=torch.int64)
        target_poses_cpu = torch.zeros((self.num_envs, 7), dtype=torch.float32, device='cpu')
        for i in range(self.num_envs):
            target_idx = self.target_root_indices[i].cpu().item()
            if target_idx < self.root_states.shape[0]:
                target_poses_cpu[i, 0:3] = self.root_states[target_idx, 0:3].cpu()
                target_poses_cpu[i, 3:7] = self.root_states[target_idx, 3:7].cpu()
        self._set_root_states_indexed(target_indices_cpu, target_poses_cpu, keep_vel=False)

        # 关节复位到 0（使用tensor方式，批量处理所有环境）
        for i in range(self.num_envs):
            start_idx = i * self.dof_count
            end_idx = (i + 1) * self.dof_count
            self.dof_states[start_idx:end_idx, 0] = 0.0  # 位置
            self.dof_states[start_idx:end_idx, 1] = 0.0  # 速度
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
        
        # 运行几步simulation让机械臂稳定（防止移动）
        for _ in range(5):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        
        # 刷新状态
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.progress.zero_()
        self.target_view_counter.zero_()
        self.target_discovered.zero_()
        self.steps_since_last_detection.zero_()
        self.last_detection_step.zero_()
        self.explored_voxels.clear()
        
        # 重置奖励函数相关状态
        self.prev_action = None
        self.prev_discovery_state.zero_()
        self.prev_dist = None
        self.prev_visit_key = None
        # 重置成功奖励标志位
        if hasattr(self, '_success_reward_given'):
            self._success_reward_given.zero_()
        else:
            self._success_reward_given = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 重置渐进式安全奖励相关状态
        self._should_terminate.zero_()
        self._termination_reason = [None] * self.num_envs
        self.danger_count.zero_()
        
        # 重置多角度观察状态（每个环境独立）
        self.observation_angles = [[] for _ in range(self.num_envs)]
        self.multiview_steps.zero_()
        
        # 重置末端转动奖励相关状态
        self.wrist_rotation_steps.zero_()
        
        # 重置Z轴向上探索奖励相关状态
        # 确保 _prev_ee_pos_upward 已初始化且 ee_pos 已读取
        if hasattr(self, 'ee_pos') and self.ee_pos is not None:
            if hasattr(self, '_prev_ee_pos_upward') and self._prev_ee_pos_upward is not None:
                self._prev_ee_pos_upward.copy_(self.ee_pos)
            else:
                self._prev_ee_pos_upward = self.ee_pos.clone()
        
        # 🎯 注意：障碍物重置已经在前面完成（在生成目标位置之前）
        
        # 强制执行几个物理步骤以确保 Isaac Gym 状态正确初始化
        # 这在 reset 后是必要的，因为 rigid body states 可能还没有更新
        for _ in range(3):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        
        # 🎯 确保动态障碍物位置正确更新到仿真
        if self.num_dynamic_obstacles > 0 and len(self.dynamic_obstacle_handles) > 0:
            # 注意：在reset时，动态障碍物可能已经创建，只需要更新位置
            # 但需要确保root_states已经刷新
            try:
                # 🎯 关键修复：GPU同步，确保状态一致
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                self._update_dynamic_obstacle_positions_to_sim()
                
                # 🎯 关键修复：更新后同步，确保状态已写入
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # 🎯 关键修复：标记动态障碍物已更新
                self._dynamic_obstacles_updated = True
                
                # 再次运行几步确保可视化更新
                for _ in range(2):
                    self.gym.simulate(self.sim)
                    self.gym.fetch_results(self.sim, True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if self.enable_viewer:
                        self.gym.step_graphics(self.sim)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
            except Exception as e:
                print(f"[Warning] Failed to update dynamic obstacles in reset(): {e}")
                import traceback
                traceback.print_exc()
        
        # 强制刷新 rigid body states 并确保更新末端位姿
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._read_wrist_pose()
        
        # 如果读取的值仍然是 NaN/inf，使用安全的默认值（批量处理）
        invalid_mask = torch.isnan(self.ee_pos).any(dim=1) | torch.isinf(self.ee_pos).any(dim=1)
        if invalid_mask.any():
            self.ee_pos[invalid_mask] = self._last_valid_ee_pos[invalid_mask]
        
        invalid_quat_mask = torch.isnan(self.ee_quat).any(dim=1) | torch.isinf(self.ee_quat).any(dim=1)
        if invalid_quat_mask.any():
            self.ee_quat[invalid_quat_mask] = self._last_valid_ee_quat[invalid_quat_mask]
        
        # 🎯 修复：在处理完 NaN/inf 后，更新 prev_ee_quat 为当前值（确保数据有效）
        if hasattr(self, 'ee_quat') and self.ee_quat is not None:
            if self.ee_quat.shape[0] == self.num_envs:
                self.prev_ee_quat.copy_(self.ee_quat)
            else:
                # 如果形状不匹配，重置为单位四元数
                self.prev_ee_quat.zero_()
                self.prev_ee_quat[:, 3] = 1.0
        
        # 更新viewer相机视角（如果启用可视化）
        self._update_viewer_camera()
        
        # 🎯 修复：在reset后立即显示viewer窗口（如果启用）
        if self.enable_viewer and self.viewer is not None:
            try:
                self.gym.poll_viewer_events(self.viewer)
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)  # 非阻塞模式，避免阻塞训练
            except Exception as e:
                print(f"[Warning] Failed to render viewer in reset(): {e}")
        
        # 🎯 标记图形状态已初始化（reset()中已经执行了仿真和step_graphics()）
        # 这样render_depth()就知道图形状态已经准备好
        self._graphics_initialized = True
        
        return self.observe()
    
    def _compute_exploration_coverage(self) -> float:
        """计算探索覆盖率（基于体素地图，简化估计）- 返回单个值用于所有环境"""
        # 简化实现：基于已访问的位置数量估计
        # 探索覆盖率基本基于时间进度和发现目标的概率
        # 使用第一个环境的进度和发现状态（简化实现）
        base_coverage = (self.progress[0].float() / self.max_steps).item()
        # 如果发现了目标，说明探索更有效
        discovery_bonus = 0.2 if self.target_discovered[0].item() else 0.0
        return min(1.0, base_coverage + discovery_bonus)
    
    def _compute_nearest_obstacle_distance(self, env_idx: int = 0, use_voxel: bool = True) -> float:
        """
        计算到最近障碍物的距离
        
        Args:
            env_idx: 环境索引（多环境支持）
            use_voxel: 如果True，从体素地图推断（正确方式）；如果False，使用真实位置（仅用于调试）
        
        Returns:
            到最近障碍物表面的距离（米）
        """
        if len(self.obstacle_handles) == 0:
            return 10.0  # 没有障碍物，返回很大的安全距离
        
        ee = self.ee_pos[env_idx]
        
        # 优先从体素地图推断障碍物距离（符合主动探索任务）
        if use_voxel and self.current_voxel is not None and self.voxel_grid_origin is not None and self.voxel_res is not None:
            return self._compute_nearest_obstacle_from_voxel(ee)
        
        # 🎯 修复：只检查该环境对应的障碍物
        # 环境 env_idx 的障碍物索引范围：[env_idx * num_obstacles, (env_idx + 1) * num_obstacles)
        obstacle_start_idx = env_idx * self.num_obstacles
        obstacle_end_idx = min((env_idx + 1) * self.num_obstacles, len(self.obstacle_handles))
        
        distances = []
        for i in range(obstacle_start_idx, obstacle_end_idx):
            if i >= len(self.obstacle_positions) or i >= len(self.obstacle_radii):
                continue
            obs_pos = self.obstacle_positions[i]
            obs_radius = self.obstacle_radii[i]
            dist_to_center = torch.norm(ee - obs_pos).item()
            dist_to_surface = max(0.0, dist_to_center - obs_radius.item())
            distances.append(dist_to_surface)
        
        if not distances:
            return 10.0
        
        return min(distances)
    
    def _compute_link_obstacle_distances_batch(self, use_voxel: bool = False) -> torch.Tensor:
        """
        批量计算5个关键连杆到障碍物的最短距离
        
        Args:
            use_voxel: 如果True，从体素地图推断；如果False，使用真实位置
        
        Returns:
            [num_envs, 5] 每个环境5个连杆到最近障碍物的距离（米）
        """
        N = self.num_envs
        distances = torch.zeros(N, 5, device=self.device)
        
        if len(self.obstacle_handles) == 0 and (self.num_dynamic_obstacles == 0 or self.dyn_obs_state is None):
            distances.fill_(10.0)  # 没有障碍物，返回安全距离
            return distances
        
        # 确保link_indices已初始化
        if not hasattr(self, 'link_indices') or len(self.link_indices) < 5:
            # 如果索引未初始化，使用末端位置作为替代
            # 使用简化方法：直接计算ee到障碍物的距离
            for env_idx in range(N):
                ee = self.ee_pos[env_idx]  # [3]
                min_dist = 10.0
                
                # 静态障碍物
                for i in range(env_idx * self.num_obstacles, min((env_idx + 1) * self.num_obstacles, len(self.obstacle_handles))):
                    if i >= len(self.obstacle_positions):
                        continue
                    obs_pos = self.obstacle_positions[i]
                    obs_radius = self.obstacle_radii[i].item()
                    dist = max(0.0, torch.norm(ee - obs_pos).item() - obs_radius)
                    min_dist = min(min_dist, dist)
                
                distances[env_idx, :] = min_dist
            return distances
        
        # 确保已刷新刚体状态
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # GPU同步（确保tensor数据已准备好）
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 获取5个关键连杆的位置
        link_names = ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link']
        num_bodies = self.gym.get_asset_rigid_body_count(self.ur10_asset)
        
        for link_idx, link_name in enumerate(link_names):
            if link_name not in self.link_indices:
                # 如果找不到该连杆，使用末端距离作为替代
                # 简化计算：直接使用ee到障碍物的距离
                for env_idx in range(N):
                    ee = self.ee_pos[env_idx]
                    min_dist = 10.0
                    for i in range(env_idx * self.num_obstacles, min((env_idx + 1) * self.num_obstacles, len(self.obstacle_handles))):
                        if i >= len(self.obstacle_positions):
                            continue
                        obs_pos = self.obstacle_positions[i]
                        obs_radius = self.obstacle_radii[i].item()
                        dist = max(0.0, torch.norm(ee - obs_pos).item() - obs_radius)
                        min_dist = min(min_dist, dist)
                    distances[env_idx, link_idx] = min_dist
                continue
            
            body_index = self.link_indices[link_name]
            
            # 为每个环境计算该连杆到障碍物的距离
            for env_idx in range(N):
                # 计算该环境该连杆的索引
                link_state_idx = env_idx * num_bodies + body_index
                
                if link_state_idx >= self.rigid_body_states.shape[0]:
                    distances[env_idx, link_idx] = 10.0
                    continue
                
                # 获取连杆位置
                link_pos = self.rigid_body_states[link_state_idx, :3].to(self.device)  # [3]
                
                # 计算到静态障碍物的距离
                min_dist = 10.0  # 默认安全距离
                
                # 静态障碍物
                obstacle_start_idx = env_idx * self.num_obstacles
                obstacle_end_idx = min((env_idx + 1) * self.num_obstacles, len(self.obstacle_handles))
                
                for i in range(obstacle_start_idx, obstacle_end_idx):
                    if i >= len(self.obstacle_positions) or i >= len(self.obstacle_radii):
                        continue
                    obs_pos = self.obstacle_positions[i]
                    obs_radius = self.obstacle_radii[i].item()
                    dist_to_surface = max(0.0, torch.norm(link_pos - obs_pos).item() - obs_radius)
                    min_dist = min(min_dist, dist_to_surface)
                
                # 动态障碍物
                try:
                    if (self.num_dynamic_obstacles > 0 and self.dyn_obs_state is not None and
                        self.dyn_obs_radii is not None):
                        
                        dyn_obstacle_start_idx = env_idx * self.num_dynamic_obstacles
                        dyn_obstacle_end_idx = (env_idx + 1) * self.num_dynamic_obstacles
                        
                        if (dyn_obstacle_end_idx <= self.dyn_obs_state.shape[0] and
                            dyn_obstacle_end_idx <= self.dyn_obs_radii.shape[0]):
                            
                            dyn_obs_positions = self.dyn_obs_state[dyn_obstacle_start_idx:dyn_obstacle_end_idx, :3]
                            dyn_obs_radii = self.dyn_obs_radii[dyn_obstacle_start_idx:dyn_obstacle_end_idx]
                            
                            if len(dyn_obs_positions) > 0:
                                # 确保 link_pos 是3维向量
                                if link_pos.shape[0] == 3:
                                    dist_to_centers = torch.norm(link_pos.unsqueeze(0) - dyn_obs_positions, dim=1)
                                    dist_to_surfaces = torch.clamp(dist_to_centers - dyn_obs_radii, min=0.0)
                                    min_dyn_dist = dist_to_surfaces.min().item()
                                    min_dist = min(min_dist, min_dyn_dist)
                except Exception as e:
                    pass  # 忽略动态障碍物计算错误
                
                distances[env_idx, link_idx] = min_dist
        
        return distances
    
    def _compute_nearest_obstacle_distance_batch(self, use_voxel: bool = True) -> torch.Tensor:
        """
        批量计算所有环境到最近障碍物的距离（优化版本）
        参考isaac-training：选择最近的N个动态障碍物进行检测
        
        Args:
            use_voxel: 如果True，从体素地图推断；如果False，使用真实位置
        
        Returns:
            [N] 每个环境到最近障碍物的距离（米）
        """
        N = self.num_envs
        distances = torch.zeros(N, device=self.device)
        
        if len(self.obstacle_handles) == 0 and (self.num_dynamic_obstacles == 0 or self.dyn_obs_state is None):
            distances.fill_(10.0)  # 没有障碍物，返回安全距离
            return distances
        
        # 🎯 优化：批量计算所有环境的障碍物距离
        if use_voxel and self.current_voxel is not None and self.voxel_grid_origin is not None and self.voxel_res is not None:
            # 从体素地图批量推断（当前实现是逐个环境）
            for i in range(N):
                distances[i] = self._compute_nearest_obstacle_from_voxel(self.ee_pos[i])
            return distances
        
        # 批量计算真实障碍物距离（静态+动态）
        ee_pos = self.ee_pos  # [N, 3]
        
        for env_idx in range(N):
            min_dist = 10.0  # 默认安全距离
            
            # 检查静态障碍物
            obstacle_start_idx = env_idx * self.num_obstacles
            obstacle_end_idx = min((env_idx + 1) * self.num_obstacles, len(self.obstacle_handles))
            
            for i in range(obstacle_start_idx, obstacle_end_idx):
                if i >= len(self.obstacle_positions) or i >= len(self.obstacle_radii):
                    continue
                obs_pos = self.obstacle_positions[i]  # [3]
                obs_radius = self.obstacle_radii[i].item()
                
                # 计算到障碍物表面的距离
                dist_to_center = torch.norm(ee_pos[env_idx] - obs_pos).item()
                dist_to_surface = max(0.0, dist_to_center - obs_radius)
                min_dist = min(min_dist, dist_to_surface)
            
            # 🎯 改进：检查动态障碍物（选择最近的N个，参考isaac-training）
            # 添加安全检查，避免段错误
            try:
                if (self.num_dynamic_obstacles > 0 and 
                    self.dyn_obs_state is not None and 
                    self.dyn_obs_radii is not None and
                    self.dyn_obs_state.shape[0] > 0 and
                    self.dyn_obs_radii.shape[0] > 0):
                    
                    dyn_obstacle_start_idx = env_idx * self.num_dynamic_obstacles
                    dyn_obstacle_end_idx = (env_idx + 1) * self.num_dynamic_obstacles
                    
                    # 确保索引在有效范围内
                    if (dyn_obstacle_end_idx > dyn_obstacle_start_idx and
                        dyn_obstacle_end_idx <= self.dyn_obs_state.shape[0] and
                        dyn_obstacle_end_idx <= self.dyn_obs_radii.shape[0]):
                        
                        dyn_obs_positions = self.dyn_obs_state[dyn_obstacle_start_idx:dyn_obstacle_end_idx, :3]  # [num_dyn_obs, 3]
                        dyn_obs_radii = self.dyn_obs_radii[dyn_obstacle_start_idx:dyn_obstacle_end_idx]  # [num_dyn_obs]
                        
                        # 确保形状正确
                        if (dyn_obs_positions.shape[0] > 0 and 
                            dyn_obs_positions.shape[1] == 3 and
                            dyn_obs_radii.shape[0] == dyn_obs_positions.shape[0]):
                            
                            # 计算2D距离（XY平面）
                            ee_pos_2d = ee_pos[env_idx, :2].unsqueeze(0)  # [1, 2]
                            dyn_obs_pos_2d = dyn_obs_positions[:, :2]  # [num_dyn_obs, 2]
                            dyn_obs_distance_2d = torch.norm(dyn_obs_pos_2d - ee_pos_2d, dim=1)  # [num_dyn_obs]
                            
                            # 选择最近的N个动态障碍物
                            num_closest = min(self.num_closest_dyn_obs, len(dyn_obs_distance_2d))
                            if num_closest > 0:
                                _, closest_idx = torch.topk(dyn_obs_distance_2d, num_closest, largest=False)
                                
                                # 确保索引有效
                                if closest_idx.max() < dyn_obs_positions.shape[0]:
                                    # 计算到最近N个障碍物的3D距离
                                    closest_dyn_obs_pos = dyn_obs_positions[closest_idx]  # [num_closest, 3]
                                    closest_dyn_obs_radii = dyn_obs_radii[closest_idx]  # [num_closest]
                                    
                                    # 计算到表面的距离
                                    dist_to_centers = torch.norm(ee_pos[env_idx].unsqueeze(0) - closest_dyn_obs_pos, dim=1)  # [num_closest]
                                    dist_to_surfaces = torch.clamp(dist_to_centers - closest_dyn_obs_radii, min=0.0)  # [num_closest]
                                    
                                    # 取最小值
                                    min_dyn_dist = dist_to_surfaces.min().item()
                                    min_dist = min(min_dist, min_dyn_dist)
            except Exception as e:
                # 如果动态障碍物检查失败，只使用静态障碍物结果
                pass
            
            distances[env_idx] = min_dist
        
        return distances
    
    def _compute_nearest_obstacle_from_voxel(self, ee_pos: torch.Tensor) -> float:
        """
        从体素地图推断到最近障碍物的距离
        
        方法：在体素地图中搜索占据的体素，计算EE到这些体素的距离
        """
        if self.current_voxel is None or self.voxel_grid_origin is None or self.voxel_res is None:
            return 10.0  # 没有体素地图，返回安全距离
        
        voxel = self.current_voxel  # 可能是 [1, 1, Vx, Vy, Vz], [1, Vx, Vy, Vz], 或 [Vx, Vy, Vz]
        # 移除所有 batch 和 channel 维度，只保留空间维度
        while voxel.dim() > 3:
            voxel = voxel.squeeze(0)  # 移除最外层维度
        
        Vx, Vy, Vz = voxel.shape  # 现在应该是 [Vx, Vy, Vz]
        
        # 找到所有占据的体素（occupancy > 0.5）
        occupied_mask = voxel > 0.5
        if not occupied_mask.any():
            return 10.0  # 没有探测到障碍物，返回安全距离
        
        # 获取占据体素的索引
        occupied_indices = torch.nonzero(occupied_mask, as_tuple=False)  # [N, 3]
        
        # 检查体素分辨率和网格原点是否有效
        if torch.isnan(self.voxel_res).any() or torch.isinf(self.voxel_res).any():
            return 10.0  # 无效数据，返回安全距离
        if torch.isnan(self.voxel_grid_origin).any() or torch.isinf(self.voxel_grid_origin).any():
            return 10.0  # 无效数据，返回安全距离
        
        # 将体素索引转换为世界坐标
        occupied_world = occupied_indices.float() * self.voxel_res.unsqueeze(0) + self.voxel_grid_origin.unsqueeze(0)
        
        # 检查世界坐标是否有效
        if torch.isnan(occupied_world).any() or torch.isinf(occupied_world).any():
            return 10.0  # 无效数据，返回安全距离
        
        # 计算EE到所有占据体素的距离
        ee_pos_expanded = ee_pos.unsqueeze(0)  # [1, 3]
        distances = torch.norm(occupied_world - ee_pos_expanded, dim=1)  # [N]
        
        # 检查距离是否有效
        if torch.isnan(distances).any() or torch.isinf(distances).any():
            return 10.0  # 无效数据，返回安全距离
        
        # 返回最小距离（实际应该减去体素半径，但简化处理）
        min_dist = distances.min().item()
        if not torch.isfinite(torch.tensor(min_dist)):
            return 10.0  # 无效数据，返回安全距离
        
        # 减去体素半径的一半（粗略估计）
        voxel_radius = torch.norm(self.voxel_res).item() * 0.5
        if not torch.isfinite(torch.tensor(voxel_radius)):
            voxel_radius = 0.1  # 默认体素半径
        
        result = max(0.0, min_dist - voxel_radius)
        return result if torch.isfinite(torch.tensor(result)) else 10.0

    def _compute_ray_directions(self):
        """预计算 LiDAR 射线方向（在末端执行器坐标系中）"""
        # 水平角度: 0 到 360 度
        h_angles = torch.linspace(0, 360 - self.lidar_hres, self.lidar_hbeams, device=self.device)
        h_angles_rad = torch.deg2rad(h_angles)
        
        # 垂直角度: vfov[0] 到 vfov[1]（度转弧度）
        v_angles_deg = torch.linspace(self.lidar_vfov[0], self.lidar_vfov[1], self.lidar_vbeams, device=self.device)
        v_angles_rad = torch.deg2rad(v_angles_deg)
        
        # 创建网格
        h_grid, v_grid = torch.meshgrid(h_angles_rad, v_angles_rad, indexing='ij')
        
        # 计算射线方向（球坐标转笛卡尔坐标）
        # X: 前方, Y: 左方, Z: 上方（末端执行器坐标系）
        x = torch.cos(v_grid) * torch.cos(h_grid)
        y = torch.cos(v_grid) * torch.sin(h_grid)
        z = torch.sin(v_grid)
        
        # [h_beams, v_beams, 3]
        self.ray_directions = torch.stack([x, y, z], dim=-1)
        
        print(f"[LiDAR] 射线方向已预计算: {self.ray_directions.shape}, h_beams={self.lidar_hbeams}, v_beams={self.lidar_vbeams}")
    
    def _update_lidar(self):
        """GPU 并行版 LiDAR，使用统一 actor 布局，避免越界。"""
        if not getattr(self, "use_lidar_input", True):
            return
        
        # 首次初始化射线方向
        if getattr(self, "ray_directions", None) is None:
            self._compute_ray_directions()
        
        # ====== 第6步：修正 LiDAR 的静态障碍索引生成 ======
        # 没有静态障碍就直接返回最大量程
        if self.num_obstacles == 0:
            self.lidar_scan = torch.full(
                (self.num_envs, 1, self.lidar_hbeams, self.lidar_vbeams),
                self.lidar_range, device=self.device, dtype=torch.float32
            )
            return
        
        # 1) 射线方向（已初始化）
        # 2) 生成静态障碍索引（与当前 actors_per_env 一致）
        actors_per_env = getattr(self, "actors_per_env", 2 + self.num_obstacles + getattr(self, "num_dynamic_obstacles", 0))
        obstacle_start_idx = 2                    # 静态从 2 开始
        total_static = self.num_envs * self.num_obstacles
        
        # root_states 是 CPU 大张量
        root_states_device = self.root_states.device if hasattr(self, "root_states") and self.root_states is not None else torch.device("cpu")
        env_ids = torch.arange(self.num_envs, device=root_states_device).unsqueeze(1).expand(-1, self.num_obstacles).reshape(-1)
        obs_ids = torch.arange(self.num_obstacles, device=root_states_device).unsqueeze(0).expand(self.num_envs, -1).reshape(-1)
        obstacle_indices = env_ids * actors_per_env + obstacle_start_idx + obs_ids  # CPU long
        obstacle_indices = obstacle_indices.to(dtype=torch.int32, device='cpu', non_blocking=True).contiguous()
        
        # 3) 读取静态障碍位置
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # 🎯 修复：不要重新 acquire，只 refresh（避免破坏绑定）
        if not hasattr(self, "root_states") or self.root_states is None:
            raise RuntimeError("[LiDAR] root_states not initialized! Call _reacquire_root_tensor() first.")
        
        all_obs_pos = self.root_states[obstacle_indices.long(), 0:3]         # CPU
        obstacle_positions = all_obs_pos.reshape(self.num_envs, self.num_obstacles, 3).to(self.device)
        
        # 4) 半径整理（与你当前的 self.obstacle_radii 组织方式匹配）
        if hasattr(self, "obstacle_radii") and self.obstacle_radii is not None:
            if len(self.obstacle_radii) == self.num_envs * self.num_obstacles:
                obstacle_radii = self.obstacle_radii.reshape(self.num_envs, self.num_obstacles).to(self.device)
            elif len(self.obstacle_radii) == self.num_obstacles:
                obstacle_radii = self.obstacle_radii.to(self.device).unsqueeze(0).expand(self.num_envs, -1)
            else:
                print(f"[Warning] obstacle_radii unexpected len={len(self.obstacle_radii)}, fallback to 0.1")
                obstacle_radii = torch.full((self.num_envs, self.num_obstacles), 0.1, device=self.device)
        else:
            obstacle_radii = torch.full((self.num_envs, self.num_obstacles), 0.1, device=self.device)
        
        # 末端位姿
        ee_pos = self.ee_pos   # [num_envs, 3]  (device)
        ee_quat = self.ee_quat # [num_envs, 4]  (device)
        
        # —— 批量旋转所有射线方向 —— 
        hB, vB = self.lidar_hbeams, self.lidar_vbeams
        num_rays = hB * vB
        ray_dirs_local = self.ray_directions.unsqueeze(0).expand(self.num_envs, -1, -1, -1)              # [N, hB, vB, 3]
        ray_dirs_flat  = ray_dirs_local.reshape(self.num_envs * num_rays, 3)                             # [N*num_rays, 3]
        ee_quat_flat   = ee_quat.unsqueeze(1).expand(-1, num_rays, -1).reshape(self.num_envs * num_rays, 4)
        ray_dirs_world_flat = quat_rotate_vector_batch(ee_quat_flat, ray_dirs_flat)                      # [N*num_rays, 3]
        ray_dirs_world = ray_dirs_world_flat.reshape(self.num_envs, hB, vB, 3)
        
        # 5) 后续：射线旋转、相交计算、最小距离聚合
        # —— 几何求交（射线-球）——
        ee_pos_2d = ee_pos.unsqueeze(1).unsqueeze(1)                   # [N,1,1,3]
        obs_pos_3d = obstacle_positions.unsqueeze(1).unsqueeze(1)      # [N,1,1,num_obstacles,3]
        ray_dirs_4d = ray_dirs_world.unsqueeze(3)                      # [N,hB,vB,1,3]
        oc = obs_pos_3d - ee_pos_2d.unsqueeze(3)                       # [N,hB,vB,num_obstacles,3]
        proj = (ray_dirs_4d * oc).sum(dim=-1)                          # [N,hB,vB,num_obstacles]
        oc2 = (oc ** 2).sum(dim=-1)                                    # [N,hB,vB,num_obstacles]
        r = obstacle_radii.unsqueeze(1).unsqueeze(1)                   # [N,1,1,num_obstacles]
        hit = (oc2 <= r**2) & (proj > 0)
        disc = (r**2 - oc2).clamp(min=0.0)
        hit_dist = torch.where(hit, proj - torch.sqrt(disc), torch.full_like(proj, self.lidar_range))
        hit_dist = hit_dist.clamp(min=0.0, max=self.lidar_range)
        min_dist = hit_dist.min(dim=-1)[0]                             # [N,hB,vB]
        
        # 存储（与上游保持一致的 range-minus 格式或直接距离，看你训练脚本）
        self.lidar_scan = (self.lidar_range - min_dist).unsqueeze(1)   # [N,1,hB,vB]
        self.lidar_scan = torch.clamp(self.lidar_scan, 0.0, self.lidar_range)

    def observe(self, voxel: Optional[torch.Tensor] = None, 
                grid_origin: Optional[torch.Tensor] = None,
                voxel_res: Optional[torch.Tensor] = None):
        """
        返回字典格式观测（参考无人机环境）：
        - state: [num_envs, 18] - 18维观测空间（保持原来的格式）
            1. 目标状态（4维）：检测状态、置信度、距离、水平方向
            2. 机械臂状态（9维）：关节角度(6维)、位置误差向量(3维)
            3. 感知信息（5维）：障碍物到5个连杆的最短距离
        - lidar: [num_envs, 1, h_beams, v_beams] - LiDAR扫描数据
        - dynamic_obstacle: [num_envs, 1, num_closest, 8] - 动态障碍物状态
        
        Args:
            voxel: 当前体素地图（已废弃，保留用于向后兼容）
            grid_origin: 体素网格原点（已废弃，保留用于向后兼容）
            voxel_res: 体素分辨率（已废弃，保留用于向后兼容）
        """
        # 更新LiDAR扫描数据（如果启用）
        if self.use_lidar_input:
            try:
                self._update_lidar()
            except Exception as e:
                print(f"[Warning] _update_lidar() failed: {e}")
                import traceback
                traceback.print_exc()
                # 使用默认值
                if self.lidar_scan is None:
                    self.lidar_scan = torch.zeros((self.num_envs, 1, self.lidar_hbeams, self.lidar_vbeams), device=self.device)
        
        # 批量处理所有环境的观测 [num_envs, 3]
        ee = self.ee_pos  # [num_envs, 3]
        tgt = self.target_pos  # [num_envs, 3]
        
        # 检查并修复 NaN/inf 值（批量处理）
        invalid_ee_mask = torch.isnan(ee).any(dim=1) | torch.isinf(ee).any(dim=1)
        if invalid_ee_mask.any():
            print(f"[Env] Warning: ee_pos contains NaN/inf in {invalid_ee_mask.sum().item()} environments, replacing with zeros")
            ee = torch.where(torch.isnan(ee) | torch.isinf(ee), torch.zeros_like(ee), ee)
        
        invalid_tgt_mask = torch.isnan(tgt).any(dim=1) | torch.isinf(tgt).any(dim=1)
        if invalid_tgt_mask.any():
            print(f"[Env] Warning: target_pos contains NaN/inf in {invalid_tgt_mask.sum().item()} environments, replacing with origin")
            tgt = torch.where(torch.isnan(tgt) | torch.isinf(tgt), torch.zeros_like(tgt), tgt)
        
        # 计算距离向量
        dist_vec = tgt - ee  # [num_envs, 3] 从末端指向目标
        goal_distance = torch.norm(dist_vec, dim=1, keepdim=True)  # [num_envs, 1]
        goal_distance = torch.where(torch.isnan(goal_distance) | torch.isinf(goal_distance),
                                   torch.tensor(1.0, device=self.device),
                                   goal_distance)
        
        # 1. 目标状态（4维）- 批量处理 [num_envs, 4]
        goal_detected = self.target_discovered.float().unsqueeze(1)  # [num_envs, 1]
        goal_confidence = torch.where(self.target_discovered.unsqueeze(1), 
                                      torch.tensor(0.9, device=self.device), 
                                      torch.tensor(0.0, device=self.device))  # [num_envs, 1]
        # goal_distance已在上面计算 [num_envs, 1]
        # 目标水平方向（简化：只用XY平面的角度）
        goal_direction_xy = torch.atan2(dist_vec[:, 1], dist_vec[:, 0]).unsqueeze(1)  # [num_envs, 1]
        
        # 2. 机械臂状态（9维）- 批量处理 [num_envs, 9]
        # 2.1 关节角度（6维）
        joint_positions = self.dof_states[:, 0].to(self.device).view(self.num_envs, self.dof_count)  # [num_envs, 6]
        # 2.2 位置误差向量（3维）：目标位置 - 末端位置
        position_error = dist_vec  # [num_envs, 3]
        
        # 3. 感知信息（5维）- 障碍物到5个连杆的最短距离
        try:
            dobs = self._compute_link_obstacle_distances_batch(use_voxel=False)  # [num_envs, 5]
            # 确保数值有效
            dobs = torch.where(torch.isfinite(dobs), dobs, torch.tensor(10.0, device=self.device))
            dobs = torch.clamp(dobs, min=0.0)  # [num_envs, 5]
        except Exception as e:
            print(f"[Warning] _compute_link_obstacle_distances_batch() failed: {e}")
            import traceback
            traceback.print_exc()
            dobs = torch.full((self.num_envs, 5), 1.0, device=self.device)  # 默认1米
        
        # 拼接state观测 [num_envs, 4+9+5=18]
        state_obs = torch.cat([
            goal_detected,           # [num_envs, 1]  - 检测状态
            goal_confidence,         # [num_envs, 1]  - 置信度
            goal_distance,           # [num_envs, 1]  - 目标距离
            goal_direction_xy,       # [num_envs, 1]  - 目标水平方向
            joint_positions,         # [num_envs, 6]  - 关节角度
            position_error,          # [num_envs, 3]  - 位置误差向量
            dobs                     # [num_envs, 5]  - 障碍物到连杆距离
        ], dim=1)  # 最终 [num_envs, 18]
        
        # 2. LiDAR观测（如果启用）
        if self.use_lidar_input and self.lidar_scan is not None:
            lidar_obs = self.lidar_scan  # [num_envs, 1, h_beams, v_beams]
        else:
            # 默认值
            lidar_obs = torch.zeros((self.num_envs, 1, self.lidar_hbeams, self.lidar_vbeams), device=self.device)
        
        # 🎯 已禁用动态障碍物：不再计算动态障碍物观测
        # 返回字典格式观测（只包含 state 和 lidar）
        obs = {
            "state": state_obs,           # [num_envs, 18] - 18维观测空间（包含ArUco检测状态）
            "lidar": lidar_obs,           # [num_envs, 1, h_beams, v_beams]
            # 🎯 已禁用动态障碍物：不再返回 dynamic_obstacle
        }
        
        return obs

    def step(self, action_xyz: torch.Tensor):
        """
        执行动作（关节角度增量控制，6维：每个关节的角度增量）
        
        Args:
            action_xyz: [N, 6] 关节角度增量 [dq1, dq2, dq3, dq4, dq5, dq6]（弧度）
                       - 坐标系：关节空间（Joint Space），不需要坐标系转换
                       - 关节顺序：shoulder_pan → shoulder_lift → elbow → wrist_1 → wrist_2 → wrist_3
                       - 参考 ur5e_DDPG_trajectory_planning_template: action_bound = [-0.0189, 0.0189]
        
        Returns:
            obs, reward, done, info
        """
        # 动作是关节角度增量，直接使用（关节空间，无需坐标系转换）
        action_xyz = action_xyz.to(self.device)
        
        # 🎯 确保动作形状正确 [num_envs, 6]
        if action_xyz.dim() == 1:
            action_xyz = action_xyz.unsqueeze(0)  # [6] -> [1, 6]
        
        # 确保batch size匹配
        assert action_xyz.shape[0] == self.num_envs, f"Action batch size {action_xyz.shape[0]} != num_envs {self.num_envs}"
        
        
        # 获取当前关节角度（Isaac Gym DOF顺序与URDF一致）
        current_joint_pos = self._get_joint_positions()  # [num_envs, 6]
        
        # 计算目标关节角度：当前角度 + 增量（关节空间直接累加）
        target_joint_pos = current_joint_pos + action_xyz  # [num_envs, 6]
        
        # 角度限幅：确保在关节限位范围内（使用URDF定义，批量处理）
        target_joint_pos[:, 0] = torch.clamp(target_joint_pos[:, 0], -2*math.pi, 2*math.pi)  # shoulder_pan
        target_joint_pos[:, 1] = torch.clamp(target_joint_pos[:, 1], -2*math.pi, 2*math.pi)  # shoulder_lift
        target_joint_pos[:, 2] = torch.clamp(target_joint_pos[:, 2], -math.pi, math.pi)  # elbow
        for j in range(3, self.dof_count):
            target_joint_pos[:, j] = torch.clamp(target_joint_pos[:, j], -2*math.pi, 2*math.pi)  # wrist_1-3
        
        # 🎯 关键：为每个环境设置位置目标（参考ur5e模板：直接设置state作为目标）
        # ur5e模板: data.ctrl[0] = -3500 * (data.qpos[0] - state[0]) - 100 * (data.qvel[0] - 0)
        # Isaac Gym: 设置位置目标，PD控制器自动计算力矩
        for i in range(self.num_envs):
            pos = target_joint_pos[i].detach().cpu().numpy()
            self.gym.set_actor_dof_position_targets(self.envs[i], self.robot_handles[i], pos)
        

        self.progress += 1
        
        # 🎯 关键修复：在物理模拟之前更新动态障碍物逻辑位置和位置到仿真
        # 这样可以在物理模拟之前就设置好位置，物理模拟会基于这个位置进行
        if self.num_dynamic_obstacles > 0 and len(self.dynamic_obstacle_handles) > 0:
            try:
                # 先更新动态障碍物的逻辑位置（在 dyn_obs_state 中）
                # 注意：这里需要 ee_pos，但 ee_pos 在 _read_wrist_pose() 中更新
                # 所以我们需要先读取一次（如果还没有读取）
                if not hasattr(self, 'ee_pos') or self.ee_pos is None:
                    self.gym.refresh_rigid_body_state_tensor(self.sim)
                    self._read_wrist_pose()
                self._move_dynamic_obstacles()
                
                # 🎯 关键修复：在物理模拟之前更新位置到仿真
                # 这样物理模拟会基于我们设置的位置进行，而不是 [0,0,0]
                self._update_dynamic_obstacle_positions_to_sim()
                # 🎯 标记动态障碍物已更新
                self._dynamic_obstacles_updated = True
            except Exception as e:
                print(f"[Warning] Failed to move/update dynamic obstacles before simulation: {e}")
        
        # 🎯 关键：执行仿真步骤，让PD控制器驱动机械臂移动到目标位置
        self._simulate_and_fetch()
        
        # 🎯 关键修复：在物理模拟之后，再次强制更新动态障碍物位置到仿真
        # 因为物理模拟可能会改变位置，我们需要再次强制更新
        if self.num_dynamic_obstacles > 0 and len(self.dynamic_obstacle_handles) > 0:
            try:
                # 🎯 先刷新状态，检查物理模拟后的位置
                self.gym.refresh_actor_root_state_tensor(self.sim)
                
                # 🎯 调试：检查物理模拟后的位置（前几步）
                debug_mode = hasattr(self, 'dyn_obs_step_count') and self.dyn_obs_step_count < 3
                if debug_mode:
                    self.actors_per_env = 2 + self.num_obstacles + self.num_dynamic_obstacles
                    first_obs_idx = 0 * self.actors_per_env + 2 + self.num_obstacles + 0
                    if first_obs_idx < self.root_states.shape[0]:
                        pos_after_sim = self.root_states[first_obs_idx, :3].cpu().numpy()
                        expected_pos = self.dyn_obs_state[0, :3].cpu().numpy()
                        print(f"[Debug] After simulation, root_states[{first_obs_idx}] = [{pos_after_sim[0]:.3f}, {pos_after_sim[1]:.3f}, {pos_after_sim[2]:.3f}], expected=[{expected_pos[0]:.3f}, {expected_pos[1]:.3f}, {expected_pos[2]:.3f}]")
                
                # 强制更新位置到仿真（覆盖物理模拟的结果）
                self._update_dynamic_obstacle_positions_to_sim()
                
                # 🎯 再次刷新，验证更新后的位置
                self.gym.refresh_actor_root_state_tensor(self.sim)
                if debug_mode:
                    if first_obs_idx < self.root_states.shape[0]:
                        pos_after_update = self.root_states[first_obs_idx, :3].cpu().numpy()
                        print(f"[Debug] After update (post-sim), root_states[{first_obs_idx}] = [{pos_after_update[0]:.3f}, {pos_after_update[1]:.3f}, {pos_after_update[2]:.3f}], expected=[{expected_pos[0]:.3f}, {expected_pos[1]:.3f}, {expected_pos[2]:.3f}]")
            except Exception as e:
                print(f"[Warning] Failed to update dynamic obstacles after simulation: {e}")
        
        # 更新末端位姿（在执行动作后）
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._read_wrist_pose()
        
        # 批量处理NaN检查（已在_read_wrist_pose中处理，这里不需要单独处理）
        
        # 更新障碍物位置（障碍物可能移动，虽然当前是静态的）
        self._update_obstacle_positions()
        
        # 更新LiDAR扫描数据（如果启用）
        if self.use_lidar_input:
            try:
                self._update_lidar()
            except Exception as e:
                print(f"[Warning] _update_lidar() failed in step(): {e}")
        
        # 🎯 重要：在更新动态障碍物后，等待一小段时间让Isaac Gym稳定
        # 这可以避免段错误（通过执行一次小的仿真步骤来稳定状态）
        # 更新 ArUco 发现逻辑（批量处理）
        # 🎯 重新启用：已修复camera_pose()中的问题（移除flatten()，使用更安全的标量提取）
        was_discovered = self.target_discovered.clone()
        try:
            # 在调用前检查必要的状态
            if (hasattr(self, 'ee_pos') and self.ee_pos is not None and 
                hasattr(self, 'ee_quat') and self.ee_quat is not None and
                hasattr(self, 'target_pos') and self.target_pos is not None):
                self._update_aruco_detection()
            else:
                # 如果状态未准备好，跳过检测
                pass
        except Exception as e:
            print(f"[Warning] _update_aruco_detection() failed in step(): {e}")
            import traceback
            traceback.print_exc()
            # 如果检测失败，保持当前状态不变
        
        # 更新距上次检测的步数（批量处理）
        newly_discovered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        newly_lost = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 批量更新检测步数
        try:
            self.steps_since_last_detection[~self.target_discovered] = (
                self.progress[~self.target_discovered] - self.last_detection_step[~self.target_discovered]
            )
            self.steps_since_last_detection[self.target_discovered] = 0
        except Exception as e:
            print(f"[Error] Failed to update steps_since_last_detection: {e}")
            import traceback
            traceback.print_exc()
        
        # 更新last_detection_step
        try:
            self.last_detection_step[newly_discovered] = self.progress[newly_discovered]
            self.last_detection_step[newly_lost] = self.progress[newly_lost]
        except Exception as e:
            print(f"[Error] Failed to update last_detection_step: {e}")
            import traceback
            traceback.print_exc()
        
        # 多角度观察状态更新（每个环境独立）
        try:
            # 🎯 修复：将布尔张量转换为Python列表，避免在循环中访问张量元素导致段错误
            newly_discovered_list = newly_discovered.cpu().tolist() if isinstance(newly_discovered, torch.Tensor) else newly_discovered
            newly_lost_list = newly_lost.cpu().tolist() if isinstance(newly_lost, torch.Tensor) else newly_lost
            target_discovered_list = self.target_discovered.cpu().tolist() if isinstance(self.target_discovered, torch.Tensor) else self.target_discovered
            
            for i in range(self.num_envs):
                if newly_discovered_list[i]:
                    # 刚发现，重置多角度观察状态
                    self.observation_angles[i] = []
                    self.multiview_steps[i] = 0
                elif newly_lost_list[i]:
                    # 刚刚丢失目标，重置多角度观察状态
                    self.observation_angles[i] = []
                    self.multiview_steps[i] = 0
                elif target_discovered_list[i]:
                    # 继续观察，增加步数
                    self.multiview_steps[i] += 1
        except Exception as e:
            print(f"[Error] Failed in multiview observation update loop: {e}")
            import traceback
            traceback.print_exc()

        # 伪基础奖励：负距离（批量计算）
        try:
            base_reward = -torch.norm(self.target_pos - self.ee_pos, dim=1)  # [num_envs]
        except Exception as e:
            print(f"[Error] Failed to compute base_reward: {e}")
            import traceback
            traceback.print_exc()
            base_reward = torch.zeros(self.num_envs, device=self.device)
        
        # 检查成功终止条件：发现目标且存在安全路径（批量处理）
        try:
            success, success_reason = self._check_success_conditions()
        except Exception as e:
            print(f"[Warning] _check_success_conditions() failed: {e}")
            import traceback
            traceback.print_exc()
            # 如果失败，返回默认值（不成功）
            success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            success_reason = ["检查失败"] * self.num_envs
        
        try:
            timeout = (self.progress >= self.max_steps)  # [num_envs]
            
            # 检查渐进式安全终止条件（在compute_rewards中设置）
            collision_terminate = self._should_terminate  # [num_envs]
            
            done = timeout | success | collision_terminate  # [num_envs]
        except Exception as e:
            print(f"[Error] Failed to compute timeout/done: {e}")
            import traceback
            traceback.print_exc()
            timeout = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        # info字典（每个环境的信息）
        try:
            # 🎯 修复：将布尔张量转换为列表，避免在列表推导式中直接索引张量导致段错误
            collision_terminate_list = collision_terminate.cpu().tolist() if isinstance(collision_terminate, torch.Tensor) else collision_terminate
            success_list = success.cpu().tolist() if isinstance(success, torch.Tensor) else success
            
            # 确保 success_reason 是列表
            if not isinstance(success_reason, list):
                success_reason = [success_reason] * self.num_envs if success_reason is not None else [None] * self.num_envs
            
            # 构建 termination_reason（使用 Python 列表而不是张量）
            termination_reason = []
            for i in range(self.num_envs):
                if collision_terminate_list[i]:
                    term_reason = self._termination_reason[i] if i < len(self._termination_reason) else "碰撞终止"
                elif success_list[i]:
                    term_reason = success_reason[i] if i < len(success_reason) else "成功"
                else:
                    term_reason = None
                termination_reason.append(term_reason)
            
            info = {
                'success': success,  # [num_envs]
                'timeout': timeout,  # [num_envs]
                'collision_terminate': collision_terminate,  # [num_envs]
                'success_reason': success_reason,  # list of strings or None
                'termination_reason': termination_reason  # list of strings or None
            }
        except Exception as e:
            print(f"[Error] Failed to build info dictionary: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认值
            info = {
                'success': success if isinstance(success, torch.Tensor) else torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
                'timeout': timeout if isinstance(timeout, torch.Tensor) else torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
                'collision_terminate': collision_terminate if isinstance(collision_terminate, torch.Tensor) else torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
                'success_reason': success_reason if isinstance(success_reason, list) else [None] * self.num_envs,
                'termination_reason': [None] * self.num_envs
            }
        
        # 获取观测（在step的最后，确保状态已更新）
        try:
            obs = self.observe()  # [num_envs, obs_dim]
        except Exception as e:
            print(f"[Error] Failed to observe(): {e}")
            import traceback
            traceback.print_exc()
            # 返回默认观测
            obs = torch.zeros((self.num_envs, 16), device=self.device, dtype=torch.float32)
        
        # 🎯 修复：确保所有返回值都是安全的，在返回前进行最终验证
        try:
            # 确保 base_reward 已初始化（使用 try-except 而不是 locals()）
            try:
                _ = base_reward.shape  # 尝试访问，如果不存在会抛出 NameError
            except (NameError, AttributeError):
                print(f"[Warning] base_reward not initialized or invalid, using default")
                base_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            
            # 确保所有张量都在正确的设备上并且是连续的
            obs = obs.to(device=self.device).contiguous() if isinstance(obs, torch.Tensor) else obs
            base_reward = base_reward.to(device=self.device).contiguous() if isinstance(base_reward, torch.Tensor) else base_reward
            done = done.to(device=self.device).contiguous() if isinstance(done, torch.Tensor) else done
            
            # 验证形状
            if isinstance(obs, torch.Tensor) and obs.shape[0] != self.num_envs:
                print(f"[Warning] obs shape mismatch: {obs.shape[0]} != {self.num_envs}")
                obs = obs[:self.num_envs] if obs.shape[0] > self.num_envs else torch.cat([obs, torch.zeros((self.num_envs - obs.shape[0], obs.shape[1]), device=self.device, dtype=obs.dtype)])
            
            if isinstance(base_reward, torch.Tensor) and base_reward.shape[0] != self.num_envs:
                print(f"[Warning] base_reward shape mismatch: {base_reward.shape[0]} != {self.num_envs}")
                base_reward = base_reward[:self.num_envs] if base_reward.shape[0] > self.num_envs else torch.cat([base_reward, torch.zeros((self.num_envs - base_reward.shape[0],), device=self.device, dtype=base_reward.dtype)])
            
            if isinstance(done, torch.Tensor) and done.shape[0] != self.num_envs:
                print(f"[Warning] done shape mismatch: {done.shape[0]} != {self.num_envs}")
                done = done[:self.num_envs] if done.shape[0] > self.num_envs else torch.cat([done, torch.zeros((self.num_envs - done.shape[0],), device=self.device, dtype=done.dtype)])
            
        except Exception as e:
            print(f"[Error] Failed in final validation before return: {e}")
            import traceback
            traceback.print_exc()
            # 确保即使出错也能返回有效值
            try:
                _ = base_reward.shape
            except (NameError, AttributeError):
                base_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            if not isinstance(obs, torch.Tensor):
                obs = torch.zeros((self.num_envs, 16), device=self.device, dtype=torch.float32)
            if not isinstance(done, torch.Tensor):
                done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        
        return obs, base_reward, done, info

    def camera_pose(self, env_idx: int = 0) -> torch.Tensor:
        """
        获取相机在世界坐标系中的位姿（支持多环境）
        正确应用：wrist位姿 × 相机局部变换
        
        Args:
            env_idx: 环境索引，默认为0（第一个环境）
        
        Returns:
            T_camera: [4, 4] 相机变换矩阵（世界坐标系）
        """
        try:
            # 基础安全检查
            if not hasattr(self, 'ee_pos') or self.ee_pos is None:
                raise ValueError("ee_pos not initialized")
            if not hasattr(self, 'ee_quat') or self.ee_quat is None:
                raise ValueError("ee_quat not initialized")
            
            # 环境索引安全检查
            if env_idx < 0 or env_idx >= self.num_envs:
                env_idx = 0
            if env_idx >= self.ee_pos.shape[0] or env_idx >= self.ee_quat.shape[0]:
                env_idx = 0
            
            # 🎯 修复：直接使用张量操作，避免.item()转换
            wrist_pos = self.ee_pos[env_idx].to(device=self.device, dtype=torch.float32)  # [3]
            wrist_quat = self.ee_quat[env_idx].to(device=self.device, dtype=torch.float32)  # [4]
            
            # 验证数据有效性
            if wrist_pos.numel() != 3 or wrist_quat.numel() != 4:
                raise ValueError(f"Invalid tensor size: pos={wrist_pos.numel()}, quat={wrist_quat.numel()}")
            
            # 检查NaN/Inf，使用张量操作
            if torch.isnan(wrist_quat).any() or torch.isinf(wrist_quat).any():
                wrist_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
            
            # 归一化四元数（保持张量）
            q_norm = wrist_quat / torch.norm(wrist_quat)
            
            # 🎯 修复：使用更简洁的张量操作计算旋转矩阵
            x, y, z, w = q_norm[0], q_norm[1], q_norm[2], q_norm[3]
            
            # 直接创建旋转矩阵，避免复杂的stack操作
            R_wrist = torch.zeros(3, 3, device=self.device, dtype=torch.float32)
            R_wrist[0, 0] = 1 - 2*(y*y + z*z)
            R_wrist[0, 1] = 2*(x*y - w*z)
            R_wrist[0, 2] = 2*(x*z + w*y)
            R_wrist[1, 0] = 2*(x*y + w*z)
            R_wrist[1, 1] = 1 - 2*(x*x + z*z)
            R_wrist[1, 2] = 2*(y*z - w*x)
            R_wrist[2, 0] = 2*(x*z - w*y)
            R_wrist[2, 1] = 2*(y*z + w*x)
            R_wrist[2, 2] = 1 - 2*(x*x + y*y)
            
            # 构建wrist变换矩阵
            T_wrist = torch.eye(4, device=self.device, dtype=torch.float32)
            T_wrist[:3, :3] = R_wrist
            T_wrist[:3, 3] = wrist_pos
            
            # 相机局部变换
            def get_camera_local_transform():
                """安全获取相机局部变换（使用张量操作）"""
                T_local = torch.eye(4, device=self.device, dtype=torch.float32)
                
                try:
                    # 处理旋转
                    if hasattr(self, 'camera_mount_quat') and self.camera_mount_quat is not None:
                        mount_quat = self.camera_mount_quat
                        if isinstance(mount_quat, torch.Tensor):
                            mount_quat = mount_quat.to(device=self.device, dtype=torch.float32)
                            if mount_quat.numel() == 4:
                                # 归一化
                                q_mount_norm = mount_quat / torch.norm(mount_quat)
                                qx, qy, qz, qw = q_mount_norm[0], q_mount_norm[1], q_mount_norm[2], q_mount_norm[3]
                                
                                # 使用直接赋值创建旋转矩阵
                                R_local = torch.zeros(3, 3, device=self.device, dtype=torch.float32)
                                R_local[0, 0] = 1 - 2*(qy*qy + qz*qz)
                                R_local[0, 1] = 2*(qx*qy - qw*qz)
                                R_local[0, 2] = 2*(qx*qz + qw*qy)
                                R_local[1, 0] = 2*(qx*qy + qw*qz)
                                R_local[1, 1] = 1 - 2*(qx*qx + qz*qz)
                                R_local[1, 2] = 2*(qy*qz - qw*qx)
                                R_local[2, 0] = 2*(qx*qz - qw*qy)
                                R_local[2, 1] = 2*(qy*qz + qw*qx)
                                R_local[2, 2] = 1 - 2*(qx*qx + qy*qy)
                                
                                T_local[:3, :3] = R_local
                    elif hasattr(self, 'camera_mount_rpy') and self.camera_mount_rpy is not None:
                        mount_rpy = self.camera_mount_rpy
                        if isinstance(mount_rpy, torch.Tensor):
                            mount_rpy = mount_rpy.to(device=self.device, dtype=torch.float32)
                        else:
                            mount_rpy = torch.tensor(mount_rpy, device=self.device, dtype=torch.float32)
                        
                        if mount_rpy.numel() == 3:
                            rx, ry, rz = mount_rpy[0], mount_rpy[1], mount_rpy[2]
                            
                            # 使用张量三角函数
                            cx, sx = torch.cos(rx), torch.sin(rx)
                            cy, sy = torch.cos(ry), torch.sin(ry)
                            cz, sz = torch.cos(rz), torch.sin(rz)
                            
                            # 直接构建旋转矩阵（使用直接赋值避免张量列表问题）
                            Rz = torch.zeros(3, 3, device=self.device, dtype=torch.float32)
                            Rz[0, 0] = cz
                            Rz[0, 1] = -sz
                            Rz[0, 2] = 0.0
                            Rz[1, 0] = sz
                            Rz[1, 1] = cz
                            Rz[1, 2] = 0.0
                            Rz[2, 0] = 0.0
                            Rz[2, 1] = 0.0
                            Rz[2, 2] = 1.0
                            
                            Ry = torch.zeros(3, 3, device=self.device, dtype=torch.float32)
                            Ry[0, 0] = cy
                            Ry[0, 1] = 0.0
                            Ry[0, 2] = sy
                            Ry[1, 0] = 0.0
                            Ry[1, 1] = 1.0
                            Ry[1, 2] = 0.0
                            Ry[2, 0] = -sy
                            Ry[2, 1] = 0.0
                            Ry[2, 2] = cy
                            
                            Rx = torch.zeros(3, 3, device=self.device, dtype=torch.float32)
                            Rx[0, 0] = 1.0
                            Rx[0, 1] = 0.0
                            Rx[0, 2] = 0.0
                            Rx[1, 0] = 0.0
                            Rx[1, 1] = cx
                            Rx[1, 2] = -sx
                            Rx[2, 0] = 0.0
                            Rx[2, 1] = sx
                            Rx[2, 2] = cx
                            
                            T_local[:3, :3] = Rz @ Ry @ Rx
                    
                    # 处理位移
                    if hasattr(self, 'camera_mount_offset') and self.camera_mount_offset is not None:
                        offset = self.camera_mount_offset
                        if isinstance(offset, torch.Tensor):
                            offset = offset.to(device=self.device, dtype=torch.float32)
                            if offset.numel() == 3:
                                T_local[:3, 3] = offset
                    
                except Exception as e:
                    print(f"[Warning] Camera local transform configuration error: {e}")
                    # 保持单位矩阵
                
                return T_local
            
            T_local = get_camera_local_transform()
            
            # 计算最终相机位姿
            T_camera = T_wrist @ T_local
            
            # 🎯 最终检查：确保结果有效
            if torch.isnan(T_camera).any() or torch.isinf(T_camera).any():
                print(f"[Warning] Invalid camera pose detected for env {env_idx}, using default")
                T_camera = torch.eye(4, device=self.device, dtype=torch.float32)
                T_camera[:3, 3] = torch.tensor([0.5, 0.0, 0.5], device=self.device, dtype=torch.float32)
            
            return T_camera
                
        except Exception as e:
            print(f"[Error] camera_pose() failed for env {env_idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回安全的默认位姿
            T_camera = torch.eye(4, device=self.device, dtype=torch.float32)
            T_camera[:3, 3] = torch.tensor([0.5, 0.0, 0.5], device=self.device, dtype=torch.float32)
            return T_camera

    def render_depth(self, env_idx: int = 0) -> torch.Tensor:
        """
        获取 Isaac Gym 渲染的深度（米）
        
        注意：此函数不执行仿真步骤，只渲染当前状态的相机图像
        仿真步骤应该在step()函数中统一执行
        
        Args:
            env_idx: 环境索引，默认为0（第一个环境）
        """
        # 🎯 关键修复：当启用viewer时，不创建相机传感器，直接返回空深度图
        if self.enable_viewer:
            # 当启用viewer时，相机传感器未创建，返回空深度图
            return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
        
        # 🎯 修复：检查相机句柄是否有效，如果无效应该抛出错误
        if self.camera_handle is None or self.camera_handle == -1:
            raise RuntimeError(f"Camera handle is invalid ({self.camera_handle}). Cannot render depth. Camera sensors must be properly initialized.")
        
        # 🎯 修复：使用正确的环境对象（env_idx对应的环境）
        if env_idx < 0 or env_idx >= self.num_envs:
            env_idx = 0
        
        env = self.envs[env_idx] if hasattr(self, 'envs') and self.envs is not None and env_idx < len(self.envs) else self.env
        camera_handle = self.camera_handles[env_idx] if hasattr(self, 'camera_handles') and self.camera_handles is not None and env_idx < len(self.camera_handles) else self.camera_handle
        
        # 🎯 安全检查：验证 sim 和 env 对象是否有效
        if self.sim is None:
            print(f"[Error] sim is None, cannot render depth")
            return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
        
        if env is None:
            print(f"[Error] env is None, cannot render depth")
            return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
        
        if camera_handle is None or camera_handle == -1:
            print(f"[Error] camera_handle is invalid ({camera_handle}), cannot render depth")
            return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
        
        # 🎯 重要：必须在获取相机图像前渲染所有相机传感器
        # 注意：根据Isaac Gym文档，render_all_camera_sensors()需要在step_graphics()之后调用
        # 当启用viewer时，step_graphics()必须在render_all_camera_sensors()之前调用
        # 🎯 关键修复：防止竞争条件和GPU内存泄漏 - 添加完善的错误处理和资源管理
        depth_img = None
        try:
            # 🎯 关键修复：在函数开始处添加更严格的GPU内存管理
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 确保所有之前的GPU操作完成
                torch.cuda.empty_cache()  # 清理GPU缓存，防止内存泄漏
            
            # 🎯 关键修复：检查动态障碍物是否刚更新，如果是，需要确保GPU状态同步
            if hasattr(self, '_dynamic_obstacles_updated') and self._dynamic_obstacles_updated:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # 再次同步，确保动态障碍物更新完成
                # 重置标志
                self._dynamic_obstacles_updated = False
            
            # 🎯 关键修复：验证图形上下文有效性
            if self.sim is None:
                print(f"[Error] sim is None, cannot render depth")
                return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
            
            # 渲染所有相机传感器（必须在step_graphics()之后）
            # 🎯 注意：当启用viewer时，训练循环必须在render_depth()之前调用step_graphics()
            # 🎯 关键修复：添加更完善的错误处理和资源管理，防止GPU内存泄漏
            try:
                # 🎯 关键修复：在调用render_all_camera_sensors()之前，确保图形状态已准备
                if self.enable_viewer and self.viewer is not None:
                    # 验证viewer仍然有效
                    try:
                        # 尝试获取viewer状态（如果viewer无效，这可能会失败）
                        pass  # 暂时不检查，因为Isaac Gym没有提供直接检查viewer有效性的API
                    except:
                        print(f"[Warning] Viewer may be invalid, attempting to render anyway")
                
                # 🎯 关键修复：参考manipulator_env_gym.py的实现方式
                # manipulator_env_gym.py不使用render_all_camera_sensors()，因为它使用LiDAR而非深度相机
                # 但我们的代码需要深度图像，所以必须调用render_all_camera_sensors()
                # 关键：manipulator_env_gym.py在step()中调用step_graphics()，然后draw_viewer()
                # 我们的训练循环已经在render_depth()之前调用了step_graphics()
                # 所以这里应该可以安全地调用render_all_camera_sensors()
                
                # 🎯 关键修复：当启用viewer时，确保在调用render_all_camera_sensors()之前，step_graphics()已经完成
                # 参考manipulator_env_gym.py：step_graphics()在step()中调用，在draw_viewer()之前
                # 我们的训练循环已经在render_depth()之前调用了step_graphics()，所以这里应该安全
                if self.enable_viewer and self.viewer is not None:
                    # 再次确保GPU同步（虽然训练循环已经调用了step_graphics()）
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    # 🎯 关键修复：验证step_graphics()是否已经调用（通过检查图形状态）
                    # 注意：训练循环应该在render_depth()之前调用step_graphics()
                    # 这里我们假设step_graphics()已经调用，直接调用render_all_camera_sensors()
                
                # 🎯 关键修复：完全参考manipulator_env_gym.py的实现方式
                # manipulator_env_gym.py不使用render_all_camera_sensors()，因为它使用LiDAR而非深度相机
                # 但我们的代码需要深度图像，所以必须调用render_all_camera_sensors()
                # 🚨 关键发现：当启用viewer时，render_all_camera_sensors()可能与viewer的图形上下文冲突导致段错误
                # 解决方案：当启用viewer时，尝试使用更安全的方式获取深度图像
                # 或者：在启用viewer时，跳过render_all_camera_sensors()，直接尝试获取图像（可能为空，但不会段错误）
                
                # 🎯 关键修复：当启用viewer时，尝试跳过render_all_camera_sensors()以避免图形上下文冲突
                # 参考manipulator_env_gym.py：它不使用render_all_camera_sensors()，所以我们也尝试跳过
                # 但这样可能导致图像为空，所以需要检查并处理
                if self.enable_viewer and self.viewer is not None:
                    # 🚨 关键修复：当启用viewer时，不调用render_all_camera_sensors()，避免图形上下文冲突
                    # 这可能导致深度图像为空，但可以避免段错误
                    # 注意：step_graphics()已经在训练循环中调用，所以图形状态应该已更新
                    print(f"[Warning] Skipping render_all_camera_sensors() when viewer is enabled to avoid segfault")
                    # 不调用render_all_camera_sensors()，直接尝试获取图像
                    # 如果图像为空，将返回空深度图
                else:
                    # 不启用viewer时，正常调用render_all_camera_sensors()
                    try:
                        # 🎯 关键修复：在调用render_all_camera_sensors()之前，再次确保GPU同步
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        self.gym.render_all_camera_sensors(self.sim)
                        
                        # 🎯 关键修复：渲染后立即同步GPU，确保渲染完成
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except RuntimeError as e:
                        # 如果render_all_camera_sensors()失败，可能是图形上下文问题
                        print(f"[Error] render_all_camera_sensors() failed with RuntimeError: {e}")
                        print(f"[Error] This may indicate a graphics context conflict")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # 返回空深度图，避免段错误
                        return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
                    
            except RuntimeError as e:
                print(f"[Error] RuntimeError in render_all_camera_sensors(): {e}")
                import traceback
                traceback.print_exc()
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 返回空深度图，避免段错误
                return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
            except Exception as e:
                print(f"[Error] Exception in render_all_camera_sensors(): {e}")
                import traceback
                traceback.print_exc()
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 返回空深度图，避免段错误
                return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
            
            # 🎯 关键修复：获取相机图像，添加完善的错误处理
            try:
                # 验证相机句柄有效性
                if camera_handle is None or camera_handle == -1:
                    print(f"[Error] Invalid camera_handle: {camera_handle}")
                    return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
                
                # 🎯 关键修复：在获取图像前再次同步GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # 获取深度图像
                depth_img = self.gym.get_camera_image(self.sim, env, camera_handle, gymapi.IMAGE_DEPTH)
                
                # 🎯 关键修复：立即同步GPU，确保图像数据已传输
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
            except RuntimeError as e:
                print(f"[Error] RuntimeError in get_camera_image(): {e}")
                import traceback
                traceback.print_exc()
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 返回空深度图，避免段错误
                return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
            except Exception as e:
                print(f"[Error] Exception in get_camera_image(): {e}")
                import traceback
                traceback.print_exc()
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 返回空深度图，避免段错误
                return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
            
            # 🎯 关键修复：验证图像数据有效性
            if depth_img is None:
                print(f"[Warning] Camera image is None, returning empty depth image")
                return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
            
            if not hasattr(depth_img, 'size') or depth_img.size == 0:
                print(f"[Warning] Camera image is empty (size={depth_img.size if hasattr(depth_img, 'size') else 'N/A'}), returning empty depth image")
                return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
            
            # 🎯 关键修复：安全地复制图像数据，避免内存泄漏
            try:
                depth = torch.from_numpy(depth_img.copy()).to(self.device)
            except Exception as e:
                print(f"[Error] Failed to convert depth image to tensor: {e}")
                import traceback
                traceback.print_exc()
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
            
            # 🎯 关键修复：确保深度图形状正确
            if depth.shape != (self.img_h, self.img_w):
                print(f"[Warning] Depth image shape mismatch: expected ({self.img_h}, {self.img_w}), got {depth.shape}")
                # 清理无效的tensor
                del depth
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
            
            # 🎯 关键修复：将非有限值替换为一个较大的深度（裁剪由体素构建时 max_range 控制）
            depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, 5.0))
            
            # 🎯 关键修复：最终GPU同步，确保所有操作完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            return depth
            
        except RuntimeError as e:
            print(f"[Error] RuntimeError in render_depth(): {e}")
            import traceback
            traceback.print_exc()
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 返回空的深度图
            return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
        except Exception as e:
            print(f"[Error] Exception in render_depth(): {e}")
            import traceback
            traceback.print_exc()
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 返回空的深度图
            return torch.zeros((self.img_h, self.img_w), device=self.device, dtype=torch.float32)
        finally:
            # 🎯 关键修复：确保资源清理（即使发生异常）
            if depth_img is not None:
                # Python的垃圾回收会自动处理numpy数组，但我们可以显式清理
                del depth_img
            # 定期清理GPU缓存，防止内存泄漏
            if torch.cuda.is_available() and hasattr(self, '_render_depth_call_count'):
                self._render_depth_call_count = getattr(self, '_render_depth_call_count', 0) + 1
                # 每100次调用清理一次缓存
                if self._render_depth_call_count % 100 == 0:
                    torch.cuda.empty_cache()
            else:
                self._render_depth_call_count = 1

    # ================== 奖励计算（安全性优先的探索奖励） ==================
    def compute_rewards(self, action_xyz: torch.Tensor, prev_angle_deg: Optional[torch.Tensor], cfg: dict, voxel_data: Optional[torch.Tensor] = None):
        """
        端到端两阶段奖励函数（探索+到达）
        
        阶段1（未发现）：探索奖励 + 障碍物惩罚
        阶段2（已发现）：多角度观察 + 相机朝向变化 + 障碍物惩罚
        阶段3（发现后）：靠近奖励（端到端到达目标奖励：位置误差惩罚 + 零误差鼓励）+ 障碍物惩罚
        
        Args:
            action_xyz: [N, 6] 动作（关节角度增量）
            prev_angle_deg: 上一步的角度（用于shaping，保留兼容性）
            cfg: 配置字典
            voxel_data: 保留兼容性，暂不使用
        
        Returns:
            reward: [N] 总奖励
            angle_h: [N] 当前水平角度（度）
        """
        N = self.num_envs
        total_reward = torch.zeros(N, device=self.device)
        
        # 1. 障碍物惩罚（所有阶段，最高优先级）
        try:
            obstacle_penalty = self._compute_obstacle_penalty_batch(cfg)  # [N]
            if obstacle_penalty.shape[0] == N:
                total_reward += obstacle_penalty
            else:
                print(f"[Warning] obstacle_penalty shape mismatch: {obstacle_penalty.shape[0]} != {N}")
                if obstacle_penalty.shape[0] > N:
                    total_reward += obstacle_penalty[:N]
                else:
                    total_reward[:obstacle_penalty.shape[0]] += obstacle_penalty
        except Exception as e:
            print(f"[Warning] Failed to compute obstacle penalty: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. 发现奖励（一次性）
        try:
            discovery_reward = self._compute_discovery_rewards(cfg)
            if discovery_reward.shape[0] == N:
                total_reward += discovery_reward
            elif discovery_reward.shape[0] > N:
                total_reward += discovery_reward[:N]
            else:
                total_reward[:discovery_reward.shape[0]] += discovery_reward
        except Exception as e:
            print(f"[Warning] Failed to compute discovery rewards: {e}")
        
        # 3. 根据发现状态切换奖励
        discovered_mask = self.target_discovered  # [N]
        unknown_mask = ~discovered_mask          # [N]
        
        # 3.1 未发现阶段：探索奖励
        if unknown_mask.any():
            try:
                exploration_reward = self._compute_simplified_exploration_rewards(action_xyz, cfg)  # [N]
                if exploration_reward.shape[0] == N:
                    total_reward += exploration_reward
                else:
                    if exploration_reward.shape[0] > N:
                        total_reward += exploration_reward[:N]
                    else:
                        total_reward[:exploration_reward.shape[0]] += exploration_reward
            except Exception as e:
                print(f"[Warning] Failed to compute exploration rewards: {e}")
        
        # 3.2 已发现阶段：观察奖励 + 靠近奖励
        if discovered_mask.any():
            try:
                # 3.2.1 观察奖励（多角度观察 + 相机朝向变化）
                observation_reward = self._compute_simplified_observation_rewards(action_xyz, cfg)  # [N]
                if observation_reward.shape[0] == N:
                    total_reward += observation_reward
                else:
                    if observation_reward.shape[0] > N:
                        total_reward += observation_reward[:N]
                    else:
                        total_reward[:observation_reward.shape[0]] += observation_reward
            except Exception as e:
                print(f"[Warning] Failed to compute observation rewards: {e}")
            
            # 3.2.2 靠近奖励（发现ArUco后的端到端到达目标奖励）
            try:
                approaching_reward = self._compute_approaching_reward(cfg)  # [N]
                if approaching_reward.shape[0] == N:
                    total_reward += approaching_reward
                else:
                    if approaching_reward.shape[0] > N:
                        total_reward += approaching_reward[:N]
                    else:
                        total_reward[:approaching_reward.shape[0]] += approaching_reward
            except Exception as e:
                print(f"[Warning] Failed to compute approaching rewards: {e}")
        
        # 4. 成功奖励（一次性，当到达目标时）
        try:
            success_reward = self._compute_success_reward(cfg)  # [N]
            if success_reward.shape[0] == N:
                total_reward += success_reward
            else:
                if success_reward.shape[0] > N:
                    total_reward += success_reward[:N]
                else:
                    total_reward[:success_reward.shape[0]] += success_reward
        except Exception as e:
            print(f"[Warning] Failed to compute success rewards: {e}")
        
        # 计算当前角度（用于角度塑形）
        try:
            rel = (self.target_pos - self.ee_pos)
            if rel.shape[0] != N:
                rel = rel[:N] if rel.shape[0] > N else torch.cat([rel, torch.zeros(N - rel.shape[0], 3, device=self.device)])
            
            angle_h = torch.rad2deg(torch.atan2(rel[:, 1].abs(), rel[:, 0].abs() + 1e-6))
            if angle_h.shape[0] != N:
                angle_h = angle_h[:N] if angle_h.shape[0] > N else torch.zeros(N, device=self.device)
            
            if total_reward.shape[0] != N:
                total_reward = total_reward[:N] if total_reward.shape[0] > N else torch.zeros(N, device=self.device)
            
            total_reward = total_reward.to(self.device)
            angle_h = angle_h.to(self.device)
            
        except Exception as e:
            print(f"[Warning] Error computing angle_h: {e}")
            angle_h = torch.zeros(N, device=self.device)
            if total_reward.shape[0] != N:
                total_reward = torch.zeros(N, device=self.device)
        
        # 更新 prev_ee_quat（用于旋转计算）
        try:
            if hasattr(self, 'ee_quat') and self.ee_quat is not None:
                if self.ee_quat.shape[0] == N and hasattr(self, 'prev_ee_quat') and self.prev_ee_quat.shape[0] == N:
                    self.prev_ee_quat.copy_(self.ee_quat)
        except Exception as e:
            print(f"[Warning] Failed to update prev_ee_quat: {e}")
        
        return total_reward, angle_h
    
    def _compute_obstacle_penalty_batch(self, cfg: dict) -> torch.Tensor:
        """
        基于论文公式的障碍物惩罚：ω2·Σψi
        参考论文: ψi = max(0, 1 - d/d_max) when d < d_max
        
        Args:
            cfg: 配置字典
        
        Returns:
            [N] 障碍物惩罚张量（负数表示惩罚）
        """
        N = self.num_envs
        penalty = torch.zeros(N, device=self.device)
        
        try:
            # 获取5个连杆到障碍物的距离
            link_distances = self._compute_link_obstacle_distances_batch(use_voxel=False)  # [num_envs, 5]
            
            # 确保形状正确
            if link_distances.shape[0] != N:
                if link_distances.shape[0] > N:
                    link_distances = link_distances[:N, :]
                else:
                    link_distances_padded = torch.full((N, 5), 10.0, device=self.device)
                    link_distances_padded[:link_distances.shape[0], :] = link_distances
                    link_distances = link_distances_padded
            
            # 论文参数
            omega2 = cfg['reward'].get('omega2', 0.15)  # 障碍物惩罚系数
            d_max = cfg['reward'].get('d_max', 0.1)   # 安全距离阈值（10cm，与config.yaml一致）
            
            # 计算每个连杆的惩罚系数 ψi
            # ψi = max(0, 1 - d/d_max) when d < d_max
            psi = torch.clamp(1.0 - link_distances / d_max, min=0.0)  # [N, 5]
            
            # 只对距离小于d_max的连杆应用惩罚
            psi[link_distances >= d_max] = 0.0
            
            # 论文公式：penalty = -ω2 * Σψi
            psi_sum = psi.sum(dim=1)  # [N] 每个环境的总和
            penalty = -omega2 * psi_sum
            
            # 检查 NaN/Inf
            if torch.isnan(penalty).any() or torch.isinf(penalty).any():
                nan_mask = torch.isnan(penalty) | torch.isinf(penalty)
                penalty[nan_mask] = 0.0
            
        except Exception as e:
            print(f"[Warning] _compute_obstacle_penalty_batch failed: {e}")
            import traceback
            traceback.print_exc()
        
        return penalty
    
    def _compute_simplified_exploration_rewards(self, action_xyz: torch.Tensor, cfg: dict) -> torch.Tensor:
        """
        简化的探索奖励（未发现目标阶段）
        整合：末端旋转 + 末端平移 + 向上探索
        
        Args:
            action_xyz: [N, 6] 关节角度增量
            cfg: 配置字典
        
        Returns:
            [N] 探索奖励张量
        """
        N = self.num_envs
        reward = torch.zeros(N, device=self.device)
        
        try:
            # 1. 末端旋转奖励（多角度探索）
            unknown_mask = ~self.target_discovered
            if unknown_mask.any():
                wrist_rotation_reward = self._compute_wrist_rotation_reward(cfg)
                if wrist_rotation_reward.shape[0] == N:
                    reward += wrist_rotation_reward
            
            # 2. 末端平移奖励（避免静止）
            # 简化：使用关节运动幅度作为移动量度
            movement_magnitude = torch.norm(action_xyz, dim=1)  # [N]
            wrist_movement_coef = cfg['reward'].get('wrist_movement_coef', 2.0)
            reward += wrist_movement_coef * movement_magnitude
            
            # 3. 向上探索奖励（鼓励向上移动）
            unknown_mask = ~self.target_discovered
            if unknown_mask.any():
                upward_reward = self._compute_upward_exploration_reward(cfg)
                if upward_reward.shape[0] == N:
                    reward += upward_reward
            
            # 检查 NaN/Inf
            if torch.isnan(reward).any() or torch.isinf(reward).any():
                nan_mask = torch.isnan(reward) | torch.isinf(reward)
                reward[nan_mask] = 0.0
                
        except Exception as e:
            print(f"[Warning] _compute_simplified_exploration_rewards failed: {e}")
            import traceback
            traceback.print_exc()
        
        return reward
    
    def _compute_simplified_observation_rewards(self, action_xyz: torch.Tensor, cfg: dict) -> torch.Tensor:
        """
        简化的观察奖励（已发现目标阶段）
        包含：多角度观察 + 相机朝向变化
        
        Args:
            action_xyz: [N, 6] 关节角度增量
            cfg: 配置字典
        
        Returns:
            [N] 观察奖励张量
        """
        N = self.num_envs
        reward = torch.zeros(N, device=self.device)
        
        try:
            # 1. 多角度观察奖励
            discovered_mask = self.target_discovered
            if discovered_mask.any():
                multiview_reward = self._compute_multiview_observation_reward(action_xyz, cfg)
                if multiview_reward.shape[0] == N:
                    reward += multiview_reward
            
            # 2. 相机朝向变化奖励（简化：使用末端旋转）
            wrist_rotation_reward = self._compute_wrist_rotation_reward(cfg)
            view_change_coef = cfg['reward'].get('view_change_coef', 0.15)
            if wrist_rotation_reward.shape[0] == N:
                reward += view_change_coef * wrist_rotation_reward
            
            # 🎯 注意：端到端到达目标奖励已移至 _compute_approaching_reward()，在发现后才应用
            
            # 检查 NaN/Inf
            if torch.isnan(reward).any() or torch.isinf(reward).any():
                nan_mask = torch.isnan(reward) | torch.isinf(reward)
                reward[nan_mask] = 0.0
                
        except Exception as e:
            print(f"[Warning] _compute_simplified_observation_rewards failed: {e}")
            import traceback
            traceback.print_exc()
        
        return reward
    
    def _compute_approaching_reward(self, cfg: dict) -> torch.Tensor:
        """
        靠近奖励（发现ArUco码后的端到端到达目标奖励）
        
        只在发现目标后应用，鼓励机械臂靠近并到达目标。
        
        公式：
        1. 位置误差惩罚: -ω₁e² (误差越大，惩罚越大)
        2. 零误差鼓励: ln(e² + τₑ) (误差越小，奖励越大；误差为0时奖励最大)
        e = ||pt - pe|| = ||target_pos - ee_pos|| (欧氏距离)
        
        Args:
            cfg: 配置字典
        
        Returns:
            [N] 靠近奖励张量（只对已发现目标的环境应用）
        """
        N = self.num_envs
        reward = torch.zeros(N, device=self.device)
        
        try:
            # 只对已发现目标的环境应用靠近奖励
            discovered_mask = self.target_discovered  # [N]
            
            if not discovered_mask.any():
                return reward  # 没有环境发现目标，返回零奖励
            
            # 获取配置参数
            omega1 = cfg['reward'].get('omega1', 1.0)  # 误差权重系数（与config.yaml一致）
            tau_e = cfg['reward'].get('tau_e', 2.0)   # 误差阈值
            
            # 计算末端执行器与目标的欧氏距离
            error = torch.norm(self.target_pos - self.ee_pos, dim=1)  # [N]
            error_squared = error ** 2  # e²
            
            # 位置误差惩罚: -ω₁e² (负数，误差越大惩罚越大)
            position_error_penalty = -omega1 * error_squared
            
            # 零误差鼓励: ln(e² + τₑ) (正数，误差越小奖励越大)
            zero_error_encouragement = torch.log(error_squared + tau_e)
            
            # 组合到达目标奖励
            reaching_reward = position_error_penalty + zero_error_encouragement
            
            # 只对已发现目标的环境应用奖励
            reward[discovered_mask] = reaching_reward[discovered_mask]
            
            # 检查 NaN/Inf
            if torch.isnan(reward).any() or torch.isinf(reward).any():
                nan_mask = torch.isnan(reward) | torch.isinf(reward)
                reward[nan_mask] = 0.0
                
        except Exception as e:
            print(f"[Warning] _compute_approaching_reward failed: {e}")
            import traceback
            traceback.print_exc()
        
        return reward
    
    def _compute_success_reward(self, cfg: dict) -> torch.Tensor:
        """
        成功奖励（一次性，当到达目标时）
        
        当末端执行器距离目标小于success_distance时，给予一次性大奖励。
        使用标志位避免重复奖励。
        
        Args:
            cfg: 配置字典
        
        Returns:
            [N] 成功奖励张量（只对成功到达目标的环境应用一次）
        """
        N = self.num_envs
        reward = torch.zeros(N, device=self.device)
        
        try:
            # 获取配置参数
            success_reward = cfg['reward'].get('success_reward', 100.0)  # 成功奖励值
            success_distance = cfg['reward'].get('success_distance', 0.1)  # 成功距离阈值
            
            # 计算末端执行器与目标的欧氏距离
            error = torch.norm(self.target_pos - self.ee_pos, dim=1)  # [N]
            
            # 检查是否成功到达目标
            success_mask = error < success_distance  # [N]
            
            # 初始化成功标志位（如果不存在）
            if not hasattr(self, '_success_reward_given'):
                self._success_reward_given = torch.zeros(N, dtype=torch.bool, device=self.device)
            
            # 确保标志位形状正确
            if self._success_reward_given.shape[0] != N:
                if self._success_reward_given.shape[0] > N:
                    self._success_reward_given = self._success_reward_given[:N]
                else:
                    padding = torch.zeros(N - self._success_reward_given.shape[0], dtype=torch.bool, device=self.device)
                    self._success_reward_given = torch.cat([self._success_reward_given, padding])
            
            # 只对首次成功到达的环境给予奖励
            newly_successful = success_mask & (~self._success_reward_given)  # [N]
            
            # 给予成功奖励
            reward[newly_successful] = success_reward
            
            # 更新标志位
            self._success_reward_given[newly_successful] = True
            
            # 检查 NaN/Inf
            if torch.isnan(reward).any() or torch.isinf(reward).any():
                nan_mask = torch.isnan(reward) | torch.isinf(reward)
                reward[nan_mask] = 0.0
                
        except Exception as e:
            print(f"[Warning] _compute_success_reward failed: {e}")
            import traceback
            traceback.print_exc()
        
        return reward
    
    def _compute_discovery_rewards(self, cfg: dict) -> torch.Tensor:
        """
        探索发现奖励（目标发现）
        
        Returns:
            reward: [N] 发现奖励张量
        """
        N = self.num_envs
        reward = torch.zeros(N, device=self.device)
        
        # 批量处理所有环境的发现状态
        if not hasattr(self, 'prev_discovery_state') or self.prev_discovery_state is None:
            self.prev_discovery_state = torch.zeros(N, dtype=torch.bool, device=self.device)
        
        if self.prev_discovery_state.shape[0] != N:
            self.prev_discovery_state = torch.zeros(N, dtype=torch.bool, device=self.device)
        
        # 批量检测：从上一步未发现变为发现的环境
        newly_discovered = self.target_discovered & (~self.prev_discovery_state)
        
        if newly_discovered.any():
            discovery_bonus = cfg['reward'].get('discovery_bonus', 5.0)
            reward[newly_discovered] += discovery_bonus
        
        # 更新历史状态
        self.prev_discovery_state = self.target_discovered.clone()
        
        return reward
    
    def _estimate_obstacle_distance_from_voxel(self, voxel_data: torch.Tensor) -> Optional[float]:
        """
        从体素地图估计最近障碍物距离
        
        Args:
            voxel_data: 体素地图 [1, 1, Vx, Vy, Vz] 或 [1, Vx, Vy, Vz] 或 [Vx, Vy, Vz]
        
        Returns:
            估计的障碍物距离（米），如果无法估计则返回None
        """
        if self.voxel_grid_origin is None or self.voxel_res is None:
            return None
        
        ee_pos = self.ee_pos[0]
        return self._compute_nearest_obstacle_from_voxel(ee_pos)
    
    def _compute_multiview_observation_reward(self, action_xyz: torch.Tensor, cfg: dict) -> torch.Tensor:
        """
        多角度观察奖励：发现目标后，鼓励多角度观察以验证安全路径（适配关节角度增量控制）
        
        核心思想：
        - 发现目标后，不鼓励接近，而是鼓励多角度观察
        - 奖励观察不同角度（鼓励探索性运动）
        - 验证安全路径，满足条件后探索成功终止
        
        Args:
            action_xyz: [N, 6] 关节角度增量 [dq1, dq2, dq3, dq4, dq5, dq6]（弧度）
            cfg: 配置字典
        
        Returns:
            [N] 多角度观察奖励张量
        """
        N = self.num_envs
        reward = torch.zeros(N, device=self.device)
        
        # 🎯 修复：批量处理所有已发现目标的环境
        discovered_mask = self.target_discovered
        if not discovered_mask.any():
            return reward
        
        # 为每个已发现目标的环境计算奖励
        for i in range(N):
            if not discovered_mask[i]:
                continue
            
            # 🎯 修复：使用支持多环境的camera_pose
            cam_T = self.camera_pose(env_idx=i)  # [4, 4]
            cam_pos = cam_T[:3, 3]  # [3]
            
            target_pos = self.target_pos[i]
            
            # 计算从目标到相机的方向向量（观察方向）
            rel = cam_pos - target_pos
            rel_xy = rel[:2]  # 投影到XY平面
            if torch.norm(rel_xy) > 1e-6:
                observation_angle = torch.atan2(rel_xy[1], rel_xy[0]).item()  # 弧度
                observation_angle_deg = math.degrees(observation_angle)  # 转换为度
            else:
                observation_angle_deg = 0.0
            
            # 2. 奖励新角度观察（鼓励多角度确认）
            angle_threshold = cfg['reward'].get('multiview_angle_threshold', 30.0)  # 30度视为新角度
            
            # 获取该环境的历史角度列表
            if isinstance(self.observation_angles, list) and i < len(self.observation_angles):
                angles_list = self.observation_angles[i] if isinstance(self.observation_angles[i], list) else []
            else:
                angles_list = []
            
            is_new_angle = True
            for prev_angle in angles_list:
                # 忽略非数值项
                if isinstance(prev_angle, (list, tuple)):
                    continue
                angle_diff = abs(observation_angle_deg - float(prev_angle))
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                if angle_diff < angle_threshold:
                    is_new_angle = False
                    break
            
            if is_new_angle:
                angles_list.append(observation_angle_deg)
                if len(angles_list) > 12:
                    angles_list.pop(0)
                # 更新observation_angles
                if isinstance(self.observation_angles, list):
                    if i < len(self.observation_angles):
                        self.observation_angles[i] = angles_list
                    else:
                        while len(self.observation_angles) <= i:
                            self.observation_angles.append([])
                        self.observation_angles[i] = angles_list
                
                new_angle_bonus = cfg['reward'].get('new_angle_bonus', 0.2)
                reward[i] += new_angle_bonus
            
            # 3. 鼓励探索性运动（末端关节运动，改变视角）- 批量处理
            wrist_increment = action_xyz[i, 3:]  # 末端关节 [dq4, dq5, dq6]
            wrist_movement = torch.norm(wrist_increment).item()
            
            # 末端关节运动越大，视角变化越大
            multiview_movement_bonus = cfg['reward'].get('multiview_movement_bonus', 0.1)
            multiview_movement_scale = cfg['reward'].get('multiview_movement_scale', 50.0)
            reward[i] += multiview_movement_bonus * wrist_movement * multiview_movement_scale
            
            # 4. 持续观察奖励（鼓励持续多角度观察）
            multiview_continuation_bonus = cfg['reward'].get('multiview_continuation_bonus', 0.05)
            multiview_continuation_steps = cfg['reward'].get('multiview_continuation_steps', 10)
            steps_factor = torch.clamp(self.multiview_steps[i].float() / float(multiview_continuation_steps), max=1.0).item()
            reward[i] += multiview_continuation_bonus * steps_factor
            
            
            # 5. 多角度覆盖奖励（观察角度越多，奖励越大）
            angle_coverage_bonus = cfg['reward'].get('angle_coverage_bonus', 0.15)
            angle_coverage_normalize = cfg['reward'].get('angle_coverage_normalize', 8.0)
            num_unique_angles = len(angles_list)
            # 如果观察到4个以上不同角度，给予额外奖励
            if num_unique_angles >= 4:
                reward[i] += angle_coverage_bonus * (num_unique_angles / angle_coverage_normalize)
        
        return reward
    
    def _compute_wrist_rotation_reward(self, cfg: dict) -> torch.Tensor:
        """
        末端转动奖励：在未知条件下（未发现目标时），鼓励末端多角度转动以探索环境
        
        Args:
            cfg: 配置字典
        
        Returns:
            [num_envs] 奖励张量
        """
        try:
            num_envs = self.num_envs
            
            # 初始化奖励张量
            reward = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
            
            # 检查状态是否存在
            if not hasattr(self, 'ee_quat') or self.ee_quat is None:
                return reward
            if not hasattr(self, 'prev_ee_quat') or self.prev_ee_quat is None:
                return reward
            
            # 确保形状匹配
            if self.ee_quat.shape[0] != num_envs:
                return reward
            if self.prev_ee_quat.shape[0] != num_envs:
                return reward
            
            # 🎯 安全检查：确保四元数的形状是正确的 [num_envs, 4]
            if self.ee_quat.shape[1] != 4 or self.prev_ee_quat.shape[1] != 4:
                return reward
            
            # 🎯 批量计算四元数相对旋转（使用张量操作，避免循环）
            # 当前和前一帧的四元数
            q_curr = self.ee_quat.clone()  # [num_envs, 4]，使用clone避免原地修改
            q_prev = self.prev_ee_quat.clone()  # [num_envs, 4]，使用clone避免原地修改
            
            # 检查是否有 NaN/Inf
            curr_valid = torch.isfinite(q_curr).all(dim=1)  # [num_envs]
            prev_valid = torch.isfinite(q_prev).all(dim=1)  # [num_envs]
            valid_mask = curr_valid & prev_valid
            
            # 对于无效的四元数，使用单位四元数
            if not valid_mask.all():
                q_curr = torch.where(valid_mask.unsqueeze(1).expand_as(q_curr), 
                                    q_curr, 
                                    torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand_as(q_curr))
                q_prev = torch.where(valid_mask.unsqueeze(1).expand_as(q_prev), 
                                    q_prev, 
                                    torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand_as(q_prev))
            
            # 归一化四元数（批量处理）
            q_curr_norm = torch.zeros_like(q_curr)
            q_prev_norm = torch.zeros_like(q_prev)
            
            q_curr_norms = torch.norm(q_curr, dim=1, keepdim=True)  # [num_envs, 1]
            q_prev_norms = torch.norm(q_prev, dim=1, keepdim=True)  # [num_envs, 1]
            
            # 避免除零
            q_curr_norm = torch.where(q_curr_norms > 1e-6, 
                                      q_curr / (q_curr_norms + 1e-8), 
                                      torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand_as(q_curr))
            q_prev_norm = torch.where(q_prev_norms > 1e-6, 
                                      q_prev / (q_prev_norms + 1e-8), 
                                      torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand_as(q_prev))
            
            # 计算相对旋转四元数：q_relative = q_prev^-1 * q_curr
            # 四元数逆：对于单位四元数 [x,y,z,w]，逆为 [-x,-y,-z,w]
            q_prev_inv = torch.cat([
                -q_prev_norm[:, :3],  # -x, -y, -z
                q_prev_norm[:, 3:4]   # w
            ], dim=1)  # [num_envs, 4]
            
            # 四元数乘法：q_relative = q_prev_inv * q_curr
            # q1 * q2 = [q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
            #            q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
            #            q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
            #            q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z]
            q_rel = torch.zeros_like(q_curr_norm)
            q_rel[:, 0] = q_prev_inv[:, 3] * q_curr_norm[:, 0] + q_prev_inv[:, 0] * q_curr_norm[:, 3] + q_prev_inv[:, 1] * q_curr_norm[:, 2] - q_prev_inv[:, 2] * q_curr_norm[:, 1]
            q_rel[:, 1] = q_prev_inv[:, 3] * q_curr_norm[:, 1] - q_prev_inv[:, 0] * q_curr_norm[:, 2] + q_prev_inv[:, 1] * q_curr_norm[:, 3] + q_prev_inv[:, 2] * q_curr_norm[:, 0]
            q_rel[:, 2] = q_prev_inv[:, 3] * q_curr_norm[:, 2] + q_prev_inv[:, 0] * q_curr_norm[:, 1] - q_prev_inv[:, 1] * q_curr_norm[:, 0] + q_prev_inv[:, 2] * q_curr_norm[:, 3]
            q_rel[:, 3] = q_prev_inv[:, 3] * q_curr_norm[:, 3] - q_prev_inv[:, 0] * q_curr_norm[:, 0] - q_prev_inv[:, 1] * q_curr_norm[:, 1] - q_prev_inv[:, 2] * q_curr_norm[:, 2]
            
            # 归一化相对旋转四元数（添加数值稳定性检查）
            q_rel_norms = torch.norm(q_rel, dim=1, keepdim=True)  # [num_envs, 1]
            q_rel_norm = torch.where(q_rel_norms > 1e-6, 
                                     q_rel / (q_rel_norms + 1e-8), 
                                     torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand_as(q_rel))
            
            # 检查归一化后的四元数是否有效
            q_rel_valid = torch.isfinite(q_rel_norm).all(dim=1)  # [num_envs]，检查每一行的所有元素是否有限
            if not q_rel_valid.all():
                invalid_mask = ~q_rel_valid  # [num_envs]
                num_invalid = invalid_mask.sum().item()
                if num_invalid > 0:
                    default_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(num_invalid, 4)
                    q_rel_norm[invalid_mask] = default_quat
            
            # 计算旋转角度（从四元数的 w 分量提取）
            # 对于单位四元数 [x,y,z,w]，旋转角度为 2*acos(|w|)
            # 🎯 修复：更严格的数值稳定性检查，避免 acos 接收超出范围的值
            w_abs = torch.clamp(torch.abs(q_rel_norm[:, 3]), min=1e-6, max=1.0 - 1e-6)  # [num_envs]，避免边界值
            # 进一步检查，确保没有 NaN 或 Inf
            valid_w_mask = torch.isfinite(w_abs) & (w_abs >= 0.0) & (w_abs <= 1.0)
            if not valid_w_mask.all():
                # 如果有无效值，设为安全默认值
                w_abs = torch.where(valid_w_mask, w_abs, torch.ones_like(w_abs) * 0.999)  # 接近1但不等于1
            
            rotation_angle_rad = 2.0 * torch.acos(w_abs)  # [num_envs]
            rotation_angle_deg = torch.rad2deg(rotation_angle_rad)  # [num_envs]
            
            # 再次检查角度是否有效
            if torch.isnan(rotation_angle_deg).any() or torch.isinf(rotation_angle_deg).any():
                nan_inf_mask = torch.isnan(rotation_angle_deg) | torch.isinf(rotation_angle_deg)
                rotation_angle_deg[nan_inf_mask] = 0.0
            
            # 🎯 奖励设计：
            # 1. 基础旋转奖励：旋转角度越大，奖励越高（有上限）
            # 🎯 优化：使用非线性奖励，鼓励更大的旋转角度
            rotation_coef = cfg['reward'].get('wrist_rotation_coef', 0.05)
            rotation_threshold = cfg['reward'].get('wrist_rotation_threshold', 5.0)  # 最小旋转角度阈值（度）
            
            # 只奖励超过阈值的环境
            valid_rotation_mask = rotation_angle_deg > rotation_threshold
            # 🎯 修复：基础旋转奖励只对未发现目标的环境应用（避免与view_change_coef重复）
            unknown_mask = ~self.target_discovered  # [num_envs]
            valid_rotation_and_unknown_mask = valid_rotation_mask & unknown_mask
            if valid_rotation_and_unknown_mask.any():
                # 🎯 优化：使用平方函数，大幅奖励大角度旋转
                # 归一化到 [0, 1]，使用平方以鼓励更大角度的旋转
                rotation_max_angle = cfg['reward'].get('wrist_rotation_max_angle', 180.0)
                normalized_rotation = torch.clamp(rotation_angle_deg / rotation_max_angle, max=1.0)
                # 使用平方使大角度获得更多奖励，鼓励大幅旋转
                reward_factor = normalized_rotation[valid_rotation_and_unknown_mask] ** 1.5  # 使用1.5次方，介于线性和平方之间
                # 🎯 增大基础奖励，确保即使小角度也有足够的激励，但大角度奖励更多
                reward[valid_rotation_and_unknown_mask] = rotation_coef * (0.2 + 0.8 * reward_factor)  # 最小20%奖励，最多100%，大角度奖励更高
            
            # 🎯 添加末端不转动惩罚（在未知条件下）
            no_rotation_mask = (rotation_angle_deg <= rotation_threshold) & unknown_mask  # 未转动且未发现目标
            if no_rotation_mask.any():
                no_rotation_penalty_coef = cfg['reward'].get('wrist_no_rotation_penalty', 0.01)  # 不转动惩罚系数
                # 惩罚与旋转角度成反比（角度越小，惩罚越大，但不超过阈值）
                penalty_factor = 1.0 - torch.clamp(rotation_angle_deg / rotation_threshold, min=0.0, max=1.0)
                reward[no_rotation_mask] -= no_rotation_penalty_coef * penalty_factor[no_rotation_mask]
            
            # 2. 连续转动奖励：如果连续多步都在转动，给予额外奖励
            # 🎯 注意：只对未发现目标的环境更新计数器（因为只对它们应用奖励）
            if hasattr(self, 'wrist_rotation_steps'):
                # 注意：unknown_mask 已在上面定义，这里不需要重复定义
                # 增加转动步数计数器（只对有效转动且未发现目标的环境）
                valid_and_unknown_mask = valid_rotation_mask & unknown_mask
                self.wrist_rotation_steps[valid_and_unknown_mask] += 1
                # 重置未转动或已发现目标的环境的计数器
                self.wrist_rotation_steps[~valid_and_unknown_mask] = 0
                
                # 连续转动奖励（只对有效转动且未发现目标的环境应用）
                continuation_coef = cfg['reward'].get('wrist_rotation_continuation_coef', 0.02)
                continuation_steps = cfg['reward'].get('wrist_rotation_continuation_steps', 5)
                # 🎯 优化：更快达到最大奖励（从10步改为5步），并增加奖励幅度
                steps_factor = torch.clamp(self.wrist_rotation_steps.float() / float(continuation_steps), max=1.0)  # 更快达到最大值
                # 使用平方根使奖励增长更平滑
                steps_reward_factor = torch.sqrt(steps_factor[valid_and_unknown_mask])
                # 只对有效转动且未发现目标的环境应用连续转动奖励
                reward[valid_and_unknown_mask] += continuation_coef * steps_reward_factor
            
            # 检查 NaN/Inf
            if torch.isnan(reward).any() or torch.isinf(reward).any():
                nan_mask = torch.isnan(reward) | torch.isinf(reward)
                reward[nan_mask] = 0.0
            
            return reward
            
        except Exception as e:
            print(f"[Warning] Failed to compute wrist rotation reward: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(num_envs, device=self.device, dtype=torch.float32)
    
    def _compute_upward_exploration_reward(self, cfg: dict) -> torch.Tensor:
        """
        Z轴向上探索奖励：在未知条件下（未发现目标时），鼓励末端向上移动以探索高层空间
        
        Args:
            cfg: 配置字典
            
        Returns:
            reward: [num_envs] 形状的奖励张量
        """
        num_envs = self.num_envs
        
        try:
            # 获取未发现目标的环境
            unknown_mask = ~self.target_discovered  # [num_envs]
            if not unknown_mask.any():
                return torch.zeros(num_envs, device=self.device, dtype=torch.float32)
            
            reward = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
            
            # 获取当前末端位置
            if not hasattr(self, 'ee_pos') or self.ee_pos is None:
                return reward
            
            # 获取上一次的末端位置（如果存在）
            if not hasattr(self, '_prev_ee_pos_upward') or self._prev_ee_pos_upward is None:
                self._prev_ee_pos_upward = self.ee_pos.clone()
            
            # 计算z轴方向的变化（向上为正）
            dz = self.ee_pos[:, 2] - self._prev_ee_pos_upward[:, 2]  # [num_envs]
            
            # 获取配置参数
            upward_coef = cfg['reward'].get('upward_exploration_coef', 10.0)
            upward_threshold = cfg['reward'].get('upward_exploration_threshold', 0.01)  # 最小向上移动阈值（米）
            max_reward_height = cfg['reward'].get('upward_exploration_max_reward_height', 1.0)  # 最大奖励的高度阈值（米）
            
            # 只对未发现目标的环境计算奖励
            unknown_indices = torch.where(unknown_mask)[0]
            
            if len(unknown_indices) > 0:
                # 计算向上的移动（只考虑向上移动）
                upward_movement = torch.clamp(dz[unknown_indices], min=0.0)  # [num_unknown_envs]
                
                # 只奖励超过阈值的向上移动
                valid_upward_mask = upward_movement > upward_threshold
                
                if valid_upward_mask.any():
                    # 基础奖励：与向上移动距离成正比
                    base_reward = upward_coef * upward_movement[valid_upward_mask]
                    
                    # 高度奖励：鼓励达到更高位置（但不超过最大高度）
                    current_height = self.ee_pos[unknown_indices[valid_upward_mask], 2]  # [num_valid]
                    height_factor = torch.clamp(current_height / max_reward_height, max=1.0)  # 归一化到[0, 1]
                    height_bonus = upward_coef * 0.5 * height_factor  # 额外的高度奖励
                    
                    # 总奖励 = 基础奖励 + 高度奖励
                    total_reward_value = base_reward + height_bonus
                    
                    # 应用奖励到对应的环境
                    valid_indices = unknown_indices[valid_upward_mask]
                    reward[valid_indices] = total_reward_value
            
            # 更新上一次的位置
            self._prev_ee_pos_upward.copy_(self.ee_pos)
            
            # 检查 NaN/Inf
            if torch.isnan(reward).any() or torch.isinf(reward).any():
                nan_mask = torch.isnan(reward) | torch.isinf(reward)
                reward[nan_mask] = 0.0
            
            return reward
            
        except Exception as e:
            print(f"[Warning] Failed to compute upward exploration reward: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(num_envs, device=self.device, dtype=torch.float32)
    
    
    def _detect_exploration_pattern(self, action_xyz: torch.Tensor) -> torch.Tensor:
        """
        检测探索性运动模式（摆动、扫描等）- 适配关节角度增量控制
        
        Args:
            action_xyz: [N, 6] 关节角度增量 [dq1, dq2, dq3, dq4, dq5, dq6]（弧度）
        
        Returns:
            [N] 模式奖励分数（0-1之间）
        """
        # 🎯 修复：批量处理所有环境
        # 区分基座关节（0-2）和末端关节（3-5）
        base_joint_movement = torch.norm(action_xyz[:, :3], dim=1)  # [N] shoulder_pan, shoulder_lift, elbow
        wrist_joint_movement = torch.norm(action_xyz[:, 3:], dim=1)  # [N] wrist_1, wrist_2, wrist_3
        
        # 探索性模式：末端关节运动主导（摆动、扫描）
        # 如果末端关节运动明显大于基座关节运动，认为是探索性模式
        total_movement = base_joint_movement + wrist_joint_movement + 1e-6
        wrist_ratio = wrist_joint_movement / total_movement  # [N]
        
        # 末端关节占比高（>0.6）认为是探索性模式
        pattern_score = torch.clamp((wrist_ratio - 0.6) / 0.4, 0.0, 1.0)  # [N]
        
        return pattern_score

    # ================= 坐标转换（目标坐标系 → 世界） =================
    def _world_from_goal_rot2d(self) -> torch.Tensor:
        """
        计算 2D 旋转矩阵 R_w_g，将目标坐标系的平面向量变换到世界坐标系。
        目标坐标系定义：x 轴指向当前 EE→目标 的地面投影方向，y 轴为其左侧（右手系）。
        返回形状 [2,2] 的张量。
        """
        ee = self.ee_pos[0]
        tgt = self.target_pos[0]
        rel = tgt - ee
        # 投影到地面
        v = torch.tensor([rel[0], rel[1]], device=self.device)
        n = torch.norm(v) + 1e-9
        x_g = v / n
        y_g = torch.tensor([-x_g[1], x_g[0]], device=self.device)  # 左侧法向
        R = torch.stack([x_g, y_g], dim=1)  # [2,2], 列向量为基
        return R

    def _goal_from_world_rot2d(self) -> torch.Tensor:
        R_w_g = self._world_from_goal_rot2d()
        return R_w_g.t()

    # ============== 障碍物与 ArUco 检测辅助 ==================
    def _spawn_or_reset_obstacles(self, reset_only: bool = False):
        # 生成或重置球形障碍，满足尺寸与最小间距限制
        # 注意：为每个环境创建独立的障碍物
        if not reset_only:
            self.obstacle_handles = []  # 重置为列表
            self.obstacle_actual_radii = []
            self.obstacle_positions_np = []
        
        # 为每个环境生成障碍物位置（可以共享相同配置，但每个环境独立创建）
        for env_idx in range(self.num_envs):
            placed = []
            env = self.envs[env_idx]
            
            # 使用第一个环境的目标位置作为参考（简化实现）
            tgt_ref = self.target_pos[0].detach().cpu().numpy()
            
            for i in range(self.num_obstacles):
                # 采样半径与位置（以基座为圆环，避免过近）
                radius = np.random.uniform(self.sphere_radius_min, self.sphere_radius_max)
                for _ in range(100):
                    # 在可达工作空间内采样（圆柱约束）
                    rho = np.random.uniform(0.2, self.workspace_radius)
                    theta = np.random.uniform(-np.pi, np.pi)
                    x = rho * np.cos(theta)
                    y = rho * np.sin(theta)
                    z = np.random.uniform(self.workspace_z[0], self.workspace_z[1])
                    pos = np.array([x, y, z], dtype=np.float32)
                    # 与已有障碍、目标、基座的最小间距约束
                    ok = True
                    for p, r in placed:
                        if np.linalg.norm(pos - p) < (self.obstacle_min_spacing + r + radius):
                            ok = False; break
                    if ok and np.linalg.norm(pos - np.array([0.0, 0.0, 0.0], dtype=np.float32)) < (self.keepout_base_radius + radius):
                        ok = False
                    # 远离目标
                    if ok and np.linalg.norm(pos - tgt_ref) < (self.aruco_safe_radius + self.keepout_target_margin + radius):
                        ok = False
                    # 视线保护：不与初始末端到目标的走廊相交（线段-球最小距离）
                    ee0 = np.array([0.0, 0.0, 0.6], dtype=np.float32)  # 相机初始位置
                    if ok:
                        corridor_dist = self._point_to_segment_distance(pos, ee0, tgt_ref)
                        if corridor_dist < (radius + self.los_clearance):
                            ok = False
                    if ok:
                        placed.append((pos, radius))
                        break
                
                # 创建或移动 actor
                sphere_asset = self.gym.create_sphere(self.sim, radius, gymapi.AssetOptions())
                T = gymapi.Transform(); T.p = gymapi.Vec3(pos[0], pos[1], pos[2]); T.r = gymapi.Quat(0,0,0,1)
                
                if reset_only and len(self.obstacle_handles) > env_idx * self.num_obstacles + i:
                    # 重置模式下：使用 root_state tensor 设置障碍物位置
                    # 障碍物索引计算：每个环境有 robot(0) + target(1) + static_obstacles(num_obstacles) + dynamic_obstacles(num_dynamic_obstacles)
                    # 全局索引 = env_idx * self.actors_per_env + 2 + i
                    self.actors_per_env = 2 + self.num_obstacles + self.num_dynamic_obstacles  # robot + target + static + dynamic
                    obs_root_idx = env_idx * self.actors_per_env + 2 + i
                    
                    # ✅ 1) 绝对不要在 init 之后重新 acquire + wrap root_states
                    # 只刷新，不重绑
                    self.gym.refresh_actor_root_state_tensor(self.sim)
                    
                    if obs_root_idx < self.root_states.shape[0]:
                        new_root_state = self.root_states[obs_root_idx].clone()
                        new_root_state[0:3] = torch.tensor([pos[0], pos[1], pos[2]], device=self.device, dtype=torch.float32)
                        new_root_state[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
                        new_root_state[7:13] = 0.0
                        self.root_states[obs_root_idx] = new_root_state
                        
                        # 批量更新所有环境（简化：只更新当前环境）
                        # 🎯 使用统一的 _set_root_states_indexed 函数
                        actor_idx_cpu = torch.tensor([obs_root_idx], dtype=torch.int64, device='cpu')
                        pose_cpu = torch.zeros((1, 7), dtype=torch.float32, device='cpu')
                        pose_cpu[0, 0:3] = new_root_state[0:3].cpu()
                        pose_cpu[0, 3:7] = new_root_state[3:7].cpu()
                        self._set_root_states_indexed(actor_idx_cpu, pose_cpu, keep_vel=False)
                    
                    # 更新存储的半径和位置
                    handle_idx = env_idx * self.num_obstacles + i
                    if handle_idx < len(self.obstacle_actual_radii):
                        self.obstacle_actual_radii[handle_idx] = radius
                    if handle_idx < len(self.obstacle_positions_np):
                        self.obstacle_positions_np[handle_idx] = pos.copy()
                else:
                    # 创建新障碍物
                    h = self.gym.create_actor(env, sphere_asset, T, f"obs_{env_idx}_{i}", 0, 0)
                    self.obstacle_handles.append(h)
                    self.obstacle_actual_radii.append(radius)
                    self.obstacle_positions_np.append(pos.copy())
                    
                    # 设置障碍物颜色为蓝色
                    self.gym.set_rigid_body_color(
                        env, h, 0,
                        gymapi.MESH_VISUAL,
                        gymapi.Vec3(0.0, 0.0, 1.0)  # 蓝色
                    )
        
        # 更新障碍物位置和半径信息
        self._update_obstacle_positions()
        
        # ====== 第3步：静态障碍的初始落地（批量初始化）======
        # 布局（此时 dynamic 还没创建时）
        self.actors_per_env = 2 + self.num_obstacles  # robot(1) + target(1) + static
        print(f"[Layout] per-env: robot=1, target=1, static={self.num_obstacles}, dynamic=0, self.actors_per_env = {self.actors_per_env}")
        
        # 重新获取一次完整 root tensor
        self._reacquire_root_tensor()
        
        # ====== 批量落地静态障碍 root state ======
        if not reset_only and len(self.obstacle_handles) > 0:
            robot_offset   = 0
            target_offset  = 1
            static_offset  = 2
            all_indices = []
            all_states  = []
            
            for env_id in range(self.num_envs):
                base = env_id * self.actors_per_env
                for j in range(self.num_obstacles):
                    root_idx = base + static_offset + j
                    all_indices.append(root_idx)
                    
                    # 依据已有的静态障碍位姿数组
                    handle_idx = env_id * self.num_obstacles + j
                    if handle_idx < len(self.obstacle_positions_np):
                        pos = self.obstacle_positions_np[handle_idx]
                        px, py, pz = pos[0], pos[1], pos[2]
                    else:
                        # 兜底：使用默认位置
                        px, py, pz = 0.0, 0.0, 0.5
                    
                    state = torch.zeros(13, dtype=torch.float32)
                    state[0:3] = torch.tensor([px, py, pz], dtype=torch.float32)
                    state[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)  # 单位四元数
                    # 线速度/角速度全 0
                    all_states.append(state)
            
            if len(all_indices) > 0:
                idx = torch.tensor(all_indices, dtype=torch.int64)
                ns  = torch.stack(all_states, dim=0)  # (num_envs*num_obstacles, 13)
                ok = self._apply_root_states_indexed(idx, ns)
                if not ok:
                    print("[StaticInit] Failed to apply indexed root states for static obstacles.")
        
        # 🎯 关键修复：静态障碍物创建后，重新计算 actor 布局
        self._recompute_actor_layout()
    
    def _recompute_actor_layout(self):
        """
        统一定义每个环境内的 actor 顺序与数量，保证所有后续索引按相同布局计算。
        
        布局（每个 env）：
          0: robot
          1: target (ArUco)
          [2, ..., 2+S-1]: 静态障碍物 S = self.num_static_obstacles
          [2+S, ..., 2+S+D-1]: 动态障碍物 D = self.num_dynamic_obstacles
        """
        # 真实的静态障碍数：优先按 handle 数判断，其次按配置
        if hasattr(self, "obstacle_handles") and isinstance(self.obstacle_handles, list) and len(self.obstacle_handles) > 0:
            # 允许两种形态：
            #  - 扁平：长度 = num_envs * S
            #  - 分 env 的 list[list]：长度 = num_envs，每个子表长度 = S
            if len(self.obstacle_handles) == self.num_envs and isinstance(self.obstacle_handles[0], list):
                self.num_static_obstacles = len(self.obstacle_handles[0])
            else:
                # 扁平
                self.num_static_obstacles = (len(self.obstacle_handles) // max(1, self.num_envs))
        else:
            # 兜底（与配置保持一致）
            self.num_static_obstacles = getattr(self, "num_obstacles", 0)
        
        # 动态障碍（每 env）
        self.num_dynamic_obstacles = int(getattr(self, "num_dynamic_obstacles", 0))
        
        # 统一布局
        self.actors_per_env = 2 + self.num_static_obstacles + self.num_dynamic_obstacles
        self.static_start = 2
        self.dynamic_start = 2 + self.num_static_obstacles
        
        # 生成各类 root 索引（CPU int64）
        device_cpu = torch.device("cpu")
        env_ids = torch.arange(self.num_envs, device=device_cpu, dtype=torch.int64)
        base = env_ids * self.actors_per_env
        
        self.robot_root_indices = base.clone()                                  # [num_envs]
        self.target_root_indices = base + 1                                     # [num_envs]
        
        if self.num_static_obstacles > 0:
            off = torch.arange(self.num_static_obstacles, device=device_cpu, dtype=torch.int64)
            self.static_root_indices = (base.view(-1, 1) + (self.static_start + off).view(1, -1)).reshape(-1)
        else:
            self.static_root_indices = torch.empty(0, dtype=torch.int64, device=device_cpu)
        
        if self.num_dynamic_obstacles > 0:
            off = torch.arange(self.num_dynamic_obstacles, device=device_cpu, dtype=torch.int64)
            self.dynamic_root_indices = (base.view(-1, 1) + (self.dynamic_start + off).view(1, -1)).reshape(-1)
        else:
            self.dynamic_root_indices = torch.empty(0, dtype=torch.int64, device=device_cpu)
        
        # 供调试
        print(f"[Layout] per-env: robot=1, target=1, static={self.num_static_obstacles}, dynamic={self.num_dynamic_obstacles}, self.actors_per_env = {self.actors_per_env}")
    
    def _set_root_states_indexed(self, indices_cpu: torch.Tensor, new_poses_world_cpu: torch.Tensor, keep_vel=False):
        """
        用于批量更新一组 actor 的根状态（位置+朝向），确保形状严格匹配。
        
        参数:
          indices_cpu: [N] (int64, CPU) — 目标 actor root 索引
          new_poses_world_cpu: [N, 7] (float32, CPU) — 位置(x,y,z)+四元数(x,y,z,w)
          keep_vel: 是否保留当前速度（默认 False -> 置零）
        """
        assert indices_cpu.device.type == "cpu" and indices_cpu.dtype in (torch.int64, torch.long)
        assert new_poses_world_cpu.device.type == "cpu" and new_poses_world_cpu.shape[0] == indices_cpu.shape[0]
        
        N = indices_cpu.shape[0]
        if N == 0:
            return
        
        # 🎯 关键修复：使用 _apply_root_states_indexed 统一处理
        # 准备新状态（N, 13）
        new_states = torch.zeros((N, 13), dtype=torch.float32, device="cpu")
        new_states[:, 0:3] = new_poses_world_cpu[:, 0:3]     # pos
        new_states[:, 3:7] = new_poses_world_cpu[:, 3:7]     # quat (x,y,z,w)
        
        if keep_vel:
            # 复制当前速度
            self.gym.refresh_actor_root_state_tensor(self.sim)
            # 🎯 修复：不要重新 acquire，只 refresh（避免破坏绑定）
            if not hasattr(self, "root_states") or self.root_states is None:
                raise RuntimeError("[RootState] root_states not initialized! Call _reacquire_root_tensor() first.")
            cur_cpu = self.root_states.detach().cpu()
            new_states[:, 7:13] = cur_cpu[indices_cpu, 7:13]
        else:
            # 速度清零
            new_states[:, 7:13] = 0.0
        
        # 使用统一的 helper 函数
        if not self._apply_root_states_indexed(indices_cpu, new_states):
            print(f"[RootState] Failed to apply indexed root states for {N} actors")
    
    def _reacquire_root_tensor(self):
        """在创建完一批 actor（静态/动态）之后立即调用，保证拿到完整 (num_actors, 13) root tensor。"""
        _root = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root)

        # 🎯 关键修复：不要重新赋值 self.root_states，这会破坏与 Isaac Gym 的绑定！
        # Isaac Gym 返回的 tensor 应该已经是正确的格式
        # 如果格式不对，说明初始化有问题，这里只检查并警告
        if self.root_states.device.type != "cpu":
            print(f"[RootTensor] WARNING: root_states is on {self.root_states.device}, expected CPU")
        if self.root_states.dtype != torch.float32:
            print(f"[RootTensor] WARNING: root_states dtype is {self.root_states.dtype}, expected float32")
        if not self.root_states.is_contiguous():
            print(f"[RootTensor] WARNING: root_states is not contiguous")

        print(f"[RootTensor] Re-acquired: shape={tuple(self.root_states.shape)}")


    def _spawn_dynamic_obstacles(self):
        """创建动态障碍物 - 修复为动态物体"""
        if self.num_dynamic_obstacles == 0:
            return
        
        self.dynamic_obstacle_handles = []
        total_dyn_obs = self.num_envs * self.num_dynamic_obstacles
        
        # 初始化动态障碍物状态
        self.dyn_obs_state = torch.zeros((total_dyn_obs, 13), device=self.device)
        self.dyn_obs_state[:, 3] = 1.0  # 四元数w分量
        self.dyn_obs_goal = torch.zeros((total_dyn_obs, 3), device=self.device)
        self.dyn_obs_origin = torch.zeros((total_dyn_obs, 3), device=self.device)
        self.dyn_obs_radii = torch.zeros((total_dyn_obs,), device=self.device)
        self.dyn_obs_step_count = 0
        
        # 初始速度
        initial_vel = (self.dynamic_obstacle_vel_range[0] + self.dynamic_obstacle_vel_range[1]) / 2.0
        self.dyn_obs_vel_norm = torch.full((total_dyn_obs, 1), initial_vel, device=self.device)
        random_direction = torch.randn(total_dyn_obs, 3, device=self.device)
        random_direction = random_direction / (torch.norm(random_direction, dim=1, keepdim=True) + 1e-6)
        self.dyn_obs_vel = self.dyn_obs_vel_norm * random_direction
        
        # 🎯 关键修复：确保动态障碍物是动态物体
        for env_idx in range(self.num_envs):
            env = self.envs[env_idx]
            
            for dyn_idx in range(self.num_dynamic_obstacles):
                radius = np.random.uniform(
                    self.dynamic_obstacle_radius_min,
                    self.dynamic_obstacle_radius_max
                )
                
                # 🎯 修复1：创建动态障碍物资产
                asset_options = gymapi.AssetOptions()
                asset_options.density = 100.0  # 设置密度
                asset_options.disable_gravity = True  # 禁用重力，避免下落
                # 🎯 关键：确保是动态物体
                asset_options.fix_base_link = False  # 不固定基座
                
                sphere_asset = self.gym.create_sphere(self.sim, radius, asset_options)
                
                # 生成位置（保持原有逻辑）
                for _ in range(100):
                    rho = np.random.uniform(0.3, self.workspace_radius * 0.8)
                    theta = np.random.uniform(-np.pi, np.pi)
                    x = rho * np.cos(theta)
                    y = rho * np.sin(theta)
                    z = np.random.uniform(
                        self.workspace_z[0] + 0.1,
                        self.workspace_z[1] - 0.1
                    )
                    pos = np.array([x, y, z], dtype=np.float32)
                    
                    if np.linalg.norm(pos[:2]) < self.keepout_base_radius:
                        continue
                    
                    min_dist = 10.0
                    for static_idx in range(env_idx * self.num_obstacles, 
                                        (env_idx + 1) * self.num_obstacles):
                        if static_idx < len(self.obstacle_positions_np):
                            static_pos = self.obstacle_positions_np[static_idx]
                            dist = np.linalg.norm(pos - static_pos)
                            if static_idx < len(self.obstacle_actual_radii):
                                static_rad = self.obstacle_actual_radii[static_idx]
                            else:
                                static_rad = (self.sphere_radius_min + self.sphere_radius_max) / 2.0
                            min_dist = min(min_dist, dist - static_rad - radius)
                    
                    if min_dist > 0.15:
                        break
                
                # 创建动态障碍物
                T = gymapi.Transform()
                T.p = gymapi.Vec3(pos[0], pos[1], pos[2])
                T.r = gymapi.Quat(0, 0, 0, 1)
                
                # 🎯 修复2：创建为动态actor
                handle = self.gym.create_actor(
                    env, sphere_asset, T, 
                    f"dyn_obs_{env_idx}_{dyn_idx}", 
                    0,  # 碰撞组
                    0   # 碰撞过滤
                )
                
                # 🎯 修复3：将障碍物设置为动态（mass > 0）
                # 注意：静态物体（mass=0）无法通过 set_actor_root_state_tensor_indexed 更新位置
                # 所以我们需要使用动态物体，但在物理模拟后立即强制更新位置
                # Isaac Gym 的 RigidBodyProperties 只支持 mass 属性，不支持 damping
                # 阻尼需要在 asset 创建时通过 AssetOptions 设置
                rigid_props = self.gym.get_actor_rigid_body_properties(env, handle)
                if len(rigid_props) > 0:
                    # 设置为动态（mass > 0），这样才能通过 set_actor_root_state_tensor_indexed 更新
                    rigid_props[0].mass = 1.0
                    self.gym.set_actor_rigid_body_properties(env, handle, rigid_props)
                
                self.dynamic_obstacle_handles.append(handle)
                
                # 设置颜色为红色
                self.gym.set_rigid_body_color(
                    env, handle, 0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(1.0, 0.0, 0.0)  # 红色
                )
                
                # 存储初始状态
                global_idx = env_idx * self.num_dynamic_obstacles + dyn_idx
                self.dyn_obs_state[global_idx, :3] = torch.tensor(pos, device=self.device)
                self.dyn_obs_origin[global_idx] = torch.tensor(pos, device=self.device)
                
                # 生成初始目标
                goal_offset = torch.tensor([
                    np.random.uniform(-self.dynamic_obstacle_local_range[0], self.dynamic_obstacle_local_range[0]),
                    np.random.uniform(-self.dynamic_obstacle_local_range[1], self.dynamic_obstacle_local_range[1]),
                    np.random.uniform(-self.dynamic_obstacle_local_range[2], self.dynamic_obstacle_local_range[2])
                ], device=self.device)
                initial_goal = self.dyn_obs_origin[global_idx] + goal_offset
                initial_goal[0] = torch.clamp(initial_goal[0], -self.workspace_radius * 0.8, self.workspace_radius * 0.8)
                initial_goal[1] = torch.clamp(initial_goal[1], -self.workspace_radius * 0.8, self.workspace_radius * 0.8)
                initial_goal[2] = torch.clamp(initial_goal[2], self.workspace_z[0] + 0.1, self.workspace_z[1] - 0.1)
                self.dyn_obs_goal[global_idx] = initial_goal
                
                self.dyn_obs_radii[global_idx] = radius
        
        # 🎯 关键修复：在所有障碍物创建完成后，重新获取 root tensor（因为 actor 数量变了）
        # 这是唯一允许重新 acquire 的地方（创建完 actor 后）
        self._reacquire_root_tensor()
        
        if len(self.dynamic_obstacle_handles) > 0:
            try:
                # ====== 第5步：动态障碍物初始位置批量设置 ======
                # 计算每个环境的 actors 数量
                self.actors_per_env = 2 + self.num_obstacles + self.num_dynamic_obstacles
                
                # 批量准备所有动态障碍物的初始状态
                # 🎯 使用 get_actor_index 获取真实索引（不手算）
                all_dyn_indices = []
                all_dyn_states = []
                
                for i, handle in enumerate(self.dynamic_obstacle_handles):
                    env_idx = i // self.num_dynamic_obstacles
                    global_idx = i
                    
                    if global_idx < self.dyn_obs_state.shape[0]:
                        # 🎯 使用 get_actor_index 获取真实的全局 SIM 索引
                        sim_id = self.gym.get_actor_index(self.envs[env_idx], handle, gymapi.DOMAIN_SIM)
                        obs_root_idx = int(sim_id)
                        
                        if obs_root_idx < self.root_states.shape[0]:
                            all_dyn_indices.append(obs_root_idx)
                            
                            # 准备初始状态
                            new_state = torch.zeros(13, dtype=torch.float32, device='cpu')
                            new_state[0:3] = self.dyn_obs_state[global_idx, :3].cpu()
                            new_state[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device='cpu')
                            new_state[7:13] = 0.0  # 初始速度为0
                            all_dyn_states.append(new_state)
                
                # 🎯 批量更新所有动态障碍物的初始位置
                if len(all_dyn_indices) > 0:
                    idx_tensor = torch.tensor(all_dyn_indices, dtype=torch.int64, device='cpu')
                    states_tensor = torch.stack(all_dyn_states, dim=0)  # (num_dyn, 13)
                    ok = self._apply_root_states_indexed(idx_tensor, states_tensor)
                    if not ok:
                        print("[DynamicInit] Failed to apply indexed root states for dynamic obstacles.")
                
                # 🎯 刷新并验证初始位置
                self.gym.refresh_actor_root_state_tensor(self.sim)
                
                # 验证前几个障碍物的位置（使用缓存的索引）
                if hasattr(self, 'dyn_root_indices') and self.dyn_root_indices.numel() > 0:
                    for i in range(min(2, len(self.dynamic_obstacle_handles), self.dyn_root_indices.numel())):
                        obs_root_idx = int(self.dyn_root_indices[i].item())
                        if obs_root_idx < self.root_states.shape[0]:
                            actual_pos = self.root_states[obs_root_idx, :3].cpu().numpy()
                            expected_pos = self.dyn_obs_state[i, :3].cpu().numpy()
                            diff = np.linalg.norm(actual_pos - expected_pos)
                            if diff > 0.01:
                                print(f"[Warning] Failed to set initial position for obstacle {i}: expected={expected_pos}, actual={actual_pos}, diff={diff:.6f}m")
                            else:
                                print(f"[Debug] Successfully set initial position for obstacle {i}: {actual_pos}")
                
                print(f"[Dynamic Obstacles] ✅ 成功创建 {len(self.dynamic_obstacle_handles)} 个动态障碍物")
                
                # 🎯 调试：验证障碍物类型
                for i, handle in enumerate(self.dynamic_obstacle_handles[:2]):  # 检查前2个
                    env_idx = i // self.num_dynamic_obstacles
                    env = self.envs[env_idx]
                    rigid_props = self.gym.get_actor_rigid_body_properties(env, handle)
                    if len(rigid_props) > 0:
                        mass = rigid_props[0].mass
                        print(f"[Debug] Dynamic obstacle {i} mass: {mass}")
                        
            except Exception as e:
                print(f"[Warning] Failed to update dynamic obstacle positions during spawn: {e}")
                import traceback
                traceback.print_exc()
                
        # ====== 第4步：动态障碍创建之后，更新布局 + 重新获取 root tensor ======
        self.actors_per_env = 2 + self.num_obstacles + self.num_dynamic_obstacles
        self._reacquire_root_tensor()
        print(f"[Layout] per-env: robot=1, target=1, static={self.num_obstacles}, dynamic={self.num_dynamic_obstacles}, self.actors_per_env = {self.actors_per_env}")
        
        # 🎯 关键修复：动态障碍物创建后，重新计算 actor 布局
        self._recompute_actor_layout()
        
        # 🎯 关键修复：使用 get_actor_index 获取真实 SIM 索引（不手算）
        if len(self.dynamic_obstacle_handles) > 0:
            all_dyn_indices = []
            for i, handle in enumerate(self.dynamic_obstacle_handles):
                env_idx = i // self.num_dynamic_obstacles
                # 🎯 使用 get_actor_index 获取真实的全局 SIM 索引
                sim_id = self.gym.get_actor_index(self.envs[env_idx], handle, gymapi.DOMAIN_SIM)
                all_dyn_indices.append(int(sim_id))
            self.dyn_root_indices = torch.tensor(all_dyn_indices, dtype=torch.int32, device='cpu').contiguous()
            print(f"[Debug] Cached {self.dyn_root_indices.numel()} dynamic obstacle root indices (via get_actor_index): {self.dyn_root_indices.tolist()}")
                
    def test_dynamic_obstacles_visibility(self):
        """测试动态障碍物是否可见和移动"""
        print("=== Testing Dynamic Obstacles Visibility ===")
        
        # 设置一个障碍物到明显位置
        test_pos = torch.tensor([1.0, 0.0, 0.8], device=self.device)
        self.dyn_obs_state[0, :3] = test_pos
        
        # 更新到仿真
        self._update_dynamic_obstacle_positions_to_sim()
        
        print("请检查可视化窗口：")
        print("1. 是否能看到红色的动态障碍物球体？")
        print("2. 位置是否在 [1.0, 0.0, 0.8] 附近？")
        
        # 测试移动
        for i in range(5):
            # 移动障碍物
            self.dyn_obs_state[0, 0] -= 0.2  # 向左移动
            self._update_dynamic_obstacle_positions_to_sim()
            
            print(f"移动测试 {i+1}: 障碍物位置 x={self.dyn_obs_state[0, 0].item():.2f}")
            
            # 等待一下，让用户看到变化
            import time
            time.sleep(1.0)

    def _update_obstacle_positions(self):
        """
        更新障碍物位置和半径信息（用于距离计算）
        
        注意：障碍物存储结构：
        - obstacle_handles: 所有环境的静态障碍物列表，总数为 num_envs * num_obstacles
        - dynamic_obstacle_handles: 所有环境的动态障碍物列表，总数为 num_envs * num_dynamic_obstacles
        - 环境 env_idx 的静态障碍物索引范围：[env_idx * num_obstacles, (env_idx + 1) * num_obstacles)
        - 环境 env_idx 的动态障碍物索引范围：[env_idx * num_dynamic_obstacles, (env_idx + 1) * num_dynamic_obstacles)
        """
        # 🎯 更新静态障碍物位置
        total_obstacles = len(self.obstacle_handles)
        
        # 扩展 obstacle_positions 和 obstacle_radii 以存储所有环境的静态障碍物
        if len(self.obstacle_positions) < total_obstacles:
            # 扩展张量以容纳所有障碍物
            new_positions = torch.zeros((total_obstacles, 3), device=self.device)
            new_radii = torch.zeros((total_obstacles,), device=self.device)
            if len(self.obstacle_positions) > 0:
                new_positions[:len(self.obstacle_positions)] = self.obstacle_positions
                new_radii[:len(self.obstacle_radii)] = self.obstacle_radii
            self.obstacle_positions = new_positions
            self.obstacle_radii = new_radii
        
        # 更新所有静态障碍物的位置和半径
        for i in range(total_obstacles):
            if i < len(self.obstacle_positions_np):
                self.obstacle_positions[i] = torch.tensor(
                    self.obstacle_positions_np[i],
                    device=self.device
                )
            if i < len(self.obstacle_actual_radii):
                self.obstacle_radii[i] = torch.tensor(
                    self.obstacle_actual_radii[i],
                    device=self.device
                )
            else:
                # 回退：使用固定值范围的中点
                self.obstacle_radii[i] = torch.tensor(
                    (self.sphere_radius_min + self.sphere_radius_max) / 2.0,
                    device=self.device
                )
        
        # 🎯 更新动态障碍物位置（从dyn_obs_state读取）
        if self.num_dynamic_obstacles > 0 and self.dyn_obs_state is not None:
            # 动态障碍物位置已通过_move_dynamic_obstacles()更新到dyn_obs_state中
            # 这里不需要额外更新，因为位置直接从dyn_obs_state读取
            pass

    def _move_dynamic_obstacles(self):
        """更新动态障碍物位置（参考isaac-training的实现）"""
        try:
            if self.num_dynamic_obstacles == 0 or self.dyn_obs_state is None or len(self.dynamic_obstacle_handles) == 0:
                return
            
            total_dyn_obs = self.num_envs * self.num_dynamic_obstacles
            
            # 🎯 安全检查：确保所有必要的状态变量都存在且形状正确
            if (not hasattr(self, 'dyn_obs_state') or self.dyn_obs_state is None or
                not hasattr(self, 'dyn_obs_goal') or self.dyn_obs_goal is None or
                not hasattr(self, 'dyn_obs_origin') or self.dyn_obs_origin is None or
                not hasattr(self, 'dyn_obs_vel') or self.dyn_obs_vel is None or
                not hasattr(self, 'dyn_obs_vel_norm') or self.dyn_obs_vel_norm is None):
                print(f"[Warning] Dynamic obstacle state variables not initialized properly")
                return
            
            # 验证形状
            if (self.dyn_obs_state.shape[0] != total_dyn_obs or
                self.dyn_obs_goal.shape[0] != total_dyn_obs or
                self.dyn_obs_origin.shape[0] != total_dyn_obs or
                self.dyn_obs_vel.shape[0] != total_dyn_obs or
                self.dyn_obs_vel_norm.shape[0] != total_dyn_obs):
                print(f"[Warning] Dynamic obstacle state shape mismatch. Expected {total_dyn_obs}, got: "
                      f"state={self.dyn_obs_state.shape[0] if self.dyn_obs_state is not None else None}, "
                      f"goal={self.dyn_obs_goal.shape[0] if self.dyn_obs_goal is not None else None}, "
                      f"vel={self.dyn_obs_vel.shape[0] if self.dyn_obs_vel is not None else None}")
                return
            
            # Step 1: 检查是否需要新目标（到达当前目标时生成新目标）
            if self.dyn_obs_step_count != 0:
                dyn_obs_goal_dist = torch.norm(
                    self.dyn_obs_state[:, :3] - self.dyn_obs_goal, dim=1
                )
                dyn_obs_new_goal_mask = dyn_obs_goal_dist < self.dynamic_obstacle_goal_threshold
            else:
                dyn_obs_new_goal_mask = torch.ones(total_dyn_obs, dtype=torch.bool, device=self.device)
            
            # 采样新目标（在局部范围内随机，相对于机械臂末端位置）
            num_new_goal = dyn_obs_new_goal_mask.sum().item()
            if num_new_goal > 0:
                # 🎯 修复：动态障碍物应该围绕机械臂末端移动，而不是固定的原点
                # 获取需要新目标的障碍物的环境索引
                new_goal_indices = torch.where(dyn_obs_new_goal_mask)[0]  # [num_new_goal]
                env_indices_for_new_goals = new_goal_indices // self.num_dynamic_obstacles  # 每个障碍物对应的环境索引
                
                # 确保 ee_pos 已更新且有效
                if not hasattr(self, 'ee_pos') or self.ee_pos is None:
                    # 如果 ee_pos 不可用，使用原点（降级处理）
                    ee_pos_for_goals = self.dyn_obs_origin[new_goal_indices]
                else:
                    # 获取每个障碍物对应的机械臂末端位置
                    # env_indices_for_new_goals 是 [num_new_goal]，每个元素是该障碍物所属的环境索引
                    ee_pos_for_goals = self.ee_pos[env_indices_for_new_goals]  # [num_new_goal, 3]
                
                # 在局部范围内随机采样目标位置（相对于机械臂末端）
                sample_x = -self.dynamic_obstacle_local_range[0] + \
                    2.0 * self.dynamic_obstacle_local_range[0] * \
                    torch.rand(num_new_goal, 1, device=self.device)
                sample_y = -self.dynamic_obstacle_local_range[1] + \
                    2.0 * self.dynamic_obstacle_local_range[1] * \
                    torch.rand(num_new_goal, 1, device=self.device)
                sample_z = -self.dynamic_obstacle_local_range[2] + \
                    2.0 * self.dynamic_obstacle_local_range[2] * \
                    torch.rand(num_new_goal, 1, device=self.device)
                sample_goal_local = torch.cat([sample_x, sample_y, sample_z], dim=1)  # [num_new_goal, 3]
                
                # 🎯 修复：基于机械臂末端位置生成目标，而不是固定的原点
                # 确保索引安全
                if (new_goal_indices.shape[0] == num_new_goal and
                    sample_goal_local.shape[0] == num_new_goal and
                    ee_pos_for_goals.shape[0] == num_new_goal and
                    new_goal_indices.max().item() < self.dyn_obs_goal.shape[0]):
                    self.dyn_obs_goal[new_goal_indices] = \
                        ee_pos_for_goals + sample_goal_local
                
                # 限制在工作空间内
                self.dyn_obs_goal[:, 0] = torch.clamp(
                    self.dyn_obs_goal[:, 0], 
                    -self.workspace_radius * 0.8, 
                    self.workspace_radius * 0.8
                )
                self.dyn_obs_goal[:, 1] = torch.clamp(
                    self.dyn_obs_goal[:, 1],
                    -self.workspace_radius * 0.8,
                    self.workspace_radius * 0.8
                )
                self.dyn_obs_goal[:, 2] = torch.clamp(
                    self.dyn_obs_goal[:, 2],
                    self.workspace_z[0] + 0.1,
                    self.workspace_z[1] - 0.1
                )
            
            # Step 2: 更新速度（每N秒定期更新，或者如果速度无效则立即更新）
            sim_dt = 1.0 / 60.0  # Isaac Gym的dt
            vel_update_steps = int(self.dynamic_obstacle_vel_update_interval / sim_dt)
            
            # 🎯 修复：检查当前速度是否有效（每次调用都检查）
            vel_norm_check = torch.norm(self.dyn_obs_vel, dim=1, keepdim=True)  # [total_dyn_obs, 1]
            min_valid_vel = self.dynamic_obstacle_vel_range[0] * 0.5  # 最小有效速度
            vel_norm_1d = vel_norm_check.squeeze(-1)  # [total_dyn_obs]，更安全的squeeze
            if vel_norm_1d.ndim > 1:
                vel_norm_1d = vel_norm_1d.flatten()
            invalid_vel_mask = (vel_norm_1d < min_valid_vel)  # [total_dyn_obs]
            
            # 如果需要定期更新或者有无效速度，则更新速度
            should_update_vel = (self.dyn_obs_step_count == 0 or 
                                self.dyn_obs_step_count % vel_update_steps == 0 or 
                                invalid_vel_mask.any())
            
            if should_update_vel:
                # 🎯 修复：确保所有障碍物都有速度（避免有的障碍物不动）
                # 随机生成速度大小（确保每个障碍物都有非零速度）
                self.dyn_obs_vel_norm = self.dynamic_obstacle_vel_range[0] + \
                    (self.dynamic_obstacle_vel_range[1] - self.dynamic_obstacle_vel_range[0]) * \
                    torch.rand(total_dyn_obs, 1, device=self.device)
                
                # 确保速度大小不小于最小值的80%（避免速度过小）
                min_vel = self.dynamic_obstacle_vel_range[0] * 0.8
                self.dyn_obs_vel_norm = torch.clamp(self.dyn_obs_vel_norm, min=min_vel, max=None)
                
                # 计算朝向目标的方向
                direction = self.dyn_obs_goal - self.dyn_obs_state[:, :3]
                direction_norm = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
                
                # 🎯 修复：如果方向向量太小（接近目标），为这些障碍物生成新目标
                # 确保 direction_norm 是一维的 [total_dyn_obs] 用于比较
                direction_norm_1d = direction_norm.squeeze(-1)  # [total_dyn_obs]，更安全的squeeze
                if direction_norm_1d.ndim > 1:
                    direction_norm_1d = direction_norm_1d.flatten()
                too_close_mask = direction_norm_1d < self.dynamic_obstacle_goal_threshold * 0.5
                if too_close_mask.any():
                    # 为太接近目标的障碍物生成新目标（相对于机械臂末端）
                    num_too_close = too_close_mask.sum().item()
                    if num_too_close > 0:
                        # 获取需要新目标的障碍物的环境索引
                        too_close_indices = torch.where(too_close_mask)[0]  # [num_too_close]
                        env_indices_for_too_close = too_close_indices // self.num_dynamic_obstacles
                        
                        # 获取每个障碍物对应的机械臂末端位置
                        if not hasattr(self, 'ee_pos') or self.ee_pos is None:
                            ee_pos_for_too_close = self.dyn_obs_origin[too_close_indices]
                        else:
                            ee_pos_for_too_close = self.ee_pos[env_indices_for_too_close]  # [num_too_close, 3]
                        
                        sample_x = -self.dynamic_obstacle_local_range[0] + \
                            2.0 * self.dynamic_obstacle_local_range[0] * \
                            torch.rand(num_too_close, 1, device=self.device)
                        sample_y = -self.dynamic_obstacle_local_range[1] + \
                            2.0 * self.dynamic_obstacle_local_range[1] * \
                            torch.rand(num_too_close, 1, device=self.device)
                        sample_z = -self.dynamic_obstacle_local_range[2] + \
                            2.0 * self.dynamic_obstacle_local_range[2] * \
                            torch.rand(num_too_close, 1, device=self.device)
                        sample_goal_local = torch.cat([sample_x, sample_y, sample_z], dim=1)
                        
                        # 🎯 修复：基于机械臂末端位置生成目标，而不是固定的原点
                        if (too_close_indices.shape[0] == num_too_close and 
                            sample_goal_local.shape[0] == num_too_close and
                            ee_pos_for_too_close.shape[0] == num_too_close and
                            too_close_indices.max().item() < self.dyn_obs_goal.shape[0]):
                            self.dyn_obs_goal[too_close_indices] = \
                                ee_pos_for_too_close + sample_goal_local
                    # 重新计算方向
                    direction = self.dyn_obs_goal - self.dyn_obs_state[:, :3]
                    direction_norm = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
                
                # 设置速度（朝向目标）
                # 🎯 修复：确保维度匹配，避免广播错误
                direction_normalized = direction / direction_norm  # [total_dyn_obs, 3]
                if self.dyn_obs_vel_norm.shape == (total_dyn_obs, 1):
                    self.dyn_obs_vel = self.dyn_obs_vel_norm * direction_normalized
                else:
                    # 降级处理：如果形状不匹配，手动扩展
                    vel_norm_expanded = self.dyn_obs_vel_norm.expand(total_dyn_obs, 1).squeeze(-1).unsqueeze(-1)
                    self.dyn_obs_vel = vel_norm_expanded * direction_normalized
                
                # 🎯 修复：再次确保所有障碍物的速度都是有效的（检查并修复零速度）
                vel_norm_check_after = torch.norm(self.dyn_obs_vel, dim=1, keepdim=True)  # [total_dyn_obs, 1]
                vel_norm_1d = vel_norm_check_after.squeeze(-1)  # [total_dyn_obs]，更安全的squeeze
                if vel_norm_1d.ndim > 1:
                    vel_norm_1d = vel_norm_1d.flatten()
                zero_vel_mask = vel_norm_1d < 1e-6  # [total_dyn_obs]
                if zero_vel_mask.any():
                    # 如果有零速度的障碍物，为其设置随机方向的速度
                    num_zero_vel = zero_vel_mask.sum().item()
                    if num_zero_vel > 0:
                        # 随机生成单位方向向量
                        random_direction = torch.randn(num_zero_vel, 3, device=self.device)
                        random_direction = random_direction / (torch.norm(random_direction, dim=1, keepdim=True) + 1e-6)
                        # 确保索引安全
                        if (zero_vel_mask.shape[0] == self.dyn_obs_vel.shape[0] and 
                            self.dyn_obs_vel_norm.shape[0] == self.dyn_obs_vel.shape[0] and
                            random_direction.shape[0] == num_zero_vel and
                            zero_vel_mask.sum().item() == num_zero_vel):
                            # 使用更安全的方式：先提取速度大小，再乘法
                            vel_norm_selected = self.dyn_obs_vel_norm[zero_vel_mask].squeeze(-1)  # [num_zero_vel]
                            if vel_norm_selected.ndim == 1:
                                # 扩展维度以匹配 random_direction [num_zero_vel, 3]
                                vel_norm_expanded = vel_norm_selected.unsqueeze(-1)  # [num_zero_vel, 1]
                                self.dyn_obs_vel[zero_vel_mask] = vel_norm_expanded * random_direction
                            else:
                                # 降级处理
                                self.dyn_obs_vel[zero_vel_mask] = self.dyn_obs_vel_norm[zero_vel_mask].squeeze(-1).unsqueeze(-1) * random_direction
            
            # Step 3: 更新位置
            # 🎯 修复：确保速度不为零，并且位置真的在更新
            # 检查速度是否有效
            vel_norm = torch.norm(self.dyn_obs_vel, dim=1)  # [total_dyn_obs]
            if vel_norm.max().item() < 1e-6:
                # 生成随机速度
                random_vel = torch.randn(total_dyn_obs, 3, device=self.device)
                random_vel = random_vel / (torch.norm(random_vel, dim=1, keepdim=True) + 1e-6)
                vel_magnitude = self.dynamic_obstacle_vel_range[0] + (
                    self.dynamic_obstacle_vel_range[1] - self.dynamic_obstacle_vel_range[0]
                ) * torch.rand(total_dyn_obs, 1, device=self.device)
                self.dyn_obs_vel = random_vel * vel_magnitude
                if self.dyn_obs_step_count % 100 == 0:
                    print(f"[Warning] All dynamic obstacle velocities were near zero! Generated new random velocities.")
            
            # 记录更新前的位置（用于验证）
            old_positions = self.dyn_obs_state[:, :3].clone()
            
            # 更新位置
            self.dyn_obs_state[:, :3] += self.dyn_obs_vel * sim_dt
            
            # 🎯 验证：检查位置是否真的改变了（每100步检查一次）
            if self.dyn_obs_step_count % 100 == 0:
                position_diff = torch.norm(self.dyn_obs_state[:, :3] - old_positions, dim=1)
                max_diff = position_diff.max().item()
                avg_diff = position_diff.mean().item()
                if max_diff < 1e-6:
                    print(f"[Error] Dynamic obstacle positions did not change! max_diff={max_diff:.6f}, "
                          f"vel_norm_max={torch.norm(self.dyn_obs_vel, dim=1).max().item():.6f}, "
                          f"sim_dt={sim_dt}")
                elif self.dyn_obs_step_count % 500 == 0:  # 每500步打印一次详细信息
                    print(f"[Dynamic Obstacles] Position update OK: max_diff={max_diff:.6f}m, avg_diff={avg_diff:.6f}m, "
                          f"vel_norm_max={torch.norm(self.dyn_obs_vel, dim=1).max().item():.6f}m/s")
            
            # 🎯 修复：检查动态障碍物是否离机械臂末端太远，如果太远就在机械臂附近重新生成
            # 确保 ee_pos 已更新（在执行 move_dynamic_obstacles 时应该已经更新）
            if hasattr(self, 'ee_pos') and self.ee_pos is not None and self.ee_pos.shape[0] == self.num_envs:
                # 计算每个障碍物到对应环境的机械臂末端的距离
                for env_idx in range(self.num_envs):
                    env_start_idx = env_idx * self.num_dynamic_obstacles
                    env_end_idx = (env_idx + 1) * self.num_dynamic_obstacles
                    
                    if env_end_idx <= self.dyn_obs_state.shape[0]:
                        # 获取该环境的所有动态障碍物位置
                        dyn_obs_positions = self.dyn_obs_state[env_start_idx:env_end_idx, :3]  # [num_dyn_obs, 3]
                        ee_pos_current = self.ee_pos[env_idx]  # [3]
                        
                        # 检查 ee_pos 是否有效
                        if torch.isnan(ee_pos_current).any() or torch.isinf(ee_pos_current).any():
                            continue
                        
                        # 计算每个障碍物到机械臂末端的距离
                        distances = torch.norm(dyn_obs_positions - ee_pos_current.unsqueeze(0), dim=1)  # [num_dyn_obs]
                        
                        # 🎯 修复：使用更合理的阈值，确保障碍物始终在机械臂附近
                        # 使用 local_range 的最大维度作为阈值（降低系数使更容易触发）
                        # 阈值 = max(local_range) * 1.2，例如 [0.4, 0.4, 0.3] -> 0.4 * 1.2 = 0.48m
                        max_range = max(self.dynamic_obstacle_local_range) * 1.2  # 最大维度 * 1.2作为阈值（更容易触发重新生成）
                        max_range_tensor = torch.tensor(max_range, device=self.device, dtype=torch.float32)
                        
                        too_far_mask = distances > max_range_tensor  # [num_dyn_obs]
                        num_too_far = too_far_mask.sum().item()
                        
                        # 在机械臂末端附近的局部范围内重新生成这些障碍物的位置
                        if num_too_far > 0:
                            too_far_indices = torch.where(too_far_mask)[0]  # 在环境内的局部索引
                            global_too_far_indices = env_start_idx + too_far_indices  # 全局索引
                            
                            # 🎯 批量生成新位置（相对于机械臂末端，使用张量操作提高效率）
                            num_too_far = too_far_indices.shape[0]
                            sample_x = ee_pos_current[0] + (
                                -self.dynamic_obstacle_local_range[0] + 
                                2.0 * self.dynamic_obstacle_local_range[0] * torch.rand(num_too_far, device=self.device)
                            )
                            sample_y = ee_pos_current[1] + (
                                -self.dynamic_obstacle_local_range[1] + 
                                2.0 * self.dynamic_obstacle_local_range[1] * torch.rand(num_too_far, device=self.device)
                            )
                            sample_z = ee_pos_current[2] + (
                                -self.dynamic_obstacle_local_range[2] + 
                                2.0 * self.dynamic_obstacle_local_range[2] * torch.rand(num_too_far, device=self.device)
                            )
                            new_positions = torch.stack([sample_x, sample_y, sample_z], dim=1)  # [num_too_far, 3]
                            
                            # 批量更新障碍物位置
                            if (global_too_far_indices.max().item() < self.dyn_obs_state.shape[0] and
                                global_too_far_indices.shape[0] == num_too_far):
                                self.dyn_obs_state[global_too_far_indices, :3] = new_positions
                                
                                # 批量更新原点
                                if global_too_far_indices.max().item() < self.dyn_obs_origin.shape[0]:
                                    self.dyn_obs_origin[global_too_far_indices] = new_positions
                                
                                # 批量生成新目标（在附近随机位置）
                                if global_too_far_indices.max().item() < self.dyn_obs_goal.shape[0]:
                                    goal_offset_x = -self.dynamic_obstacle_local_range[0] + 2.0 * self.dynamic_obstacle_local_range[0] * torch.rand(num_too_far, device=self.device)
                                    goal_offset_y = -self.dynamic_obstacle_local_range[1] + 2.0 * self.dynamic_obstacle_local_range[1] * torch.rand(num_too_far, device=self.device)
                                    goal_offset_z = -self.dynamic_obstacle_local_range[2] + 2.0 * self.dynamic_obstacle_local_range[2] * torch.rand(num_too_far, device=self.device)
                                    goal_offsets = torch.stack([goal_offset_x, goal_offset_y, goal_offset_z], dim=1)  # [num_too_far, 3]
                                    self.dyn_obs_goal[global_too_far_indices] = new_positions + goal_offsets
                                
                                # 批量重置速度，使其朝向新目标
                                if (global_too_far_indices.max().item() < self.dyn_obs_goal.shape[0] and
                                    global_too_far_indices.max().item() < self.dyn_obs_vel.shape[0] and
                                    global_too_far_indices.max().item() < self.dyn_obs_vel_norm.shape[0]):
                                    directions = self.dyn_obs_goal[global_too_far_indices] - new_positions  # [num_too_far, 3]
                                    direction_norms = torch.norm(directions, dim=1, keepdim=True).clamp(min=1e-6)  # [num_too_far, 1]
                                    directions_normalized = directions / direction_norms  # [num_too_far, 3]
                                    
                                    vel_norms = self.dyn_obs_vel_norm[global_too_far_indices]  # [num_too_far, 1]
                                    if vel_norms.shape[1] == 1:
                                        self.dyn_obs_vel[global_too_far_indices] = directions_normalized * vel_norms
                                    else:
                                        vel_norms_expanded = vel_norms.squeeze(-1).unsqueeze(-1)  # [num_too_far, 1]
                                        self.dyn_obs_vel[global_too_far_indices] = directions_normalized * vel_norms_expanded
            
            # 确保在工作空间内
            self.dyn_obs_state[:, 0] = torch.clamp(
                self.dyn_obs_state[:, 0],
                -self.workspace_radius * 0.85,
                self.workspace_radius * 0.85
            )
            self.dyn_obs_state[:, 1] = torch.clamp(
                self.dyn_obs_state[:, 1],
                -self.workspace_radius * 0.85,
                self.workspace_radius * 0.85
            )
            self.dyn_obs_state[:, 2] = torch.clamp(
                self.dyn_obs_state[:, 2],
                self.workspace_z[0] + 0.05,
                self.workspace_z[1] - 0.05
            )
            
            # Step 4: 只更新逻辑位置，不更新到仿真
            # 🎯 关键修复：移除这里的 _update_dynamic_obstacle_positions_to_sim() 调用
            # 因为 step() 中会在物理模拟后统一更新位置，避免重复调用
            # 位置更新将在 step() 中的物理模拟后统一进行
            # 注意：不再需要 try-except，因为这里不再调用更新函数
            
            self.dyn_obs_step_count += 1
            
        except Exception as e:
            print(f"[Error] _move_dynamic_obstacles() failed: {e}")
            import traceback
            traceback.print_exc()
            # 🎯 关键修复：异常时清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            # 即使出错也继续，避免完全停止训练

    def _update_dynamic_obstacle_positions_to_sim_quick(self):
        """快速更新动态障碍物位置（用于仿真子步骤中，不刷新状态）"""
        if self.num_dynamic_obstacles == 0 or self.dyn_obs_state is None or len(self.dynamic_obstacle_handles) == 0:
            return
        
        # 🎯 快速更新：直接使用之前计算的 dyn_obs_state，不刷新 root_states
        # 这样可以避免在每个子步骤中都刷新状态，提高效率
        try:
            self.actors_per_env = 2 + self.num_obstacles + self.num_dynamic_obstacles
            
            # 直接修改 root_states（假设它已经被正确初始化）
            for i in range(len(self.dynamic_obstacle_handles)):
                env_idx = i // self.num_dynamic_obstacles
                dyn_idx = i % self.num_dynamic_obstacles
                
                if env_idx >= self.num_envs or i >= self.dyn_obs_state.shape[0]:
                    continue
                
                obs_root_idx = env_idx * self.actors_per_env + 2 + self.num_obstacles + dyn_idx
                if obs_root_idx >= self.root_states.shape[0]:
                    continue
                
                # 创建新状态
                new_state = torch.zeros(13, dtype=torch.float32, device=self.device)
                new_state[0:3] = self.dyn_obs_state[i, :3]
                new_state[3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
                if self.dyn_obs_vel is not None and i < self.dyn_obs_vel.shape[0]:
                    vel = self.dyn_obs_vel[i]
                    if not (torch.isnan(vel).any() or torch.isinf(vel).any()):
                        new_state[7:10] = vel
                new_state[10:13] = 0.0
                
                self.root_states[obs_root_idx] = new_state
            
            # 使用完整的 tensor 更新
            # [patched] DISABLED full-buffer set for dynamic obstacles
        except Exception as e:
            # 如果快速更新失败，静默失败（会在完整更新中处理）
            pass

    def _update_dynamic_obstacle_positions_to_sim(self, new_pos_xyz=None, new_vel_xyz=None, debug_name="dyn"):
        """
        将动态障碍物位置/速度写回仿真（只改对应索引，不全量覆盖）。
        
        必须满足：
        - self.dyn_root_indices : torch.int32 (CPU), contiguous，长度=动态障碍物总数
        - new_pos_xyz           : [num_dyn, 3]，CPU contiguous（如果为None，使用self.dyn_obs_state）
        - new_vel_xyz（可选）   : [num_dyn, 3]，CPU contiguous（如果为None，使用self.dyn_obs_vel）
        """
        if self.num_dynamic_obstacles == 0 or self.dyn_obs_state is None or len(self.dynamic_obstacle_handles) == 0:
            return
        
        gym = self.gym
        sim = self.sim
        
        # 0) 准备索引（CPU int32 contiguous）
        # 优先使用 self.dyn_root_indices，如果没有则使用 self.dynamic_root_indices
        if hasattr(self, 'dyn_root_indices') and self.dyn_root_indices.numel() > 0:
            dyn_indices = self.dyn_root_indices
        elif hasattr(self, 'dynamic_root_indices') and self.dynamic_root_indices.numel() > 0:
            dyn_indices = self.dynamic_root_indices
        else:
            # 如果没有缓存的索引，计算它们
            self.actors_per_env = 2 + self.num_obstacles + self.num_dynamic_obstacles
            all_dyn_indices = []
            for i in range(len(self.dynamic_obstacle_handles)):
                env_idx = i // self.num_dynamic_obstacles
                dyn_idx = i % self.num_dynamic_obstacles
                obs_root_idx = env_idx * self.actors_per_env + 2 + self.num_obstacles + dyn_idx
                all_dyn_indices.append(obs_root_idx)
            dyn_indices = torch.tensor(all_dyn_indices, dtype=torch.int64, device='cpu')
            self.dyn_root_indices = dyn_indices  # 缓存起来
        
        if dyn_indices.dtype != torch.int32 or dyn_indices.device.type != "cpu" or not dyn_indices.is_contiguous():
            dyn_indices = dyn_indices.to(dtype=torch.int32, device="cpu").contiguous()
        
        assert dyn_indices.ndim == 1, f"[{debug_name}] dyn_indices must be 1-D, got {dyn_indices.shape}"
        num_dyn = dyn_indices.numel()
        if num_dyn == 0:
            return
        
        # 1) 准备 new states（CPU contiguous，13通道齐全）
        #    root state layout: [pos(3), rot(4), lin_vel(3), ang_vel(3)]
        states = torch.zeros((num_dyn, 13), dtype=torch.float32, device="cpu")
        
        # 位置
        if new_pos_xyz is not None:
            if new_pos_xyz.device.type != "cpu":
                pos_cpu = new_pos_xyz.to("cpu", non_blocking=True)
            else:
                pos_cpu = new_pos_xyz
            # 确保形状匹配
            if pos_cpu.shape[0] != num_dyn:
                print(f"[{debug_name}] WARNING: new_pos_xyz shape {pos_cpu.shape} != num_dyn {num_dyn}, truncating/padding")
                if pos_cpu.shape[0] > num_dyn:
                    pos_cpu = pos_cpu[:num_dyn]
                else:
                    padding = torch.zeros((num_dyn - pos_cpu.shape[0], 3), dtype=torch.float32, device='cpu')
                    pos_cpu = torch.cat([pos_cpu, padding], dim=0)
        else:
            # 使用 self.dyn_obs_state
            if self.dyn_obs_state.shape[0] < num_dyn:
                print(f"[{debug_name}] WARNING: dyn_obs_state shape {self.dyn_obs_state.shape[0]} < num_dyn {num_dyn}")
                pos_cpu = torch.zeros((num_dyn, 3), dtype=torch.float32, device='cpu')
            else:
                pos_cpu = self.dyn_obs_state[:num_dyn, 0:3].to("cpu", non_blocking=True)
        
        states[:, 0:3] = pos_cpu.contiguous()
        
        # 姿态：不给就默认单位四元数
        states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32).repeat(num_dyn, 1)
        
        # 线速度
        if new_vel_xyz is not None:
            vel = new_vel_xyz.to("cpu", non_blocking=True).contiguous()
            states[:, 7:10] = vel
        elif self.dyn_obs_vel is not None and self.dyn_obs_vel.shape[0] >= num_dyn:
            vel = self.dyn_obs_vel[:num_dyn].to("cpu", non_blocking=True).contiguous()
            states[:, 7:10] = vel
        else:
            states[:, 7:10] = 0.0
        
        # 角速度
        states[:, 10:13] = 0.0
        
        # 2) 写入（只写索引，不全量覆盖）
        # 🎯 关键：先更新 root_states 的对应行，然后传递完整的 root_states
        indices_long = dyn_indices.to(dtype=torch.long)
        self.root_states[indices_long] = states
        
        # 现在传递完整的 root_states 给 API
        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(self.root_states),  # ✅ 完整的 root_states
            gymtorch.unwrap_tensor(dyn_indices),        # ✅ 只给索引
            num_dyn
        )
        
        # 🎯 关键：等待 GPU 同步（如果使用 GPU PhysX）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 3) 立刻 refresh，保证下面读的是仿真里的最新数据
        gym.refresh_actor_root_state_tensor(sim)
        
        # 4) 读回做一次校验（debug）
        # 注意：self.root_states 一定要是 acquire 的"活指针"，不要自己 new tensor 覆盖它！
        rs = self.root_states  # (total_actors, 13)
        assert rs.device.type == "cpu", f"[{debug_name}] root_states must be CPU when use_gpu_pipeline=False"
        readback = rs[dyn_indices.long(), 0:3].clone()
        max_diff = (readback - pos_cpu).abs().max().item()
        if max_diff > 1e-5:
            print(f"[{debug_name}] WARNING: after refresh, readback!=target, max_diff={max_diff:.6f}")
            # 继续放开跑也行，但你想要严格的话可以抛异常
            # raise RuntimeError("root_states mismatch after set_actor_root_state_tensor_indexed")
        
        # 5) 自检：若外部有"全量覆盖"调用，会在下一次 step 后把我们写的值抹掉
        #    这里设置一个易识别的"标记位"，下一帧验证它是否仍在
        #    原理：我们利用 lin_vel.x（第7通道）存个小签名，正常物理不会在一帧内刚好改成相同 magic
        # 🎯 注意：这个检查在同一个函数内立即进行，如果 refresh 后读取到旧值，
        #    说明 API 调用可能没有生效，或者 Isaac Gym 内部状态还没有更新
        magic = 1234.5678
        states_magic = states.clone()
        states_magic[:, 7] = magic
        # 先更新 root_states
        indices_long = dyn_indices.to(dtype=torch.long)
        self.root_states[indices_long] = states_magic
        # 然后传递完整的 root_states
        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(self.root_states),  # ✅ 完整的 root_states
            gymtorch.unwrap_tensor(dyn_indices),      # ✅ 只给索引
            num_dyn
        )
        
        # 🎯 关键：等待 GPU 同步（如果使用 GPU PhysX）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 🎯 关键：先检查 root_states 中的值（在 refresh 之前）
        sig_before_refresh = self.root_states[indices_long, 7].clone()
        if (sig_before_refresh - magic).abs().max().item() > 1e-4:
            print(f"[{debug_name}] WARNING: magic signature not set in root_states before refresh!")
            print(f"        Expected magic={magic:.4f}, got sig={sig_before_refresh.tolist()}")
            print(f"        This suggests root_states binding may be broken or write failed")
        
        # 现在 refresh 并检查
        gym.refresh_actor_root_state_tensor(sim)
        sig_after_refresh = self.root_states[indices_long, 7].clone()
        if (sig_after_refresh - magic).abs().max().item() > 1e-4:
            print("[FATAL] root_state was overwritten by a subsequent full-tensor call in the same frame!")
            print("        请全局搜索并删除任何 set_actor_root_state_tensor(self.sim, ...) 的调用；只保留 indexed 写法。")
            print(f"        Expected magic={magic:.4f}, got sig={sig_after_refresh.tolist()}")
            print(f"        Before refresh: {sig_before_refresh.tolist()}")
            print(f"        After refresh: {sig_after_refresh.tolist()}")
            print(f"        This suggests that either:")
            print(f"        1. Another function called set_actor_root_state_tensor (non-indexed)")
            print(f"        2. _apply_root_states_indexed was called after this update")
            print(f"        3. root_states was re-assigned (breaking the binding)")
            print(f"        4. API call failed silently (check Isaac Gym error logs)")

    def _check_occlusion(self, cam_pos: torch.Tensor, target_pos: torch.Tensor, env_idx: int = 0) -> bool:
        """
        检查从相机到目标之间是否有障碍物遮挡
        使用射线-球体相交检测
        
        Args:
            cam_pos: [3] 相机位置（世界坐标）
            target_pos: [3] 目标位置（世界坐标）
            env_idx: 环境索引，用于确定检查哪些障碍物
        
        Returns:
            bool: 如果被遮挡返回True，否则返回False
        """
        if len(self.obstacle_handles) == 0:
            return False
        
        # 🎯 修复：只检查该环境对应的障碍物
        # 环境 env_idx 的障碍物索引范围：[env_idx * num_obstacles, (env_idx + 1) * num_obstacles)
        obstacle_start_idx = env_idx * self.num_obstacles
        obstacle_end_idx = min((env_idx + 1) * self.num_obstacles, len(self.obstacle_handles))
        
        # 射线方向
        ray_dir = target_pos - cam_pos
        ray_len = torch.norm(ray_dir).item()
        if ray_len < 1e-6:
            return False
        ray_dir = ray_dir / ray_len
        
        # 对每个障碍物进行射线-球体相交检测（只检查该环境的障碍物）
        for i in range(obstacle_start_idx, obstacle_end_idx):
            if i >= len(self.obstacle_positions) or i >= len(self.obstacle_radii):
                continue
                
            obs_pos = self.obstacle_positions[i]
            obs_radius = self.obstacle_radii[i].item()
            
            # 从相机位置到障碍物中心的向量
            to_obs = obs_pos - cam_pos
            # 投影到射线方向
            proj_len = torch.dot(to_obs, ray_dir).item()
            
            # 如果投影在射线外，跳过
            if proj_len < 0 or proj_len > ray_len:
                continue
            
            # 最近点到障碍物中心的距离
            closest_point = cam_pos + ray_dir * proj_len
            dist_to_center = torch.norm(closest_point - obs_pos).item()
            
            # 如果距离小于半径，说明射线与球体相交（遮挡）
            if dist_to_center < obs_radius:
                return True
        
        return False
    
    def _check_single_env_aruco_visibility(self, env_idx: int, cam_pos: torch.Tensor, cam_R: torch.Tensor, 
                                          rel: torch.Tensor, dist: torch.Tensor) -> bool:
        """
        检查单个环境的ArUco目标是否在视野内（辅助函数）
        
        Args:
            env_idx: 环境索引
            cam_pos: [3] 相机位置（世界坐标）
            cam_R: [3, 3] 相机旋转矩阵
            rel: [3] 从相机到目标的向量
            dist: 标量，距离
        
        Returns:
            bool: 如果目标在视野内且未被遮挡返回True
        """
        try:
            # 🎯 修复：添加输入验证
            if cam_pos.shape != (3,) or cam_R.shape != (3, 3) or rel.shape != (3,):
                return False
            if torch.isnan(cam_pos).any() or torch.isnan(cam_R).any() or torch.isnan(rel).any():
                return False
            if torch.isinf(cam_pos).any() or torch.isinf(cam_R).any() or torch.isinf(rel).any():
                return False
            
            # 🎯 修复：保持dist为张量，避免.item()
            if not torch.isfinite(dist) or dist <= 1e-6:
                return False
            
            # 相机朝 +X（局部坐标系）
            forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            forward_world = cam_R @ forward_local  # 转换到世界坐标系
            
            # 计算角度（使用纯张量操作）
            rel_normalized = rel / dist
            cos_angle = torch.clamp(torch.dot(rel_normalized, forward_world), -1.0, 1.0)
            if not torch.isfinite(cos_angle):
                return False
            # 确保cos_angle在有效范围内
            cos_angle_clamped = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
            angle = torch.acos(cos_angle_clamped)
            fov_half_rad = torch.tensor(self.fov / 2, device=self.device, dtype=torch.float32)
            in_fov = angle <= fov_half_rad
            # 对于标量布尔张量，使用 bool() 转换
            in_fov_bool = bool(in_fov)
            
            # 像素投影（使用正确的相机坐标系，纯张量操作）
            rel_cam = cam_R.T @ rel  # 转换到相机坐标系
            x_cam = rel_cam[0]
            y_cam = rel_cam[1]
            z_cam = rel_cam[2]
            
            in_img = False
            if bool(x_cam > 1e-6):  # 目标在相机前方
                if not hasattr(self, 'intrinsics') or self.intrinsics is None:
                    return False
                fx, fy, cx, cy = self.intrinsics['fx'], self.intrinsics['fy'], self.intrinsics['cx'], self.intrinsics['cy']
                # 使用张量操作计算u和v
                u_tensor = cx + fx * (y_cam / x_cam)
                v_tensor = cy + fy * (z_cam / x_cam)
                # 转换为整数索引（需要标量，但这里只在判断时使用）
                u = int(float(u_tensor.cpu().numpy())) if u_tensor.numel() == 1 else 0
                v = int(float(v_tensor.cpu().numpy())) if v_tensor.numel() == 1 else 0
                in_img = (0 <= u < self.img_w) and (0 <= v < self.img_h)
            
            # 遮挡检测：只有在FOV和图像内时才检查遮挡
            if in_fov_bool and in_img:
                if env_idx >= self.target_pos.shape[0]:
                    return False
                target_pos = self.target_pos[env_idx]
                try:
                    is_occluded = self._check_occlusion(cam_pos, target_pos, env_idx=env_idx)
                    return bool(not is_occluded)
                except Exception:
                    return False
            
            return False
        except Exception as e:
            # 如果任何检查失败，返回False（不可见）
            return False
    
    def _update_aruco_detection(self):
        """
        批量更新所有环境的ArUco检测状态
        
        🎯 修复：完整支持多环境，每个环境独立检测，添加严格的边界检查
        """
        # 🎯 修复：添加异常处理，避免段错误
        try:
            # 验证必要的状态变量是否存在
            if not hasattr(self, 'target_pos') or self.target_pos is None:
                return
            if not hasattr(self, 'ee_pos') or self.ee_pos is None:
                return
            if not hasattr(self, 'ee_quat') or self.ee_quat is None:
                return
            if self.target_pos.shape[0] != self.num_envs:
                return
            if self.ee_pos.shape[0] != self.num_envs:
                return
            
            # 为每个环境单独检测目标是否在视野内
            in_view_per_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
            for env_idx in range(self.num_envs):
                try:
                    # 🎯 修复：严格检查索引有效性
                    if env_idx >= self.num_envs or env_idx < 0:
                        continue
                    if env_idx >= self.target_pos.shape[0]:
                        continue
                    
                    # 🎯 调试：检查输入数据
                    if env_idx >= self.ee_pos.shape[0]:
                        print(f"[Error] env_idx {env_idx} out of bounds for ee_pos shape {self.ee_pos.shape}")
                        continue
                    
                    # 🎯 调试：检查数据有效性
                    if torch.isnan(self.ee_pos[env_idx]).any():
                        print(f"[Warning] NaN in ee_pos for env {env_idx}")
                        continue
                    
                    # 🎯 修复：使用支持多环境的camera_pose，添加异常处理
                    # 先检查ee_pos和ee_quat是否可以安全访问
                    try:
                        # 在调用camera_pose之前，先验证输入数据
                        if env_idx >= self.ee_pos.shape[0] or env_idx >= self.ee_quat.shape[0]:
                            continue
                        
                        # 尝试创建一个测试访问，确保内存有效
                        test_ee_pos = self.ee_pos[env_idx]
                        test_ee_quat = self.ee_quat[env_idx]
                        if test_ee_pos.numel() != 3 or test_ee_quat.numel() != 4:
                            continue
                        
                        # 如果测试通过，再调用camera_pose
                        cam_T = self.camera_pose(env_idx)  # [4, 4]
                        if cam_T is None or cam_T.shape != (4, 4):
                            continue
                        cam_pos = cam_T[:3, 3].clone()  # [3] - 使用clone确保独立内存
                        cam_R = cam_T[:3, :3].clone()  # [3, 3] - 使用clone确保独立内存
                        
                        # 验证cam_pos和cam_R的有效性
                        if torch.isnan(cam_pos).any() or torch.isnan(cam_R).any():
                            continue
                        if torch.isinf(cam_pos).any() or torch.isinf(cam_R).any():
                            continue
                    except Exception as e:
                        # camera_pose失败，跳过该环境
                        continue
                    except RuntimeError as e:
                        # 捕获可能的C++底层错误
                        print(f"[Warning] RuntimeError in camera_pose for env {env_idx}: {e}")
                        continue
                    except SystemError as e:
                        # 捕获系统级错误
                        print(f"[Warning] SystemError in camera_pose for env {env_idx}: {e}")
                        continue
                    
                    target_pos = self.target_pos[env_idx]  # [3]
                    if torch.isnan(target_pos).any() or torch.isinf(target_pos).any():
                        continue
                    
                    rel = target_pos - cam_pos
                    dist_tensor = torch.norm(rel)  # 保持张量，避免.item()
                    
                    # 🎯 修复：使用张量操作进行判断
                    # 对于标量张量，直接进行布尔运算
                    dist_valid = (dist_tensor > 1e-6) & torch.isfinite(dist_tensor)
                    # 对于标量布尔张量，使用 bool() 或者检查值
                    if not bool(dist_valid):  # 如果距离太近或无效，跳过
                        in_view_per_env[env_idx] = False
                        continue
                    
                    # 调用辅助函数检查可见性
                    try:
                        in_view_per_env[env_idx] = self._check_single_env_aruco_visibility(
                            env_idx, cam_pos, cam_R, rel, dist_tensor
                        )
                    except Exception as e:
                        print(f"[Warning] _check_single_env_aruco_visibility failed for env {env_idx}: {e}")
                        in_view_per_env[env_idx] = False
                except Exception as e:
                    # 如果单个环境检测失败，标记为不可见
                    in_view_per_env[env_idx] = False
            
            # 批量更新view_counter（添加安全检查）
            try:
                if hasattr(self, 'target_view_counter') and self.target_view_counter is not None:
                    valid_mask = in_view_per_env & (self.target_view_counter < 1000)  # 防止溢出
                    reset_mask = ~in_view_per_env
                    self.target_view_counter[valid_mask] += 1
                    self.target_view_counter[reset_mask] = 0
                    
                    # 使用批量mask更新发现状态
                    if hasattr(self, 'target_confirm_steps') and hasattr(self, 'target_discovered'):
                        # 🎯 修复：使用纯张量操作进行比较，避免.item()
                        if isinstance(self.target_confirm_steps, torch.Tensor):
                            # 如果是张量，直接使用张量比较（PyTorch会自动广播）
                            confirm_mask = (self.target_view_counter >= self.target_confirm_steps)
                        else:
                            # 如果是标量，转换为张量
                            confirm_threshold_tensor = torch.tensor(self.target_confirm_steps, device=self.device, dtype=self.target_view_counter.dtype)
                            confirm_mask = (self.target_view_counter >= confirm_threshold_tensor)
                        
                        undiscovered_mask = ~self.target_discovered.bool()
                        to_set_mask = confirm_mask & undiscovered_mask
                        
                        if bool(to_set_mask.any()):
                            num_to_set = int(float(to_set_mask.sum().cpu().numpy())) if to_set_mask.sum().numel() == 1 else 0
                            self.target_discovered[to_set_mask] = True
            except Exception as e:
                print(f"[Warning] Failed to update view counter: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            # 如果整个检测过程失败，打印错误但不中断训练
            print(f"[Warning] _update_aruco_detection() failed: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _point_to_segment_distance(p, a, b):
        # 返回点 p 到线段 ab 的最小距离
        ap = p - a
        ab = b - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)
    
    def _check_success_conditions(self):
        """
        成功条件：末端执行器到达ArUco码目标（批量处理）
        
        核心思想：不仅仅是探索，要真正到达目标。需要末端执行器与ArUco码的直线距离小于0.1m。
        
        成功条件：
        1. 末端执行器到目标的欧氏距离 < 0.1m
        
        Returns:
            (success: torch.Tensor[num_envs], reason: List[str]): 每个环境的成功状态和原因
        """
        success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        success_reason = [None] * self.num_envs
        
        # 成功距离阈值（米）
        success_distance_threshold = 0.1  # 0.1m = 10cm
        
        # 批量计算所有环境的距离
        try:
            # 计算末端执行器到目标的距离
            ee_to_target = self.target_pos - self.ee_pos  # [num_envs, 3]
            distances = torch.norm(ee_to_target, dim=1)  # [num_envs]
            
            # 检查距离是否小于阈值
            success_mask = distances < success_distance_threshold
            
            # 为每个环境设置成功状态和原因
            for i in range(self.num_envs):
                try:
                    if success_mask[i].item():
                        success[i] = True
                        distance_value = distances[i].item()
                        success_reason[i] = f"成功到达：距离 {distance_value*100:.2f}cm < {success_distance_threshold*100:.1f}cm"
                    else:
                        distance_value = distances[i].item()
                        success_reason[i] = f"未到达：距离 {distance_value*100:.2f}cm >= {success_distance_threshold*100:.1f}cm"
                except Exception as e:
                    success_reason[i] = f"检查异常: {str(e)[:50]}"
                    continue
        except Exception as e:
            # 如果批量计算失败，为所有环境设置失败原因
            for i in range(self.num_envs):
                success_reason[i] = f"距离计算失败: {str(e)[:50]}"
        
        return success, success_reason
    
    def _check_safe_path_to_target(self, env_idx: int = 0):
        """
        检查是否存在安全路径到目标
        
        核心思想：只要存在一条从末端到目标的安全路径（不需要实际移动），
        就认为探索任务成功。这更符合实际探索任务的需求。
        
        Args:
            env_idx: 环境索引
        
        Returns:
            bool: 如果存在安全路径返回True，否则返回False
        """
        current_pos = self.ee_pos[env_idx]
        target_pos = self.target_pos[env_idx]
        
        # 首先检查起点和终点是否都在安全区域内
        if not self._is_position_safe(current_pos, env_idx):
            return False
        if not self._is_position_safe(target_pos, env_idx):
            return False
        
        # 方法1: 直线路径安全性检查（最简单直接）
        try:
            if self._check_straight_path_safety(current_pos, target_pos, env_idx):
                return True
        except:
            # 如果检查失败，继续尝试其他方法
            pass
        
        # 方法2: 关键航点路径检查（如果直线路径不安全，尝试绕过障碍物）
        try:
            if self._check_waypoint_path_safety(current_pos, target_pos, env_idx):
                return True
        except:
            # 如果检查失败，返回False
            pass
        
        # 🎯 如果两种方法都失败，说明无法找到安全路径
        return False
    
    def _check_straight_path_safety(self, start: torch.Tensor, end: torch.Tensor, env_idx: int = 0):
        """
        检查直线路径的安全性
        
        Args:
            start: [3] 路径起点（世界坐标，通常是末端位置）
            end: [3] 路径终点（世界坐标，通常是目标位置）
            env_idx: 环境索引
        
        Returns:
            bool: 如果路径安全返回True，否则返回False
        """
        # 采样路径上的点进行检查
        num_samples = 20  # 增加采样点数以提高准确性
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_point = start + t * (end - start)
            
            # 检查该点是否安全（不与障碍物碰撞）
            if not self._is_position_safe(sample_point, env_idx):
                return False
        
        # 🎯 注意：这里不检查遮挡，因为路径安全检查关注的是物理碰撞，
        # 而不是视觉遮挡。遮挡检查应该用于ArUco检测的可见性验证。
        return True
    
    def _check_waypoint_path_safety(self, start: torch.Tensor, end: torch.Tensor, env_idx: int = 0):
        """检查关键航点路径的安全性（简化版）"""
        # 生成1-2个中间航点，避开障碍物
        # 方法：在起点和终点之间生成航点，检查每个航段的安全性
        
        # 航点1：中点上方抬升
        mid_point = (start + end) / 2
        waypoint1 = mid_point.clone()
        waypoint1[2] += 0.2  # 抬升20cm
        
        # 检查起点 -> 航点1 -> 终点的路径
        if (self._check_straight_path_safety(start, waypoint1, env_idx) and 
            self._check_straight_path_safety(waypoint1, end, env_idx) and
            self._is_position_safe(waypoint1, env_idx)):
            return True
        
        # 如果抬升路径不安全，尝试其他方向
        waypoint2 = mid_point.clone()
        waypoint2[2] -= 0.1  # 下降10cm（但保持在工作空间内）
        if waypoint2[2].item() < self.workspace_z[0]:
            waypoint2[2] = torch.tensor(self.workspace_z[0] + 0.05, device=self.device)
        
        if (self._check_straight_path_safety(start, waypoint2, env_idx) and 
            self._check_straight_path_safety(waypoint2, end, env_idx) and
            self._is_position_safe(waypoint2, env_idx)):
            return True
        
        return False
    
    def _is_position_safe(self, position: torch.Tensor, env_idx: int = 0):
        """
        检查单个位置的安全性
        
        Args:
            position: [3] 待检查的位置（世界坐标）
            env_idx: 环境索引，用于确定检查哪些障碍物
        
        Returns:
            bool: 如果位置安全返回True，否则返回False
        """
        # 🎯 修复：只检查该环境对应的障碍物
        # 环境 env_idx 的障碍物索引范围：[env_idx * num_obstacles, (env_idx + 1) * num_obstacles)
        obstacle_start_idx = env_idx * self.num_obstacles
        obstacle_end_idx = min((env_idx + 1) * self.num_obstacles, len(self.obstacle_handles))
        
        # 障碍物安全距离检查（静态障碍物）
        for i in range(obstacle_start_idx, obstacle_end_idx):
            if i >= len(self.obstacle_positions) or i >= len(self.obstacle_radii):
                continue
            obs_pos = self.obstacle_positions[i]
            obs_radius = self.obstacle_radii[i].item()
            dist = torch.norm(position - obs_pos).item()
            if dist < obs_radius + 0.15:  # 15cm安全裕度
                return False
        
        # 动态障碍物安全距离检查（参考isaac-training：选择最近的N个）
        # 🎯 添加安全检查，防止段错误
        try:
            if (self.num_dynamic_obstacles > 0 and 
                self.dyn_obs_state is not None and 
                self.dyn_obs_radii is not None and
                len(self.dyn_obs_state) > 0 and
                len(self.dyn_obs_radii) > 0):
                
                dyn_obstacle_start_idx = env_idx * self.num_dynamic_obstacles
                dyn_obstacle_end_idx = (env_idx + 1) * self.num_dynamic_obstacles
                
                # 确保索引在有效范围内
                if (dyn_obstacle_end_idx > dyn_obstacle_start_idx and
                    dyn_obstacle_end_idx <= len(self.dyn_obs_state) and
                    dyn_obstacle_end_idx <= len(self.dyn_obs_radii)):
                    
                    dyn_obs_positions = self.dyn_obs_state[dyn_obstacle_start_idx:dyn_obstacle_end_idx, :3]  # [num_dyn_obs, 3]
                    dyn_obs_radii = self.dyn_obs_radii[dyn_obstacle_start_idx:dyn_obstacle_end_idx]  # [num_dyn_obs]
                    
                    # 确保张量不为空且形状正确
                    if len(dyn_obs_positions) > 0 and len(dyn_obs_radii) > 0:
                        # 计算2D距离（XY平面），用于选择最近的障碍物
                        position_2d = position[:2].unsqueeze(0)  # [1, 2]
                        dyn_obs_pos_2d = dyn_obs_positions[:, :2]  # [num_dyn_obs, 2]
                        dyn_obs_distance_2d = torch.norm(dyn_obs_pos_2d - position_2d, dim=1)  # [num_dyn_obs]
                        
                        # 选择最近的N个动态障碍物
                        num_closest = min(self.num_closest_dyn_obs, len(dyn_obs_distance_2d))
                        if num_closest > 0:
                            _, closest_idx = torch.topk(dyn_obs_distance_2d, num_closest, largest=False)
                            
                            # 检查最近的N个障碍物
                            closest_dyn_obs_pos = dyn_obs_positions[closest_idx]  # [num_closest, 3]
                            closest_dyn_obs_radii = dyn_obs_radii[closest_idx]  # [num_closest]
                            
                            # 计算3D距离
                            dists = torch.norm(position.unsqueeze(0) - closest_dyn_obs_pos, dim=1)  # [num_closest]
                            dists_to_surface = dists - closest_dyn_obs_radii  # [num_closest]
                            
                            # 如果有任何一个障碍物距离 < 15cm安全裕度，返回False
                            if (dists_to_surface < 0.15).any():
                                return False
        except Exception as e:
            # 如果动态障碍物检查失败，忽略（不返回False，继续检查其他条件）
            # 这样可以防止段错误，但可能会导致安全检查不够严格
            pass
        
        # 工作空间检查
        distance_from_base = torch.norm(position[:2]).item()
        height = position[2].item()
        if (distance_from_base > self.workspace_radius or 
            height < self.workspace_z[0] or 
            height > self.workspace_z[1]):
            return False
        
        return True
    
    def _check_target_reachability(self, env_idx: int = 0):
        """检查目标是否在机械臂可达工作空间内"""
        try:
            # 检查索引有效性
            if env_idx >= self.num_envs or env_idx < 0:
                return False
            
            target_pos = self.target_pos[env_idx]
            
            # 检查tensor是否有效
            if target_pos is None or target_pos.shape[0] < 3:
                return False
            
            # 简单的工作空间检查
            distance_from_base = torch.norm(target_pos[:2]).item()
            height = target_pos[2].item()
            
            # UR10e的大致工作空间（留有余量）
            if (distance_from_base > 1.3 or  # 最大伸展距离（略小于workspace_radius）
                height < 0.1 or height > 1.2):  # 高度范围（略小于workspace_z）
                return False
            
            return True
        except Exception as e:
            # 如果检查失败，认为目标不可达
            print(f"[Warning] _check_target_reachability() failed for env {env_idx}: {e}")
            return False
    
    def _generate_valid_target_position(self, env_idx: int, max_attempts: int = 100):
        """
        为指定环境生成一个有效的目标位置（避开障碍物、基座、工作空间内）
        
        Args:
            env_idx: 环境索引
            max_attempts: 最大尝试次数
        
        Returns:
            (target_pos: torch.Tensor[3], yaw_deg: float) 或 None（如果失败）
        """
        # 确保障碍物位置已更新（在reset时调用）
        if not hasattr(self, 'obstacle_positions') or len(self.obstacle_positions) == 0:
            self._update_obstacle_positions()
        
        for attempt in range(max_attempts):
            # 随机生成位置
            target_x = torch.rand(1, device=self.device).item() * (self.target_pos_range_x[1] - self.target_pos_range_x[0]) + self.target_pos_range_x[0]
            target_y = torch.rand(1, device=self.device).item() * (self.target_pos_range_y[1] - self.target_pos_range_y[0]) + self.target_pos_range_y[0]
            target_z = torch.rand(1, device=self.device).item() * (self.target_pos_range_z[1] - self.target_pos_range_z[0]) + self.target_pos_range_z[0]
            
            target_pos = torch.tensor([target_x, target_y, target_z], device=self.device)
            
            # 检查位置是否安全
            if not self._is_position_safe(target_pos, env_idx):
                continue
            
            # 检查是否在可达工作空间内
            distance_from_base = torch.norm(target_pos[:2]).item()
            height = target_pos[2].item()
            if (distance_from_base > 1.3 or  # 最大伸展距离
                height < 0.1 or height > 1.2):  # 高度范围
                continue
            
            # 检查是否与基座保持安全距离（基座在原点，高度约0.2m）
            base_pos = torch.tensor([0.0, 0.0, 0.2], device=self.device)
            dist_to_base = torch.norm(target_pos - base_pos).item()
            if dist_to_base < self.keepout_base_radius:
                continue
            
            # 随机生成旋转角度
            yaw_deg = torch.rand(1, device=self.device).item() * (self.target_rotation_range[1] - self.target_rotation_range[0]) + self.target_rotation_range[0]
            
            return target_pos, yaw_deg
        
        # 如果所有尝试都失败，返回默认位置（在范围内但可能不安全）
        print(f"[Warning] 环境 {env_idx} 无法生成有效目标位置，使用默认位置")
        target_x = (self.target_pos_range_x[0] + self.target_pos_range_x[1]) / 2
        target_y = (self.target_pos_range_y[0] + self.target_pos_range_y[1]) / 2
        target_z = (self.target_pos_range_z[0] + self.target_pos_range_z[1]) / 2
        target_pos = torch.tensor([target_x, target_y, target_z], device=self.device)
        yaw_deg = 0.0
        return target_pos, yaw_deg

    def reset_envs(self, env_indices=None):
        """重置部分或全部环境，可批量传入env索引"""
        if env_indices is None:
            return self.reset()
        if isinstance(env_indices, int):
            env_indices = [env_indices]
        
        num_reset = len(env_indices)
        
        # 目标状态部分
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # 🎯 修复：不要重新 acquire，只 refresh（避免破坏绑定）
        if not hasattr(self, "root_states") or self.root_states is None:
            raise RuntimeError("[Reset] root_states not initialized! Call _reacquire_root_tensor() first.")
        
        # 🎯 修复：确保障碍物位置已更新（在生成目标位置之前）
        # 注意：在reset_envs中，障碍物可能不需要重置，但位置需要更新
        self._update_obstacle_positions()
        
        # 为每个需要重置的环境生成有效的目标位置和角度
        for idx, i in enumerate(env_indices):
            target_pos, yaw_deg = self._generate_valid_target_position(i, max_attempts=100)
            
            # 计算四元数（绕Z轴旋转）
            yaw_rad = torch.deg2rad(torch.tensor(yaw_deg, device=self.device))
            qx = 0.0
            qy = 0.0
            qz = torch.sin(yaw_rad / 2).item()
            qw = torch.cos(yaw_rad / 2).item()
            
            # 目标
            target_idx = self.target_root_indices[i].cpu().item()
            if target_idx < self.root_states.shape[0]:
                new_root_state = self.root_states[target_idx].clone()
                new_root_state[0:3] = target_pos
                new_root_state[3:7] = torch.tensor([qx, qy, qz, qw], device=self.device, dtype=torch.float32)
                new_root_state[7:13] = 0.0
                self.root_states[target_idx] = new_root_state
            self.target_pos[i] = target_pos
        # 批量应用root state更改（只影响env_indices）
        # 🎯 使用统一的 _set_root_states_indexed 函数
        target_indices_selected = self.target_root_indices[torch.as_tensor(env_indices, device=self.target_root_indices.device)]
        target_indices_cpu = target_indices_selected.to(device='cpu', dtype=torch.int64)
        target_poses_cpu = torch.zeros((len(env_indices), 7), dtype=torch.float32, device='cpu')
        for idx, i in enumerate(env_indices):
            target_idx = self.target_root_indices[i].cpu().item()
            if target_idx < self.root_states.shape[0]:
                target_poses_cpu[idx, 0:3] = self.root_states[target_idx, 0:3].cpu()
                target_poses_cpu[idx, 3:7] = self.root_states[target_idx, 3:7].cpu()
        self._set_root_states_indexed(target_indices_cpu, target_poses_cpu, keep_vel=False)
        # DOF复位（关节）
        for i in env_indices:
            start_idx = i * self.dof_count
            end_idx = (i + 1) * self.dof_count
            self.dof_states[start_idx:end_idx, 0] = 0.0
            self.dof_states[start_idx:end_idx, 1] = 0.0
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
        # 运行仿真稳定
        for _ in range(3):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # 🎯 读取末端位姿并更新 prev_ee_quat（在刷新后）
        self._read_wrist_pose()
        for i in env_indices:
            if hasattr(self, 'ee_quat') and self.ee_quat is not None and i < self.ee_quat.shape[0]:
                if hasattr(self, 'prev_ee_quat') and self.prev_ee_quat is not None and i < self.prev_ee_quat.shape[0]:
                    self.prev_ee_quat[i].copy_(self.ee_quat[i])
        
        # 各辅助计数/奖励状态
        for i in env_indices:
            self.progress[i] = 0
            self.target_view_counter[i] = 0
            self.target_discovered[i] = 0
            self.wrist_rotation_steps[i] = 0  # 重置末端转动步数计数器
            self.steps_since_last_detection[i] = 0
            # 重置成功奖励标志位
            if hasattr(self, '_success_reward_given') and i < self._success_reward_given.shape[0]:
                self._success_reward_given[i] = False
            self.last_detection_step[i] = 0
            self.danger_count[i] = 0
            self._should_terminate[i] = False
            self.multiview_steps[i] = 0
            self._termination_reason[i] = None
            self.observation_angles[i] = []
            self._last_valid_ee_pos[i].zero_()
            self._last_valid_ee_quat[i].zero_()
            self._last_valid_ee_quat[i][3] = 1.0  # 单位四元数
        # 可选：探索体素、奖励缓存等，如有全局列表则按需清理
        if hasattr(self, "explored_voxels"):
            self.explored_voxels.clear()
        self.prev_action = None
        # 返回当前观测
        return self.observe()
    def _push_dyn_obs_states(self, new_states: torch.Tensor):
        """
        Write a batch of dynamic obstacles' root states to PhysX using cached SIM root indices.
        - self.root_states: shared CPU tensor acquired+wrapped once
        - self.dynamic_obstacle_root_indices: CPU Long tensor of SIM actor root indices (length K)
        """
        if not hasattr(self, "dynamic_obstacle_root_indices") or self.dynamic_obstacle_root_indices.numel() == 0:
            raise RuntimeError("dynamic_obstacle_root_indices is not initialized")
        # 🎯 使用统一的 _set_root_states_indexed 函数
        indices_cpu = self.dynamic_obstacle_root_indices.to('cpu', dtype=torch.int64).contiguous()
        new_states_cpu = new_states.to('cpu')
        poses_cpu = torch.zeros((new_states_cpu.shape[0], 7), dtype=torch.float32, device='cpu')
        poses_cpu[:, 0:3] = new_states_cpu[:, 0:3]
        poses_cpu[:, 3:7] = new_states_cpu[:, 3:7]
        self._set_root_states_indexed(indices_cpu, poses_cpu, keep_vel=False)


    def _apply_root_states_indexed(self, indices: torch.Tensor, new_states: torch.Tensor) -> bool:
        """
        正确流程：
        1) 先把 new_states 写到 self.root_states[indices]
        2) 再用【整块】root_states + indices 调用 set_actor_root_state_tensor_indexed
        形状与设备约束：
        - self.root_states: (num_actors_total, 13), CPU float32, contiguous
        - indices: int64 (long), CPU, contiguous
        - new_states: (len(indices), 13), 任意设备，最终会搬到 CPU
        """
        try:
            # 1) 确保 root_states 是 CPU float32 连续内存的大张量
            # 🎯 关键修复：不要重新赋值 self.root_states，这会破坏与 Isaac Gym 的绑定！
            # 如果设备/类型不对，应该先 refresh 再检查，或者确保在创建时就正确设置
            if self.root_states.device.type != "cpu":
                # 不要重新赋值，而是先 refresh 确保同步
                self.gym.refresh_actor_root_state_tensor(self.sim)
                # 如果还是不对，说明初始化有问题，这里只能报错
                if self.root_states.device.type != "cpu":
                    print(f"[RootState][Indexed] WARNING: root_states is on {self.root_states.device}, expected CPU")
            # 同样，不要重新赋值 dtype/contiguous，只检查
            if self.root_states.dtype != torch.float32:
                print(f"[RootState][Indexed] WARNING: root_states dtype is {self.root_states.dtype}, expected float32")
            if not self.root_states.is_contiguous():
                print(f"[RootState][Indexed] WARNING: root_states is not contiguous")

            # 2) 规范化索引与待写状态
            idx_cpu = indices.to(device="cpu", dtype=torch.int64, non_blocking=False).contiguous()
            ns_cpu  = new_states.to(device="cpu", dtype=torch.float32, non_blocking=False).contiguous()

            assert ns_cpu.shape[0] == idx_cpu.numel() and ns_cpu.shape[1] == 13, \
                f"[RootState] new_states shape={tuple(ns_cpu.shape)} vs indices={idx_cpu.numel()}"

            # 3) 先写入整块 root_states
            self.root_states.index_copy_(0, idx_cpu, ns_cpu)

            # 4) 再调用 indexed API（第一个参数必须是整块 root_states）
            # 🎯 关键修复：Isaac Gym 要求索引必须是 int32，不是 int64
            idx_i32 = idx_cpu.to(dtype=torch.int32).contiguous()
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),  # ✅ 整块 root tensor
                gymtorch.unwrap_tensor(idx_i32),           # ✅ 只给索引（int32）
                idx_i32.numel()
            )
            return True
        except Exception as e:
            print(f"[RootState][Indexed] Failed: {e}")
            return False
    
    
