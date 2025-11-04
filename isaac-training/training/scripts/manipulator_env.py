"""
Manipulator Navigation Environment for UR10e
用于UR10e机械臂的导航训练环境
"""
import torch
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.orbit.assets.articulation import Articulation, ArticulationCfg
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.prims import create_prim
from utils import vec_to_new_frame
import omni.isaac.core.utils.prims as prim_utils
from ur10_wrapper import UR10ManipulatorWrapper
import time


class ManipulatorNavigationEnv(IsaacEnv):
    """
    UR10e机械臂路径规划环境
    
    任务描述:
    - 起点: 随机初始关节配置
    - 终点: 随机ArUco位置（固定）
    - 障碍物: 静态障碍物（未知位置）+ 动态障碍物（移动）
    - 观测: 深度图 + 机械臂状态 + 动态障碍物信息
    - 动作: 末端6D速度 [vx, vy, vz, wx, wy, wz]
    """
    
    def __init__(self, cfg):
        print("[Manipulator Environment]: Initializing...")
        
        # 深度相机参数 (Realsense D435i)
        self.depth_range = cfg.sensor.depth_range
        self.depth_fov_h = cfg.sensor.depth_fov_h
        self.depth_fov_v = cfg.sensor.depth_fov_v
        self.depth_h = cfg.sensor.depth_h
        self.depth_w = cfg.sensor.depth_w
        
        super().__init__(cfg, cfg.headless)
        
        # UR10e初始化（robot 已在 _design_scene 中创建并通过 super().__init__ 完成）
        self.robot.initialize()
        
        # 深度相机初始化 (眼在手上)
        # 使用通配符 env_.* 支持多个并行环境
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/UR10/ee_link/camera",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.12)),
            attach_yaw_only=False,
            pattern_cfg=patterns.GridPatternCfg(
                resolution=(self.depth_w, self.depth_h),
                size=(self.depth_fov_h, self.depth_fov_v)
            ),
            debug_vis=False,
            # 只检测障碍物和桌子，忽略其他物体（如机械臂自身）
            mesh_prim_paths=["/World/obstacles", "/World/table"],
        )
        self.depth_camera = RayCaster(ray_caster_cfg)
        self.depth_camera._initialize_impl()
        
        # 目标和状态
        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 3)
            self.target_ori = torch.zeros(self.num_envs, 4)
            # 记录前一个时间步的末端执行器线速度和角速度
            self.prev_ee_vel = torch.zeros(self.num_envs, 6)
            
            # 工作空间限制
            self.workspace_min = torch.tensor([-0.8, -0.8, 0.0], device=self.device)
            self.workspace_max = torch.tensor([0.8, 0.8, 1.2], device=self.device)
    
    def _design_scene(self):
        """设计场景"""
        # ========== 1. UR10e机械臂 ==========
        # 使用Isaac Orbit的UR10配置
        from omni.isaac.orbit_assets.ur10 import UR10_CFG
        
        ur10_cfg = UR10_CFG.replace(prim_path="/World/envs/env_.*/UR10")
        ur10_cfg.spawn.activate_contact_sensors = True
        
        # 创建 Articulation 实例
        articulation = Articulation(cfg=ur10_cfg)
        
        # 使用包装器提供统一接口
        self.robot = UR10ManipulatorWrapper(articulation)
        
        # ========== 2. 工作台 ==========
        table_cfg = AssetBaseCfg(
            prim_path="/World/table",
            spawn=sim_utils.CuboidCfg(
                size=(1.5, 1.5, 0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.5, 0.5, 0.5)
                ),
            ),
        )
        table_cfg.spawn.func(table_cfg.prim_path, table_cfg.spawn, translation=(0, 0, -0.025))
        
        # ========== 3. 光照 ==========
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(intensity=3000.0),
        )
        light.spawn.func(light.prim_path, light.spawn)
        
        # ========== 4. 静态障碍物 ==========
        self.static_obstacles = []
        for i in range(self.cfg.env.num_static_obstacles):
            obs_cfg = RigidObjectCfg(
                prim_path=f"/World/obstacles/static_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=[0.1, 0.1, 0.3],
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.2, 0.2)
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
            obstacle = RigidObject(cfg=obs_cfg)
            self.static_obstacles.append(obstacle)
        
        # ========== 5. 动态障碍物 (类人运动) ==========
        if self.cfg.env.num_dynamic_obstacles == 0:
            return
        
        self.dynamic_obstacles = []
        self.dyn_obs_state = torch.zeros(
            (self.cfg.env.num_dynamic_obstacles, 13), 
            dtype=torch.float, device=self.device
        )
        self.dyn_obs_state[:, 3] = 1.0  # 四元数w
        self.dyn_obs_goal = torch.zeros((self.cfg.env.num_dynamic_obstacles, 3), device=self.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env.num_dynamic_obstacles, 3), device=self.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env.num_dynamic_obstacles, 3), device=self.device)
        self.dyn_obs_size = torch.zeros((self.cfg.env.num_dynamic_obstacles, 3), device=self.device)
        self.dyn_obs_step_count = 0
        
        # 创建动态障碍物（球体代表气球）
        for i in range(self.cfg.env.num_dynamic_obstacles):
            # 随机初始位置（3D空间）
            origin_x = np.random.uniform(-0.6, 0.6)
            origin_y = np.random.uniform(-0.6, 0.6)
            origin_z = np.random.uniform(0.4, 1.0)  # 气球漂浮高度范围
            origin = [origin_x, origin_y, origin_z]
            
            self.dyn_obs_origin[i] = torch.tensor(origin, device=self.device)
            self.dyn_obs_state[i, :3] = torch.tensor(origin, device=self.device)
            
            # 气球尺寸（球形，直径约20cm）
            balloon_radius = 0.10 + np.random.uniform(-0.02, 0.02)  # 8-12cm半径
            balloon_diameter = balloon_radius * 2  # 直径 16-24cm
            self.dyn_obs_size[i] = torch.tensor(
                [balloon_diameter, balloon_diameter, balloon_diameter], 
                device=self.device
            )
            
            # 创建球体（气球）
            prim_utils.create_prim(f"/World/DynObs{i}", "Xform", translation=origin)
            
            # 随机气球颜色
            colors = [
                (1.0, 0.2, 0.2),  # 红色
                (0.2, 0.2, 1.0),  # 蓝色
                (1.0, 1.0, 0.2),  # 黄色
                (0.2, 1.0, 0.2),  # 绿色
                (1.0, 0.5, 0.8),  # 粉色
            ]
            color = colors[i % len(colors)]
            
            dyn_obs_cfg = RigidObjectCfg(
                prim_path=f"/World/DynObs{i}/Sphere",
                spawn=sim_utils.SphereCfg(
                    radius=balloon_radius,
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color,
                        metallic=0.1,
                        roughness=0.3
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
            obstacle = RigidObject(cfg=dyn_obs_cfg)
            self.dynamic_obstacles.append(obstacle)
    
    def move_dynamic_obstacles(self):
        """更新气球位置（3D漂浮运动）"""
        if self.cfg.env.num_dynamic_obstacles == 0:
            return
        
        # 检查是否需要新目标
        dyn_obs_goal_dist = torch.sqrt(
            torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)
        ) if self.dyn_obs_step_count != 0 else torch.zeros(
            self.dyn_obs_state.size(0), device=self.device
        )
        
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.2  # 气球到达阈值更小
        
        # 采样新目标（3D空间随机漂浮）
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        if num_new_goal > 0:
            # XYZ三个方向都随机移动
            sample_x = -self.cfg.env_dyn.local_range[0] + 2.0 * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, device=self.device)
            sample_y = -self.cfg.env_dyn.local_range[1] + 2.0 * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, device=self.device)
            sample_z = -self.cfg.env_dyn.local_range[2] + 2.0 * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, device=self.device)
            sample_goal = torch.cat([sample_x, sample_y, sample_z], dim=1)
            
            self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal
            
            # 限制气球在工作空间内漂浮
            self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], -0.7, 0.7)  # X
            self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], -0.7, 0.7)  # Y
            self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], 0.3, 1.1)  # Z (高度范围)
        
        # 每3秒更新速度（气球漂浮更慢更随机）
        if self.dyn_obs_step_count % int(3.0/self.cfg.sim.dt) == 0:
            # 气球速度更慢
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + \
                (self.cfg.env_dyn.vel_range[1] - self.cfg.env_dyn.vel_range[0]) * \
                torch.rand(self.dyn_obs_vel.size(0), 1, device=self.device)
            
            # 计算朝向目标的方向
            direction = self.dyn_obs_goal - self.dyn_obs_state[:, :3]
            direction_norm = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
            
            # 添加随机扰动（模拟气流影响）
            random_drift = (torch.rand(self.dyn_obs_vel.size(0), 3, device=self.device) - 0.5) * 0.3
            self.dyn_obs_vel = self.dyn_obs_vel_norm * (direction / direction_norm) + random_drift
        
        # 更新位置
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt
        
        # 确保气球不会飞出工作空间
        self.dyn_obs_state[:, 0] = torch.clamp(self.dyn_obs_state[:, 0], -0.75, 0.75)
        self.dyn_obs_state[:, 1] = torch.clamp(self.dyn_obs_state[:, 1], -0.75, 0.75)
        self.dyn_obs_state[:, 2] = torch.clamp(self.dyn_obs_state[:, 2], 0.25, 1.15)
        
        # 更新仿真中的可视化
        for i, obstacle in enumerate(self.dynamic_obstacles):
            obstacle.write_root_state_to_sim(self.dyn_obs_state[i:i+1])
            obstacle.update(self.cfg.sim.dt)
        
        self.dyn_obs_step_count += 1
    
    def _set_specs(self):
        """仅供类内部使用的辅助方法"""
        """定义观测、动作、奖励空间"""
        observation_dim = 13
        num_dim_each_dyn_obs_state = 10
        
        # ========== 观测空间 ==========
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec(
                        (observation_dim,), device=self.device
                    ),
                    "depth": UnboundedContinuousTensorSpec(
                        (1, self.depth_w, self.depth_h), device=self.device
                    ),
                    "joint_pos": UnboundedContinuousTensorSpec(
                        (6,), device=self.device
                    ),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec(
                        (1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state),
                        device=self.device
                    ),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # ========== 动作空间 (6D末端速度) ==========
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((6,), device=self.device)
            })
        }).expand(self.num_envs).to(self.device)
        
        # ========== 奖励空间 ==========
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)
        
        # ========== 统计 ==========
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "joint_limit": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """重置环境"""
        # ========== 1. 重置机械臂 ==========
        init_joint_pos = torch.zeros(len(env_ids), 6, device=self.device)
        init_joint_pos[:, 0] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.5
        init_joint_pos[:, 1] = -np.pi/2 + (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.6
        init_joint_pos[:, 2] = np.pi/2 + (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.6
        init_joint_pos[:, 3] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.4
        init_joint_pos[:, 4] = np.pi/2
        init_joint_pos[:, 5] = 0.0
        
        self.robot.set_joint_positions(init_joint_pos, env_ids)
        self.robot.set_joint_velocities(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids
        )
        
        # ========== 2. 随机目标位置 (ArUco) ==========
        target_pos = torch.zeros(len(env_ids), 3, device=self.device)
        target_pos[:, 0] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 1.0
        target_pos[:, 1] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.8
        target_pos[:, 2] = 0.3 + torch.rand(len(env_ids), device=self.device) * 0.5
        
        self.target_pos[env_ids] = target_pos
        
        # ========== 3. 重置障碍物 ==========
        for i, obstacle in enumerate(self.static_obstacles):
            obs_pos = torch.zeros(len(env_ids), 3, device=self.device)
            obs_pos[:, 0] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 1.2
            obs_pos[:, 1] = (torch.rand(len(env_ids), device=self.device) - 0.5) * 1.0
            obs_pos[:, 2] = 0.15
            obstacle.write_root_pos_to_sim(obs_pos)
        
        # 重置动态障碍物（气球）
        if self.cfg.env.num_dynamic_obstacles > 0:
            for i in range(self.cfg.env.num_dynamic_obstacles):
                origin = torch.zeros(1, 3, device=self.device)
                origin[:, 0] = (torch.rand(1, device=self.device) - 0.5) * 1.0  # X: -0.5 ~ 0.5
                origin[:, 1] = (torch.rand(1, device=self.device) - 0.5) * 1.0  # Y: -0.5 ~ 0.5
                origin[:, 2] = 0.4 + torch.rand(1, device=self.device) * 0.6     # Z: 0.4 ~ 1.0 (漂浮高度)
                self.dyn_obs_origin[i] = origin[0]
                self.dyn_obs_state[i, :3] = origin[0]
        
        self.prev_ee_vel[env_ids] = 0.0
        self.stats[env_ids] = 0.0
    
    def _pre_sim_step(self, tensordict: TensorDictBase):
        """应用动作"""
        # 末端速度控制
        ee_velocity = tensordict[("agents", "action")]
        self.robot.apply_ee_velocity(ee_velocity)
    
    def _post_sim_step(self, tensordict: TensorDictBase):
        """后处理"""
        # 传感器更新
        if self.cfg.env.num_dynamic_obstacles > 0:
            self.move_dynamic_obstacles()
        self.depth_camera.update(self.dt)
    
    def _compute_state_and_obs(self):
        """计算观测和奖励"""
        # ========== 1. 机械臂状态 ==========
        ee_state = self.robot.get_ee_state()
        ee_pos = ee_state[..., :3]
        ee_quat = ee_state[..., 3:7]
        ee_vel = ee_state[..., 7:13]
        joint_pos = self.robot.get_joint_positions()
        
        # ========== 2. 深度图 ==========
        depth_scan = self.depth_range - (
            (self.depth_camera.data.ray_hits_w - self.depth_camera.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.depth_range)
            .reshape(self.num_envs, 1, self.depth_w, self.depth_h)
        )
        
        # ========== 3. 目标相对状态 ==========
        rpos = self.target_pos - ee_pos  # 相对位置向量
        distance = rpos.norm(dim=-1, keepdim=True)  # 总距离
        distance_xy = rpos[..., :2].norm(dim=-1, keepdim=True) # 水平面距离
        distance_z = rpos[..., 2].unsqueeze(-1) # 垂直方向距离
        
        # 目标方向（完整3D方向，适合机械臂）
        target_dir = rpos / rpos.norm(dim=-1, keepdim=True).clamp(1e-6)
        
        rpos_normalized = rpos / distance.clamp(1e-6)  # 单位化相对位置向量
        # 将相对位置从世界坐标系转换到目标坐标系
        rpos_normalized_g = vec_to_new_frame(rpos_normalized.unsqueeze(1), target_dir.unsqueeze(1)).squeeze(1)
        
        # 末端速度（目标坐标系）
        ee_linear_vel = ee_vel[..., :3]
        # 将相对位置从世界坐标系转换到目标坐标系
        ee_linear_vel_g = vec_to_new_frame(ee_linear_vel.unsqueeze(1), target_dir.unsqueeze(1)).squeeze(1)
        
        # ========== 4. 拼接机械臂状态 ==========
        robot_state = torch.cat([
            rpos_normalized_g,  # 3
            distance_xy,        # 1
            distance_z,         # 1
            ee_linear_vel_g,    # 3
        ], dim=-1)  # total: 8 (与无人机保持一致)
        
        # ========== 5. 动态障碍物状态 ==========
        if self.cfg.env.num_dynamic_obstacles > 0:
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos = dyn_obs_pos_expanded - ee_pos.unsqueeze(1)
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos[..., :2], dim=2)
            
            # 选择最近的N个
            _, closest_idx = torch.topk(
                dyn_obs_distance_2d, 
                self.cfg.algo.feature_extractor.dyn_obs_num, 
                dim=1, largest=False
            )
            closest_dyn_obs_rpos = torch.gather(
                dyn_obs_rpos, 1, 
                closest_idx.unsqueeze(-1).expand(-1, -1, 3)
            )
            
            # 转换到目标坐标系
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir.unsqueeze(1))
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)
            
            # 速度
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_idx]
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir.unsqueeze(1))
            
            # 尺寸编码（气球）
            closest_dyn_obs_size = self.dyn_obs_size[closest_idx]
            # 气球直径范围: [0.16, 0.24]m，使用0.20m作为基准
            # encoded = (diameter / 0.20) - 1.0
            # 范围: [0.16/0.20-1, 0.24/0.20-1] = [-0.2, 0.2]
            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1) / 0.20 - 1.0
            closest_dyn_obs_height = torch.ones_like(closest_dyn_obs_width)  # 气球球形，宽高相同
            
            dyn_obs_states = torch.cat([
                closest_dyn_obs_rpos_gn,
                closest_dyn_obs_distance_2d,
                closest_dyn_obs_distance_z,
                closest_dyn_obs_vel_g,
                closest_dyn_obs_width,
                closest_dyn_obs_height
            ], dim=-1).unsqueeze(1)
        else:
            dyn_obs_states = torch.zeros(
                self.num_envs, 1, 
                self.cfg.algo.feature_extractor.dyn_obs_num, 10,
                device=self.device
            )
        
        # ========== 6. 组装观测 ==========
        obs = {
            "state": robot_state,
            "depth": depth_scan,
            "joint_pos": joint_pos,
            "dynamic_obstacle": dyn_obs_states
        }
        
        # ========== 7. 计算奖励 ==========
        self._compute_rewards(
            ee_pos, ee_linear_vel, rpos, distance, 
            depth_scan, dyn_obs_states, closest_dyn_obs_rpos
        )
        
        # ========== 8. 终止条件 ==========
        reach_goal = (distance.squeeze(-1) < 0.2)
        
        # 碰撞检测
        min_depth = depth_scan.min(dim=(2, 3))[0]
        static_collision = min_depth > (self.depth_range - 0.08)
        
        # 动态碰撞（气球碰撞检测）
        if self.cfg.env.num_dynamic_obstacles > 0:
            dyn_collision_dist = closest_dyn_obs_rpos.norm(dim=-1)
            # 修正后的碰撞阈值：
            balloon_radius = 0.12  # 气球半径 12cm
            arm_radius = 0.12      # 机械臂有效半径 12cm
            safety_margin = 0.05   # 安全裕度 5cm
            collision_threshold = balloon_radius + arm_radius + safety_margin
            # 气球半径约0.1m，机械臂半径约0.05m，碰撞阈值=0.15m
            dyn_collision = (dyn_collision_dist < collision_threshold).any(dim=1, keepdim=True)
        else:
            dyn_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        
        collision = static_collision | dyn_collision
        
        # 关节限位
        joint_limit = self._check_joint_limits(joint_pos)
        
        # 工作空间限制
        out_of_workspace = (
            (ee_pos < self.workspace_min) | (ee_pos > self.workspace_max)
        ).any(dim=-1, keepdim=True)
        
        self.terminated = collision | joint_limit | out_of_workspace
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        
        # ========== 9. 更新统计 ==========
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["joint_limit"] = joint_limit.float()
        
        return TensorDict({
            "agents": TensorDict({"observation": obs}, [self.num_envs]),
            "stats": self.stats.clone(),
        }, self.batch_size)
    
    def _compute_rewards(self, ee_pos, ee_vel, rpos, distance, depth_scan, dyn_obs_states, dyn_obs_rpos):
        """计算奖励"""
        # 1. 速度奖励 (朝目标方向)
        vel_direction = rpos / distance.clamp(1e-6)
        # 速度在目标方向上的分量大小
        reward_vel = (ee_vel * vel_direction).sum(-1, keepdim=True)
        
        # 2. 安全奖励 (静态障碍物)
        reward_safety_static = torch.log(
            (self.depth_range - depth_scan).clamp(min=0.05, max=self.depth_range)
        ).mean(dim=(2, 3))
        
        # 3. 安全奖励 (动态障碍物)
        if self.cfg.env.num_dynamic_obstacles > 0:
            dyn_obs_distance = dyn_obs_rpos.norm(dim=-1)
            reward_safety_dynamic = torch.log(
                dyn_obs_distance.clamp(min=0.1, max=self.depth_range)
            ).mean(dim=-1, keepdim=True)
        else:
            reward_safety_dynamic = 0.0
        
        # 4. 平滑奖励
        penalty_smooth = (ee_vel - self.prev_ee_vel).norm(dim=-1, keepdim=True)
        self.prev_ee_vel = ee_vel.clone()
        
        # 5. 距离奖励
        # 距离奖励：-distance（鼓励减少绝对距离）
        reward_distance = -distance * 0.5
        
        # 6. 总奖励
        self.reward = (
            reward_vel + 
            1.0 +
            reward_safety_static * 0.5 + 
            reward_safety_dynamic * 0.5 +
            reward_distance -
            penalty_smooth * 0.1
        )
    
    def _check_joint_limits(self, joint_pos):
        """检查关节限位"""
        # UR10e实际限位
        limits_lower = torch.tensor(
            [-6.28, -6.28, -6.28, -6.28, -6.28, -6.28],
            device=self.device
        )
        limits_upper = torch.tensor(
            [6.28, 6.28, 6.28, 6.28, 6.28, 6.28],
            device=self.device
        )
        
        below = (joint_pos < limits_lower).any(dim=-1, keepdim=True)
        above = (joint_pos > limits_upper).any(dim=-1, keepdim=True)
        
        return below | above
    
    def _compute_reward_and_done(self):
        """返回奖励和终止信号"""
        return TensorDict({
            "agents": {"reward": self.reward},
            "done": self.terminated | self.truncated,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }, self.batch_size)

