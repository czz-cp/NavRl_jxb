import os
import sys
import yaml
import json
import csv
from datetime import datetime
from pathlib import Path

# IMPORTANT: Isaac Gym must be imported before PyTorch
from isaacgym import gymapi

import torch
import numpy as np

# 添加项目根目录到路径，支持直接运行
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_project_root = os.path.dirname(_parent_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from isaac_gym_manipulator.active_training.env import ArmNavEnv
from isaac_gym_manipulator.active_training.ppo import PPO
from isaac_gym_manipulator.active_training.utils import set_seed, angle_shaping, build_voxel_from_depth, update_voxel_from_depth, GAE


def compute_rewards(*args, **kwargs):
    raise RuntimeError("compute_rewards moved into env.compute_rewards; update calls accordingly.")


def prepare_voxel_for_training(voxel, num_envs, voxel_shape):
    """
    统一处理体素形状，确保为 [num_envs, C, Vx, Vy, Vz]
    支持3/4/5维多输入情况，自动align
    """
    C, Vx, Vy, Vz = voxel_shape
    if voxel.dim() == 3:
        if voxel.shape == (Vx, Vy, Vz):
            voxel = voxel.unsqueeze(0).unsqueeze(0)  # [1,1,Vx,Vy,Vz]
        else:
            raise ValueError(f"3维体素形状错误: {voxel.shape}，期望 {(Vx, Vy, Vz)}")
    elif voxel.dim() == 4:
        if voxel.shape == (C, Vx, Vy, Vz):
            voxel = voxel.unsqueeze(0)
        elif voxel.shape[1:] == (Vx, Vy, Vz):
            voxel = voxel.unsqueeze(1)
        else:
            raise ValueError(f"4维体素形状错误: {voxel.shape}")
    elif voxel.dim() == 5:
        if voxel.shape[2:] != (Vx, Vy, Vz):
            raise ValueError(f"5维体素空间维度错误: {voxel.shape[2:]}, 期望 {(Vx, Vy, Vz)}")
    else:
        raise ValueError(f"不支持的体素维度: {voxel.dim()}，期望3/4/5")
    if voxel.shape[1] != C:
        if voxel.shape[1] == 1 and C > 1:
            voxel = voxel.expand(-1, C, -1, -1, -1)
        elif voxel.shape[1] != C:
            raise ValueError(f"体素通道数错误: {voxel.shape[1]}，期望 {C}")
    if voxel.shape[0] == 1 and num_envs > 1:
        voxel = voxel.expand(num_envs, -1, -1, -1, -1)
    elif voxel.shape[0] != num_envs:
        repeat_times = (num_envs + voxel.shape[0] - 1) // voxel.shape[0]
        voxel = voxel.repeat(repeat_times, 1, 1, 1, 1)[:num_envs]
    if voxel.shape != (num_envs, C, Vx, Vy, Vz):
        raise ValueError(f"体素最终形状错误: {voxel.shape}，期望 {(num_envs, C, Vx, Vy, Vz)}")
    return voxel

def prepare_aux_for_training(aux, num_envs, aux_dim):
    if aux.dim() == 1:
        if aux.shape[0] == aux_dim:
            aux = aux.unsqueeze(0)
        else:
            raise ValueError(f"1维aux形状错误: {aux.shape}，期望 [{aux_dim}]")
    elif aux.dim() == 2:
        if aux.shape[1] != aux_dim:
            raise ValueError(f"2维aux特征维度错误: {aux.shape[1]}，期望 {aux_dim}")
    else:
        raise ValueError(f"不支持的aux维度: {aux.dim()}，期望1/2")
    if aux.shape[0] == 1 and num_envs > 1:
        aux = aux.expand(num_envs, -1)
    elif aux.shape[0] != num_envs:
        repeat_times = (num_envs + aux.shape[0] - 1) // aux.shape[0]
        aux = aux.repeat(repeat_times, 1)[:num_envs]
    if aux.shape != (num_envs, aux_dim):
        raise ValueError(f"aux最终形状错误: {aux.shape}，期望 {(num_envs, aux_dim)}")
    return aux

# 单元测试辅助
def test_voxel_preparation(logger, num_envs, voxel_shape, Vx, Vy, Vz, C):
    test_cases = [
        torch.randn(Vx, Vy, Vz),
        torch.randn(C, Vx, Vy, Vz),
        torch.randn(1, C, Vx, Vy, Vz),
        torch.randn(num_envs, C, Vx, Vy, Vz),
    ]
    for i, test_voxel in enumerate(test_cases):
        try:
            result = prepare_voxel_for_training(test_voxel, num_envs, voxel_shape)
            logger.log(f"体素测试用例 {i+1} 通过: {test_voxel.shape} -> {result.shape}")
        except Exception as e:
            logger.error(f"体素测试用例 {i+1} 失败: {test_voxel.shape}", exception=e)
            raise


class DebugLogger:
    """调试信息记录器"""
    
    def __init__(self, debug_dir):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.logs_dir = self.debug_dir / "logs"
        self.stats_dir = self.debug_dir / "stats"
        self.errors_dir = self.debug_dir / "errors"
        self.states_dir = self.debug_dir / "states"
        
        for dir_path in [self.logs_dir, self.stats_dir, self.errors_dir, self.states_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 初始化日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"training_{timestamp}.log"
        self.error_file = self.errors_dir / f"errors_{timestamp}.log"
        
        # 初始化CSV文件
        self.stats_csv = self.stats_dir / f"training_stats_{timestamp}.csv"
        self.episode_csv = self.stats_dir / f"episode_stats_{timestamp}.csv"
        
        # 写入CSV头部
        with open(self.stats_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'actor_loss', 'critic_loss', 'entropy', 'approx_kl',
                'clip_fraction', 'value_clip_frac', 'mean_reward', 'std_reward',
                'min_reward', 'max_reward', 'mean_value', 'std_value',
                'mean_advantage', 'std_advantage', 'episode_count', 'success_count',
                'success_rate', 'mean_episode_length', 'grad_norm_actor', 'grad_norm_critic'
            ])
        
        with open(self.episode_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'step', 'env_idx', 'episode_reward', 'episode_length',
                'success', 'termination_reason', 'success_reason'
            ])
        
        self.log(f"调试日志系统初始化完成，保存目录: {self.debug_dir}")
    
    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        
        # 写入文件（不打印，避免重复输出）
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    def error(self, message, exception=None):
        """记录错误"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_msg = f"[{timestamp}] [ERROR] {message}"
        
        if exception:
            import traceback
            error_msg += f"\nException: {str(exception)}\n{traceback.format_exc()}"
        
        # 打印到控制台
        print(error_msg)
        
        # 写入错误文件
        with open(self.error_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + '\n')
    
    def save_stats(self, iteration, stats_dict):
        """保存训练统计"""
        with open(self.stats_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration,
                stats_dict.get('actor_loss', 0.0),
                stats_dict.get('critic_loss', 0.0),
                stats_dict.get('entropy', 0.0),
                stats_dict.get('approx_kl', 0.0),
                stats_dict.get('clip_fraction', 0.0),
                stats_dict.get('value_clip_frac', 0.0),
                stats_dict.get('mean_reward', 0.0),
                stats_dict.get('std_reward', 0.0),
                stats_dict.get('min_reward', 0.0),
                stats_dict.get('max_reward', 0.0),
                stats_dict.get('mean_value', 0.0),
                stats_dict.get('std_value', 0.0),
                stats_dict.get('mean_advantage', 0.0),
                stats_dict.get('std_advantage', 0.0),
                stats_dict.get('episode_count', 0),
                stats_dict.get('success_count', 0),
                stats_dict.get('success_rate', 0.0),
                stats_dict.get('mean_episode_length', 0.0),
                stats_dict.get('grad_norm_actor', 0.0),
                stats_dict.get('grad_norm_critic', 0.0),
            ])
    
    def save_episode(self, iteration, step, env_idx, episode_reward, episode_length, 
                     success=False, termination_reason=None, success_reason=None):
        """保存episode统计"""
        with open(self.episode_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                iteration, step, env_idx, episode_reward, episode_length,
                success, termination_reason, success_reason
            ])
    
    def save_config(self, cfg):
        """保存配置"""
        config_file = self.debug_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, default=str)
        self.log(f"配置已保存到: {config_file}")
    
    def save_state(self, name, data, iteration=None):
        """保存中间状态"""
        if iteration is not None:
            state_file = self.states_dir / f"{name}_iter_{iteration}.npy"
        else:
            state_file = self.states_dir / f"{name}.npy"
        
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        np.save(state_file, data)
        # 不打印，避免输出过多
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 状态已保存: {name} -> {state_file}\n")


def main():
    root = os.path.dirname(__file__)
    cfg_path = os.path.join(root, 'config.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 🎯 初始化调试日志系统
    debug_dir = cfg.get('train', {}).get('debug_dir', os.path.join(root, 'debug_output'))
    debug_dir = os.path.join(debug_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    logger = DebugLogger(debug_dir)
    
    print("="*80)
    print(f"[Training] 调试信息保存目录: {debug_dir}")
    print("="*80)
    logger.log("开始训练")
    logger.log(f"调试信息保存目录: {debug_dir}")
    
    # 保存配置
    logger.save_config(cfg)

    set_seed(42)
    env_cfg = cfg['env']
    env = ArmNavEnv(
        num_envs=env_cfg.get('num_envs', 1),
        max_steps=env_cfg.get('max_steps', 800),
        img_w=env_cfg.get('img_w', 160),
        img_h=env_cfg.get('img_h', 120),
        fov_deg=env_cfg.get('fov_deg', 60.0),
        enable_viewer=env_cfg.get('enable_viewer', False),
        camera_pos=env_cfg.get('camera_pos', [2.0, 2.0, 2.0]),
        camera_target=env_cfg.get('camera_target', [0.0, 0.0, 0.5]),
        cfg=cfg,  # 传递完整配置以便读取PD控制器参数
    )
    logger.log(f"环境创建成功: {env.num_envs} 个环境")
    
    # 设置运行期参数
    env.target_confirm_steps = env_cfg.get('target_confirm_steps', 5)
    env.max_range = env_cfg.get('max_range', 5.0)  # 深度相机最大范围
    env.num_obstacles = env_cfg.get('obstacles_num', 5)
    env.visualize_env_index = env_cfg.get('visualize_env_index', 0)  # 只可视化指定环境
    env.sphere_radius_min = env_cfg.get('obstacle_radius_min', 0.10)
    env.sphere_radius_max = env_cfg.get('obstacle_radius_max', 0.30)
    env.obstacle_min_spacing = env_cfg.get('obstacle_min_spacing', 0.40)
    env.aruco_safe_radius = env_cfg.get('aruco_safe_radius', 0.30)
    env.los_clearance = env_cfg.get('los_clearance', 0.25)
    
    # 动态障碍物配置
    env.num_dynamic_obstacles = env_cfg.get('num_dynamic_obstacles', 0)
    if env.num_dynamic_obstacles > 0:
        env.dynamic_obstacle_radius_min = env_cfg.get('dynamic_obstacle_radius_min', 0.08)
        env.dynamic_obstacle_radius_max = env_cfg.get('dynamic_obstacle_radius_max', 0.15)
        env.dynamic_obstacle_vel_range = env_cfg.get('dynamic_obstacle_vel_range', [0.05, 0.15])
        env.dynamic_obstacle_local_range = env_cfg.get('dynamic_obstacle_local_range', [0.4, 0.4, 0.3])
        env.dynamic_obstacle_goal_threshold = env_cfg.get('dynamic_obstacle_goal_threshold', 0.2)
        env.dynamic_obstacle_vel_update_interval = env_cfg.get('dynamic_obstacle_vel_update_interval', 2.0)
        env.num_closest_dyn_obs = env_cfg.get('num_closest_dyn_obs', 3)  # 检测最近的N个动态障碍物
        logger.log(f"动态障碍物配置: {env.num_dynamic_obstacles} 个/环境, 速度范围 {env.dynamic_obstacle_vel_range} m/s, 检测最近 {env.num_closest_dyn_obs} 个")
        
        # 🎯 重要：在配置设置完成后创建动态障碍物
        print(f"[Train] 正在创建动态障碍物...")
        env._spawn_dynamic_obstacles()
        env.test_dynamic_obstacles_visibility()  # 🎯 添加测试
        print(f"[Train] 动态障碍物创建完成")
    env.workspace_radius = env_cfg.get('workspace_radius', 1.4)
    env.workspace_z = env_cfg.get('workspace_z', [0.05, 1.2])
    env.keepout_base_radius = env_cfg.get('keepout_base_radius', 0.60)
    env.keepout_target_margin = env_cfg.get('keepout_target_margin', 0.05)
    
    # ArUco位置范围和旋转角度
    env.target_pos_range_x = env_cfg.get('target_pos_range_x', [0.5, 1.2])
    env.target_pos_range_y = env_cfg.get('target_pos_range_y', [-0.5, 0.5])
    env.target_pos_range_z = env_cfg.get('target_pos_range_z', [0.25, 0.6])
    env.target_rotation_range = env_cfg.get('target_rotation_range', [-180, 180])

    Vx, Vy, Vz = cfg['voxel']['size']
    voxel_shape = (cfg['voxel']['channels'], Vx, Vy, Vz)
    
    # 🎯 根据是否使用nav_style_features决定观测维度
    use_nav_style_features = cfg['ppo'].get('use_nav_style_features', False)
    if use_nav_style_features:
        # LiDAR + Dynamic Obstacle + State 格式
        # state维度保持18维（包含ArUco检测状态）
        state_dim = 18  # 18维观测空间：目标状态(4) + 机械臂状态(9) + 感知信息(5)
        aux_dim = state_dim  # 为了向后兼容，保留aux_dim名称
    else:
        # 旧的体素+aux格式
        aux_dim = 18  # 18维观测空间（基于论文重新设计：目标状态4维+机械臂状态9维+感知信息5维）
    
    action_dim = cfg['env']['action_dim']  # 从env配置读取，应该是6

    # 将动作缩放参数传递给PPO（从env配置读取）
    ppo_cfg = cfg['ppo'].copy()
    ppo_cfg['action_limit'] = env_cfg.get('action_limit', 0.0189)  # 默认值改为0.0189（参考ur5e_DDPG）
    ppo_cfg['num_closest_dyn_obs'] = env_cfg.get('num_closest_dyn_obs', 3)  # 传递给PPO用于动态障碍物MLP初始化
    # 注意：已移除两步缩放，不再需要linear_vel_scale和angular_vel_scale
    
    algo = PPO(ppo_cfg, (voxel_shape, aux_dim), action_dim, env.device)

    total_iters = cfg['train']['total_iterations']#总迭代次数
    rollout = cfg['ppo']['rollout_steps']#rollout步数
    gamma = cfg['ppo']['gamma']#折扣因子
    lam = cfg['ppo']['gae_lambda']#GAE的lambda参数

    prev_angle_deg = None#上一个关节角度
    env.reset()
    
    # 🎯 修复：在reset后立即显示viewer窗口（如果启用）
    if env.enable_viewer and env.viewer is not None:
        try:
            print("[Visualization] 正在显示viewer窗口...")
            env.gym.poll_viewer_events(env.viewer)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)  # 第一次使用阻塞模式确保窗口显示
            print("[Visualization] ✅ Viewer窗口已显示")
        except Exception as e:
            print(f"[Warning] 无法显示viewer窗口: {e}")
    
    # 🎯 关键修复：reset()后GPU同步，确保动态障碍物更新完成
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # 体素网格定义：以末端为中心的立方体
    grid_origin = torch.tensor([-1.0, -1.0, -1.0], device=env.device)
    grid_size = torch.tensor([Vx, Vy, Vz], device=env.device)
    voxel_res = torch.tensor([2.0 / Vx, 2.0 / Vy, 2.0 / Vz], device=env.device)

    # 🎯 自适应折扣因子配置
    use_adaptive_gamma = cfg['ppo'].get('use_adaptive_gamma', False)
    eta_min = cfg['ppo'].get('eta_min', 0.6)
    eta_max = cfg['ppo'].get('eta_max', 0.99)
    
    gae = GAE(gamma=gamma, lam=lam, device=env.device,
              use_adaptive_gamma=use_adaptive_gamma,
              eta_min=eta_min, eta_max=eta_max)
    
    # 增量体素更新选项（体素构建优化）
    use_incremental_voxel = cfg.get('voxel', {}).get('use_incremental_update', False)
    global_voxel = None#全局体素地图
    if use_incremental_voxel:
        # 初始化全局体素地图 [1, 1, Vx, Vy, Vz]（包含通道维度）
        global_voxel = torch.zeros((1, 1, Vx, Vy, Vz), device=env.device)
        print(f"[Train] 启用增量体素更新模式")

    # 预分配rollout tensors（内存优化）
    num_envs = env.num_envs
    aux_dim_val = aux_dim  # 观测维度（根据use_nav_style_features决定）
    action_dim_val = action_dim if isinstance(action_dim, int) else cfg['env']['action_dim']
    
    # 🎯 如果使用nav_style_features，需要预分配字典格式的观测
    use_nav_style_features = cfg['ppo'].get('use_nav_style_features', False)
    
    print(f"\n{'='*80}")
    print(f"[Training] Starting training: {total_iters} iterations, {rollout} rollout steps")
    print(f"  Environment: {num_envs} env(s) | Max steps: {env.max_steps}")
    print(f"  Voxel grid: {Vx}x{Vy}x{Vz} | Channels: {cfg['voxel']['channels']}")
    print(f"  Action dim: {action_dim} | Observation dim: {aux_dim_val}")
    print(f"  PPO: epochs={cfg['ppo']['epochs']} | batch_size={cfg['ppo']['batch_size']} | lr={cfg['ppo']['lr']}")
    if env.enable_viewer:
        print(f"  Visualization: ✅ ENABLED | Render freq: {env_cfg.get('render_freq', 1)} | Camera: {env_cfg.get('camera_pos', [2.0, 2.0, 2.0])}")
    else:
        print(f"  Visualization: ❌ DISABLED")
    print(f"{'='*80}\n")
    
    logger.log(f"训练开始: {total_iters} 次迭代, {rollout} rollout步数")
    logger.log(f"环境: {num_envs} 个环境 | 最大步数: {env.max_steps}")
    logger.log(f"体素网格: {Vx}x{Vy}x{Vz} | 通道数: {cfg['voxel']['channels']}")
    logger.log(f"动作维度: {action_dim} | 观测维度: {aux_dim_val}")
    logger.log(f"PPO参数: epochs={cfg['ppo']['epochs']} | batch_size={cfg['ppo']['batch_size']} | lr={cfg['ppo']['lr']}")
    
    logger.log("开始体素处理函数测试...")
    test_voxel_preparation(logger, env.num_envs, voxel_shape, Vx, Vy, Vz, cfg['voxel']['channels'])
    logger.log("体素处理函数测试通过！")

    # 🎯 初始化深度图缓存（用于aruco_render_freq控制）
    last_depth_cache = {'depth': None, 'cam_T_w': None, 'intr': None}
    
    for it in range(1, total_iters + 1):#总迭代次数
        print(f"\n[Iteration {it}/{total_iters}] Starting rollout collection...")
        
        # 🎯 关键修复：在每个iteration开始时，清空已计数环境的集合，避免跨iteration的计数问题
        if not hasattr(main, '_counted_envs_in_rollout'):
            main._counted_envs_in_rollout = set()
        else:
            main._counted_envs_in_rollout.clear()
        
        # 预分配tensor替代列表append（内存优化）
        voxels = torch.zeros((rollout, num_envs, *voxel_shape), device=env.device)#体素地图
        auxes = torch.zeros((rollout, num_envs, aux_dim_val), device=env.device)#机器人状态信息
        acts = torch.zeros((rollout, num_envs, action_dim_val), device=env.device)#执行的动作记录
        logps = torch.zeros((rollout, num_envs), device=env.device)#动作的对数概率
        rews = torch.zeros((rollout, num_envs), device=env.device)#奖励
        vals = torch.zeros((rollout, num_envs), device=env.device)#价值函数估计
        vals_den = torch.zeros((rollout, num_envs), device=env.device)#价值函数估计的归一化
        dones_list = torch.zeros((rollout, num_envs), dtype=torch.bool, device=env.device)#是否完成
        # 🎯 动作概率（用于自适应折扣因子）
        action_probs = torch.zeros((rollout, num_envs), device=env.device) if use_adaptive_gamma else None
        
        # 🎯 如果使用nav_style_features，初始化obs_dicts列表
        obs_dicts = [] if use_nav_style_features else None
        
        # 🎯 关键修复：验证张量形状匹配，防止内存访问越界
        try:
            assert voxels.shape == (rollout, num_envs, *voxel_shape), \
                f"[Error] Voxel shape mismatch: {voxels.shape} vs {(rollout, num_envs, *voxel_shape)}"
            assert auxes.shape == (rollout, num_envs, aux_dim_val), \
                f"[Error] Aux shape mismatch: {auxes.shape} vs {(rollout, num_envs, aux_dim_val)}"
            assert acts.shape == (rollout, num_envs, action_dim_val), \
                f"[Error] Acts shape mismatch: {acts.shape} vs {(rollout, num_envs, action_dim_val)}"
            assert logps.shape == (rollout, num_envs), \
                f"[Error] Logps shape mismatch: {logps.shape} vs {(rollout, num_envs)}"
            assert rews.shape == (rollout, num_envs), \
                f"[Error] Rews shape mismatch: {rews.shape} vs {(rollout, num_envs)}"
            assert vals.shape == (rollout, num_envs), \
                f"[Error] Vals shape mismatch: {vals.shape} vs {(rollout, num_envs)}"
            assert dones_list.shape == (rollout, num_envs), \
                f"[Error] Dones shape mismatch: {dones_list.shape} vs {(rollout, num_envs)}"
            if action_probs is not None:
                assert action_probs.shape == (rollout, num_envs), \
                    f"[Error] Action_probs shape mismatch: {action_probs.shape} vs {(rollout, num_envs)}"
            logger.log("✅ 张量形状验证通过")
        except AssertionError as e:
            logger.error(f"❌ 张量形状验证失败: {e}")
            raise

        # 数据收集阶段（rollout）
        episode_rewards = torch.zeros(num_envs, device=env.device)
        episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        episode_count = 0  # 总episode计数
        success_count = 0  # 成功episode计数
        
        for t in range(rollout):
            # 显示 rollout 进度
            if (t + 1) % 10 == 0 or t == 0:#每10步或第一步打印进度
                print(f"[Rollout] Step {t+1}/{rollout} (iter {it}/{total_iters})")
                logger.log(f"[Rollout] Step {t+1}/{rollout} (iter {it}/{total_iters})")
            
            # 🎯 关键修复：当使用LiDAR时，跳过所有相机和体素操作
            # 当启用viewer或使用nav_style_features时，跳过相机和体素操作
            if env.enable_viewer or use_nav_style_features:
                # 当启用viewer或使用LiDAR时，跳过所有相机和体素操作，使用空体素地图
                # 创建空的体素地图（用于向后兼容，但PPO不会使用）
                Nx, Ny, Nz = map(int, grid_size.tolist())
                voxel = torch.zeros((1, 1, Nx, Ny, Nz), device=env.device)
                # 获取观测（字典格式或旧格式）
                obs = env.observe(voxel=voxel, grid_origin=grid_origin, voxel_res=voxel_res)
                if use_nav_style_features and isinstance(obs, dict):
                    # 字典格式：提取state（18维，包含ArUco检测状态）
                    aux = obs['state']  # [num_envs, 18]
                else:
                    # 旧格式：直接使用
                    aux = obs
            else:
                # 不启用viewer且不使用LiDAR时，正常执行所有操作（使用体素地图）
                # 🎯 使用aruco_render_freq控制渲染频率（从config读取）
                aruco_render_freq = getattr(env, 'aruco_render_freq', env_cfg.get('aruco_render_freq', 5))
                if t % aruco_render_freq == 0 or t == 0:
                    # 只在满足频率条件时渲染深度图
                    depth = env.render_depth()#渲染深度图
                    cam_T_w = env.camera_pose()
                    intr = env.intrinsics#相机内参矩阵
                    # 缓存最新的深度图（用于后续步骤）
                    last_depth_cache['depth'] = depth
                    last_depth_cache['cam_T_w'] = cam_T_w
                    last_depth_cache['intr'] = intr
                else:
                    # 不渲染时，使用上一次缓存的深度图
                    if last_depth_cache['depth'] is None:
                        # 第一次调用或缓存未初始化，必须渲染
                        depth = env.render_depth()
                        cam_T_w = env.camera_pose()
                        intr = env.intrinsics
                        last_depth_cache['depth'] = depth
                        last_depth_cache['cam_T_w'] = cam_T_w
                        last_depth_cache['intr'] = intr
                    else:
                        # 使用缓存的深度图
                        depth = last_depth_cache['depth']
                        cam_T_w = last_depth_cache['cam_T_w']
                        intr = last_depth_cache['intr']
                
                # 🎯 调试：检查深度图
                if t == 0 and it == 1:
                    # 🎯 修复：检查深度图是否为空，避免在空张量上调用 min/max
                    if depth.numel() > 0:
                        depth_min = depth.min().item()
                        depth_max = depth.max().item()
                        logger.log(f"深度图形状: {depth.shape}, 范围: [{depth_min:.3f}, {depth_max:.3f}]")
                        logger.save_state("depth_first_step", depth, iteration=it)
                    else:
                        logger.log(f"深度图形状: {depth.shape}, 但张量为空（numel=0）")
                        logger.warning("深度图为空，跳过保存")
                
                # 检查深度图是否有无效值（仅在非空时检查和处理）
                if depth.numel() > 0:
                    if torch.isnan(depth).any() or torch.isinf(depth).any():
                        if t == 0:  # 只在第一步打印，避免刷屏
                            nan_count = torch.isnan(depth).sum().item()
                            inf_count = torch.isinf(depth).sum().item()
                            logger.error(f"深度图包含无效值: NaN={nan_count}, Inf={inf_count}")
                        # 将非有限值替换为最大范围值
                        depth = torch.where(torch.isfinite(depth), depth, 
                                           torch.full_like(depth, env.max_range))
                else:
                    if t == 0:  # 只在第一步打印
                        logger.warning("深度图为空，跳过有效性检查和体素构建")
                    # 如果深度图为空，使用已有的 global_voxel 或创建空体素地图
                    if global_voxel is not None:
                        voxel = global_voxel.clone()
                    else:
                        # 创建空的体素地图
                        Nx, Ny, Nz = map(int, grid_size.tolist())
                        voxel = torch.zeros((1, 1, Nx, Ny, Nz), device=env.device)
                    # 跳过体素构建，直接使用已有的体素地图
                    # 注意：后续代码仍会正常执行（动作、观测等）
                
                # 体素构建优化：使用增量更新或完全重建（仅在深度图非空时）
                if depth.numel() > 0:
                    if use_incremental_voxel and global_voxel is not None:
                        # 增量更新全局体素地图
                        global_voxel = update_voxel_from_depth(
                            global_voxel=global_voxel,
                            depth=depth,
                            cam_T_w=cam_T_w,
                            intrinsics=intr,
                            grid_origin=grid_origin,
                            grid_size=grid_size, # 体素数量 [Vx, Vy, Vz]
                            voxel_res=voxel_res,# 体素分辨率 [2.0/Vx, 2.0/Vy, 2.0/Vz]
                            max_range=env.max_range,
                            subsample=4, # 像素采样步长
                            decay_factor=cfg.get('voxel', {}).get('decay_factor', 0.95),
                            device=env.device,
                        )
                        voxel = global_voxel.clone()  # 使用更新后的全局地图
                    else:
                        # 完全重建体素地图（原始方法）
                        voxel = build_voxel_from_depth(
                            depth=depth,
                            cam_T_w=cam_T_w,
                            intrinsics=intr,
                            grid_origin=grid_origin,
                            grid_size=grid_size,
                            voxel_res=voxel_res,
                            max_range=env.max_range,
                            subsample=4,
                            device=env.device,
                        ) 
                logger.log(f"体素处理前: {voxel.shape}")
                # 获取观测（字典格式或旧格式）
                obs = env.observe(voxel=voxel, grid_origin=grid_origin, voxel_res=voxel_res)
                if use_nav_style_features and isinstance(obs, dict):
                    # 字典格式：提取state（18维，包含ArUco检测状态）
                    aux = obs['state']  # [num_envs, 18]
                else:
                    # 旧格式：直接使用
                    aux = obs
                
                logger.log(f"辅助观测处理前: {aux.shape}")
            
            # 检查 aux 是否有无效值
            if torch.isnan(aux).any() or torch.isinf(aux).any():
                if t == 0 and it == 1:  # 只在第一次打印
                    nan_count = torch.isnan(aux).sum().item()
                    inf_count = torch.isinf(aux).sum().item()
                    logger.error(f"aux包含无效值: NaN={nan_count}, Inf={inf_count}")
                aux = torch.where(torch.isnan(aux) | torch.isinf(aux), torch.zeros_like(aux), aux)
            
            # 检查 voxel 是否有无效值（仅在非viewer模式下）
            if not env.enable_viewer:
                if torch.isnan(voxel).any() or torch.isinf(voxel).any():
                    if t == 0 and it == 1:  # 只在第一次打印
                        nan_count = torch.isnan(voxel).sum().item()
                        inf_count = torch.isinf(voxel).sum().item()
                        logger.error(f"voxel包含无效值: NaN={nan_count}, Inf={inf_count}")
                    voxel = torch.where(torch.isnan(voxel) | torch.isinf(voxel), torch.zeros_like(voxel), voxel)
            
            # 🎯 调试：保存第一步的voxel和aux（仅在非viewer模式下，避免段错误）
            if not env.enable_viewer and t == 0 and it == 1:
                logger.log(f"voxel形状: {voxel.shape}")
                logger.log(f"aux形状: {aux.shape}")
                logger.save_state("voxel_first_step", voxel, iteration=it)
                logger.save_state("aux_first_step", aux, iteration=it)
            
            # 🎯 统一体素处理：确保形状为 [num_envs, C, Vx, Vy, Vz]
            voxel = prepare_voxel_for_training(voxel, num_envs, voxel_shape)
            if not env.enable_viewer:  # 仅在非viewer模式下记录日志，避免段错误
                logger.log(f"体素处理后: {voxel.shape}")
            
            # 🎯 统一辅助观测处理：确保形状为 [num_envs, aux_dim]
            aux = prepare_aux_for_training(aux, num_envs, aux_dim_val)
            if not env.enable_viewer:  # 仅在非viewer模式下记录日志，避免段错误
                logger.log(f"辅助观测处理后: {aux.shape}")
            
            # 🎯 准备观测字典（如果使用nav_style_features）
            if use_nav_style_features and isinstance(obs, dict):
                obs_dict = obs  # 直接使用字典格式
                voxel_for_ppo = None  # 不使用体素
            else:
                obs_dict = None  # 不使用字典格式
                voxel_for_ppo = voxel  # 使用体素
            
            # 🎯 动作集成策略：传递当前迭代信息
            a, logp, v = algo.act(voxel_for_ppo, aux, current_episode=it, total_episodes=total_iters, obs_dict=obs_dict) # 网络推理得到动作（批量处理）
            
            # 🎯 调试：检查动作和值（仅在非viewer模式下，避免段错误）
            if not env.enable_viewer and t == 0 and it == 1:
                logger.log(f"动作形状: {a.shape}, 范围: [{a.min().item():.4f}, {a.max().item():.4f}]")
                logger.log(f"logp形状: {logp.shape}, 范围: [{logp.min().item():.4f}, {logp.max().item():.4f}]")
                logger.log(f"值形状: {v.shape}, 范围: [{v.min().item():.4f}, {v.max().item():.4f}]")
            
            # 🎯 修复：安全地调用 step() 并解包返回值
            try:
                step_result = env.step(a) # 执行动作，获取done和info（批量）
                
                # 安全解包
                if len(step_result) == 4:
                    _, _, done, info = step_result
                else:
                    print(f"[Error] Step returned {len(step_result)} values, expected 4")
                    raise ValueError(f"Step returned {len(step_result)} values, expected 4")
                
                # 验证返回值类型
                if not isinstance(done, torch.Tensor):
                    print(f"[Warning] done is not a tensor, converting...")
                    done = torch.tensor(done, device=env.device, dtype=torch.bool)
                
                if not isinstance(info, dict):
                    print(f"[Warning] info is not a dict, creating default...")
                    info = {'success': torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
                           'timeout': torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
                           'collision_terminate': torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
                           'success_reason': [None] * env.num_envs,
                           'termination_reason': [None] * env.num_envs}
                
                
                # 🎯 修复：安全地访问和验证 info 字典
                try:
                    if isinstance(info, dict):
                        # 安全地获取 keys（不直接转换列表）
                        try:
                            info_keys = info.keys()
                        except Exception as e:
                            print(f"[Warning] Failed to access info.keys(): {e}")
                        
                        # 确保所有张量都是连续的（在访问之前）
                        try:
                            for key in ['success', 'timeout', 'collision_terminate']:
                                if key in info and isinstance(info[key], torch.Tensor):
                                    # 创建副本而不是就地修改，避免引用问题
                                    tensor_val = info[key]
                                    info[key] = tensor_val.contiguous().clone()
                        except Exception as e:
                            print(f"[Warning] Failed to sanitize info dictionary tensors: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"[Warning] info is not a dict: {type(info)}")
                except Exception as e:
                    print(f"[Error] Failed to access info dictionary: {e}")
                    import traceback
                    traceback.print_exc()
                
            except Exception as e:
                print(f"[Error] Failed in env.step() or unpacking: {e}")
                import traceback
                traceback.print_exc()
                # 返回默认值
                done = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
                info = {
                    'success': torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
                    'timeout': torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
                    'collision_terminate': torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
                    'success_reason': [None] * env.num_envs,
                    'termination_reason': [None] * env.num_envs
                }
            
            # 安全地获取奖励
            try:
                # 🎯 修复：确保输入张量是安全的（连续且在同一设备上）
                if isinstance(a, torch.Tensor):
                    a = a.contiguous()
                    if a.device != env.device:
                        a = a.to(env.device)
                
                # 同步 GPU（如果有）
                if env.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                r, curr_angle = env.compute_rewards(a, prev_angle_deg, cfg, voxel_data=voxel)  # 批量奖励计算
                
                # 再次同步 GPU
                if env.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # 🎯 修复：安全地打印，避免直接访问可能有问题的张量属性
                try:
                    r_type_str = str(type(r))
                    if isinstance(r, torch.Tensor):
                        try:
                            r_shape_str = str(r.shape)
                        except Exception as e:
                            print(f"[Warning] Failed to get r.shape: {e}")
                except Exception as e:
                    print(f"[Warning] Failed to inspect r: {e}")
                
                
                # 确保返回值是正确的类型和形状
                if not isinstance(r, torch.Tensor):
                    r = torch.tensor(r, device=env.device)
                
                if not isinstance(curr_angle, torch.Tensor):
                    curr_angle = torch.tensor(curr_angle, device=env.device)
                
                # 确保形状正确
                try:
                    if r.shape[0] != env.num_envs:
                        r = r[:env.num_envs] if r.shape[0] > env.num_envs else torch.zeros(env.num_envs, device=env.device)
                except Exception as e:
                    print(f"[Error] Failed to check r shape: {e}")
                    r = torch.zeros(env.num_envs, device=env.device)
                
                try:
                    if curr_angle.shape[0] != env.num_envs:
                        curr_angle = curr_angle[:env.num_envs] if curr_angle.shape[0] > env.num_envs else torch.zeros(env.num_envs, device=env.device)
                except Exception as e:
                    print(f"[Error] Failed to check curr_angle shape: {e}")
                    curr_angle = torch.zeros(env.num_envs, device=env.device)
                
            except Exception as e:
                print(f"[Error] Failed to compute rewards: {e}")
                import traceback
                traceback.print_exc()
                # 返回默认奖励
                r = torch.zeros(env.num_envs, device=env.device)
                curr_angle = torch.zeros(env.num_envs, device=env.device)
            
            # 🎯 调试：检查奖励（仅在有异常时打印）
            try:
                if t == 0 and it == 1:
                    try:
                        # 🎯 修复：安全地访问张量属性，避免段错误
                        if isinstance(r, torch.Tensor) and r.numel() > 0:
                            try:
                                # 先同步 GPU，确保数据可用
                                if r.device.type == 'cuda':
                                    torch.cuda.synchronize()
                                r_min = r.min().item()
                                r_max = r.max().item()
                                logger.log(f"奖励形状: {r.shape}, 范围: [{r_min:.4f}, {r_max:.4f}]")
                            except Exception as e:
                                print(f"[Warning] Failed to get r min/max: {e}")
                                logger.log(f"奖励形状: {r.shape if isinstance(r, torch.Tensor) else 'unknown'}")
                        else:
                            logger.log(f"奖励形状: {r.shape if isinstance(r, torch.Tensor) else type(r)}")
                        
                        if isinstance(done, torch.Tensor):
                            logger.log(f"done形状: {done.shape}")
                        else:
                            logger.log(f"done形状: {type(done)}")
                    except Exception as e:
                        print(f"[Warning] Failed in reward logging: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"[Error] Failed in reward logging check: {e}")
                import traceback
                traceback.print_exc()
            
            # 更新可视化（如果启用）
            render_freq = env_cfg.get('render_freq', 1)
            
            # 🎯 临时修复：如果渲染导致段错误，可以选择禁用
            # 设置为 True 可以暂时禁用渲染以验证是否是渲染导致的问题
            SKIP_RENDER_FOR_DEBUG = False  # 改为 True 可以跳过渲染
            
            if not SKIP_RENDER_FOR_DEBUG:
                try:
                    env.render(render_freq=render_freq)
                except Exception as e:
                    print(f"[Warning] env.render() failed: {e}")
                    import traceback
                    traceback.print_exc()

            # 🎯 批量存储数据（优化：直接批量赋值，无需循环）
            if use_nav_style_features:
                # 字典格式：存储完整的obs_dict
                if isinstance(obs, dict):
                    obs_dicts.append(obs)  # 存储字典格式观测
                # 仍然存储voxel和aux用于向后兼容（但PPO不会使用）
                voxels[t] = voxel  # [num_envs, C, Vx, Vy, Vz]
                auxes[t] = aux     # [num_envs, aux_dim]
            else:
                # 旧格式：直接存储voxel和aux
                voxels[t] = voxel  # [num_envs, C, Vx, Vy, Vz] -> 直接赋值
                auxes[t] = aux     # [num_envs, aux_dim]
            
            # 确保动作形状正确
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if a.shape[0] == num_envs:
                acts[t] = a  # [num_envs, action_dim]
            else:
                acts[t, 0] = a[0] if a.shape[0] > 0 else a
            
            # 批量处理 logp, v, r - 确保形状正确
            if logp.dim() == 0:
                logp = logp.unsqueeze(0)
            if logp.shape[0] == num_envs:
                logps[t] = logp
                # 🎯 计算动作概率（用于自适应折扣因子）
                # 对于连续动作空间，将logp转换为"策略质量"指标
                # 方法：使用sigmoid将logp映射到[0,1]，作为策略质量指标
                # 注意：GAE内部会将此值clip到[eta_min, eta_max]作为折扣因子
                if use_adaptive_gamma:
                    # 将logp归一化到[0,1]范围（使用sigmoid映射）
                    # logp通常在[-10, 0]范围，sigmoid(logp/5.0)将其映射到[0,1]
                    logp_normalized = torch.sigmoid(logp / 5.0)  # [0,1]范围，表示策略质量
                    action_probs[t] = logp_normalized
            else:
                # 处理logp形状不匹配的情况
                if logp.shape[0] > 0:
                    logps[t, 0] = logp[0]
                    if use_adaptive_gamma and action_probs is not None:
                        action_probs[t, 0] = torch.sigmoid(logp[0] / 5.0)
                else:
                    # 如果logp为空，使用0作为默认值
                    logps[t, 0] = 0.0
                    if use_adaptive_gamma and action_probs is not None:
                        action_probs[t, 0] = 0.5  # sigmoid(0/5.0) = 0.5
            
            if v.dim() == 0:
                v = v.unsqueeze(0)
            if v.shape[0] == num_envs:
                vals[t] = v
                vals_den[t] = algo.value_norm.denormalize(v.unsqueeze(-1)).squeeze(-1)
            else:
                vals[t, 0] = v[0] if v.shape[0] > 0 else v
                vals_den[t, 0] = algo.value_norm.denormalize(v.unsqueeze(-1)).squeeze(-1) if v.dim() > 0 else algo.value_norm.denormalize(v.unsqueeze(-1)).squeeze(-1)
            
            # 批量处理奖励
            if r.dim() == 0:
                r = r.unsqueeze(0)
            if r.shape[0] == num_envs:
                rews[t] = r
                episode_rewards += r  # 批量累加
            else:
                rews[t, 0] = r[0].item() if r.shape[0] > 0 else r.item()
                episode_rewards[0] += r[0] if r.shape[0] > 0 else r
            
            # 批量处理 done
            if isinstance(done, torch.Tensor):
                if done.dim() == 0:
                    done = done.unsqueeze(0)
                if done.shape[0] == num_envs:
                    dones_list[t] = done  # 批量赋值
                else:
                    dones_list[t, 0] = done[0].item() if done.shape[0] > 0 else done.item()
            else:
                dones_list[t, 0] = bool(done)
            
            # 批量处理 episode 长度
            episode_lengths += 1
            
            # 🎯 改进的episode结束处理（准确统计）
            if isinstance(done, torch.Tensor) and done.shape[0] == num_envs:
                done_mask = done.bool()
                if done_mask.any():
                    done_envs = torch.where(done_mask)[0].cpu().tolist()
                    
                    # 🎯 关键修复：只在环境首次进入done状态时计数，避免重复计数
                    # 使用一个标志来跟踪哪些环境已经在这个rollout中被计数过
                    if not hasattr(main, '_counted_envs_in_rollout'):
                        main._counted_envs_in_rollout = set()
                    
                    for env_idx in done_envs:
                        # 检查该环境是否已经在这个rollout中被计数过
                        env_key = (it, env_idx)
                        if env_key not in main._counted_envs_in_rollout:
                            # 首次计数，增加episode_count
                            episode_count += 1
                            main._counted_envs_in_rollout.add(env_key)
                            
                            # 统计、打印、保存episode等
                            # 获取episode结束原因
                            success = info.get('success', False) if info else False
                            timeout = info.get('timeout', False) if info else False
                            collision_terminate = info.get('collision_terminate', False) if info else False
                            
                            # 处理success（可能是tensor或列表）
                            if isinstance(success, torch.Tensor):
                                success_val = success[env_idx].item() if success.shape[0] > env_idx else False
                            elif isinstance(success, list):
                                success_val = success[env_idx] if len(success) > env_idx else False
                            else:
                                success_val = success
                            
                            # 统计成功次数
                            if success_val:
                                success_count += 1
                            
                            # 打印episode结束信息（总是打印，因为这是episode结束）
                            termination_reason = info.get('termination_reason', None) if info else None
                            success_reason = info.get('success_reason', None) if info else None
                            
                            # 处理termination_reason
                            if isinstance(termination_reason, list) and len(termination_reason) > env_idx:
                                term_reason = termination_reason[env_idx]
                            elif isinstance(termination_reason, list):
                                term_reason = termination_reason[0] if len(termination_reason) > 0 else None
                            else:
                                term_reason = termination_reason
                            
                            # 处理success_reason（优先使用success_reason，如果没有则使用termination_reason）
                            if success_val:
                                if isinstance(success_reason, list) and len(success_reason) > env_idx:
                                    succ_reason = success_reason[env_idx]
                                elif isinstance(success_reason, list):
                                    succ_reason = success_reason[0] if len(success_reason) > 0 else None
                                else:
                                    succ_reason = success_reason
                                # 如果success_reason为空，使用termination_reason
                                if succ_reason is None:
                                    succ_reason = term_reason
                            else:
                                succ_reason = None
                            
                            # 构建原因字符串
                            if success_val:
                                reason_str = f" (成功: {succ_reason})" if succ_reason else " (成功)"
                            elif isinstance(collision_terminate, torch.Tensor) and collision_terminate.shape[0] > env_idx and collision_terminate[env_idx]:
                                reason_str = f" (碰撞终止: {term_reason})" if term_reason else " (碰撞终止)"
                            elif isinstance(timeout, torch.Tensor) and timeout.shape[0] > env_idx and timeout[env_idx]:
                                reason_str = " (超时)"
                            else:
                                reason_str = ""
                            
                            print(f"  -> Env {env_idx} Episode finished at step {t+1}, reward={episode_rewards[env_idx].item():.2f}, length={episode_lengths[env_idx].item()}{reason_str}")
                            
                            # 🎯 保存episode统计
                            logger.save_episode(
                                iteration=it,
                                step=t+1,
                                env_idx=env_idx,
                                episode_reward=episode_rewards[env_idx].item(),
                                episode_length=episode_lengths[env_idx].item(),
                                success=success_val,
                                termination_reason=str(term_reason) if term_reason else None,
                                success_reason=str(succ_reason) if succ_reason else None
                            )
                            
                            # 重置该环境的统计
                            episode_rewards[env_idx] = 0.0
                            episode_lengths[env_idx] = 0
                    
                    # 🎯 修复：为每个done的环境单独reset，而不是等所有环境都done
                    # 这样可以避免done的环境一直处于done状态，导致重复计数
                    if done_mask.any():
                        done_envs = torch.where(done_mask)[0].cpu().tolist()
                        if hasattr(env, 'reset_envs'):
                            # 使用批量reset_envs方法（如果可用）
                            env.reset_envs(done_envs)
                        else:
                            # 如果没有reset_envs方法，使用完整的reset（会影响所有环境）
                            # 这是降级方案，理想情况下应该实现reset_envs
                            if done_mask.all():
                                env.reset()
                        if use_incremental_voxel:
                            global_voxel = torch.zeros((1, 1, Vx, Vy, Vz), device=env.device)
            elif done and num_envs == 1:
                episode_count += 1
                env.reset()
                if use_incremental_voxel:
                    global_voxel = torch.zeros((1, 1, Vx, Vy, Vz), device=env.device)
                episode_rewards.zero_()
                episode_lengths.zero_()

            prev_angle_deg = curr_angle.detach()  # 批量处理：curr_angle 应该是 [num_envs]

        # 🎯 计算最后一个状态的体素和价值（用于GAE）
        with torch.no_grad():
            # 🎯 当使用LiDAR时，跳过相机和体素操作
            if env.enable_viewer or use_nav_style_features:
                # 使用空体素地图（用于向后兼容，但PPO不会使用）
                Nx, Ny, Nz = map(int, grid_size.tolist())
                voxel = torch.zeros((1, 1, Nx, Ny, Nz), device=env.device)
                last_voxel = prepare_voxel_for_training(voxel, num_envs, voxel_shape)
            else:
                # 不启用viewer且不使用LiDAR时，正常执行所有操作（使用体素地图）
                # 🎯 最后一次rollout步骤，总是渲染（不使用频率控制）
                depth = env.render_depth()
                cam_T_w = env.camera_pose()
                intr = env.intrinsics
                
                # 计算最后一个状态的体素
                if use_incremental_voxel and global_voxel is not None:
                    global_voxel = update_voxel_from_depth(
                        global_voxel=global_voxel,
                        depth=depth,
                        cam_T_w=cam_T_w,
                        intrinsics=intr,
                        grid_origin=grid_origin,
                        grid_size=grid_size,
                        voxel_res=voxel_res,
                        max_range=env.max_range,
                        subsample=4,
                        decay_factor=cfg.get('voxel', {}).get('decay_factor', 0.95),
                        device=env.device,
                    )
                    voxel = global_voxel.clone()
                else:
                    voxel = build_voxel_from_depth(
                        depth=depth,
                        cam_T_w=cam_T_w,
                        intrinsics=intr,
                        grid_origin=grid_origin,
                        grid_size=grid_size,
                        voxel_res=voxel_res,
                        max_range=env.max_range,
                        subsample=4,
                        device=env.device,
                    )
                # 🎯 多环境支持：将 voxel 扩展到所有环境
                last_voxel = prepare_voxel_for_training(voxel, num_envs, voxel_shape)
            
            # 获取最后一步的观测
            last_obs = env.observe(voxel=last_voxel, grid_origin=grid_origin, voxel_res=voxel_res)
            if use_nav_style_features and isinstance(last_obs, dict):
                # 字典格式：提取state（18维，包含ArUco检测状态）
                last_aux = last_obs['state']  # [num_envs, 18]
                last_obs_dict = last_obs
                last_voxel_for_ppo = None
            else:
                # 旧格式：直接使用
                last_aux = last_obs
                last_obs_dict = None
                last_voxel_for_ppo = last_voxel
            
            _, _, last_values = algo.act(last_voxel_for_ppo, last_aux, current_episode=it, total_episodes=total_iters, obs_dict=last_obs_dict)  # last_values: [num_envs]
            
            # 🎯 确保 last_values 形状正确
            if last_values.dim() == 0:
                last_values = last_values.unsqueeze(0)
            if last_values.shape[0] != num_envs:
                if last_values.shape[0] == 1:
                    last_values = last_values.expand(num_envs)
                else:
                    # 如果形状不匹配，扩展或截断
                    if last_values.shape[0] < num_envs:
                        last_values = last_values.repeat((num_envs // last_values.shape[0] + 1))[:num_envs]
                    else:
                        last_values = last_values[:num_envs]

        # 数据已经是[T,N]格式，直接使用
        rewards = rews      # [T,N]
        values = vals       # [T,N]
        dones = dones_list  # [T,N]
        
        # 🎯 next_values 计算（参考 isaac-training）
        # next_values[t] = values[t+1] for t < T-1
        # next_values[T-1] = last_v (最后一个状态的价值估计)
        next_values = torch.zeros_like(values)  # [T,N]
        next_values[:-1] = values[1:]  # [T-1, N] = values[1:T]
        next_values[-1] = last_values  # [N] = last_v for all environments at final step
        
        # 🎯 自适应折扣因子：传递动作概率
        if use_adaptive_gamma and action_probs is not None:
            adv, ret = gae(rewards, dones, values, next_values, action_probs=action_probs)
        else:
            adv, ret = gae(rewards, dones, values, next_values)

        traj = {
            'voxel': voxels.view(-1, *voxel_shape),  # [T*N, C, Vx, Vy, Vz]
            'aux': auxes.view(-1, aux_dim_val),      # [T*N, aux_dim]
            'act': acts.view(-1, action_dim_val),    # [T*N, action_dim]
            'logp': logps.flatten(),                 # [T*N]
            'adv': adv.flatten(),                    # [T*N]
            'ret': ret.flatten(),                    # [T*N]
            'val_den': vals_den.flatten(),           # [T*N]
        }
        
        # 🎯 如果使用nav_style_features，添加obs_dict到traj
        if use_nav_style_features and obs_dicts is not None and len(obs_dicts) > 0:
            # 将obs_dicts列表转换为字典格式的tensor
            # obs_dicts是长度为T的列表，每个元素是字典 {state, lidar}（已禁用 dynamic_obstacle）
            # 🎯 修复：确保tensor可以用于反向传播（从推理模式转换）
            traj['obs_dict'] = {
                'state': torch.stack([obs['state'] for obs in obs_dicts], dim=0).view(-1, aux_dim_val).clone().detach().requires_grad_(False),  # [T*N, state_dim]
                'lidar': torch.stack([obs['lidar'] for obs in obs_dicts], dim=0).view(-1, *obs_dicts[0]['lidar'].shape[1:]).clone().detach().requires_grad_(False),  # [T*N, 1, h_beams, v_beams]
                # 🎯 已禁用动态障碍物：不再包含 dynamic_obstacle
            }
        # 训练更新
        print(f"[Training] Updating policy (iter {it}/{total_iters})...")
        logger.log(f"[Training] 开始更新策略 (iter {it}/{total_iters})")
        
        try:
            stats = algo.update(traj)
            logger.log(f"[Training] 策略更新完成")
        except Exception as e:
            logger.error(f"策略更新失败 (iter {it})", exception=e)
            raise
        
        # 计算 rollout 统计
        # 注意：dones_list已经在episode结束时记录了True，现在只需要统计True的数量
        total_episodes = dones_list.sum().item()
        mean_reward = rews.mean().item()
        mean_value = vals.mean().item()
        mean_advantage = traj['adv'].mean().item()
        std_advantage = traj['adv'].std().item()
        
        # 🎯 改进的统计指标
        success_rate = success_count / max(episode_count, 1) if episode_count > 0 else 0.0
        mean_episode_length = episode_lengths[episode_lengths > 0].float().mean().item() if episode_count > 0 else 0.0
        
        # 🎯 准备统计字典并保存
        stats_dict = {
            'actor_loss': stats['actor_loss'],
            'critic_loss': stats['critic_loss'],
            'entropy': stats['entropy'],
            'approx_kl': stats['approx_kl'],
            'clip_fraction': stats['clip_fraction'],
            'value_clip_frac': stats.get('value_clip_frac', 0.0),
            'mean_reward': mean_reward,
            'std_reward': rews.std().item(),
            'min_reward': rews.min().item(),
            'max_reward': rews.max().item(),
            'mean_value': mean_value,
            'std_value': vals.std().item(),
            'mean_advantage': mean_advantage,
            'std_advantage': std_advantage,
            'episode_count': episode_count,
            'success_count': success_count,
            'success_rate': success_rate,
            'mean_episode_length': mean_episode_length,
            'grad_norm_actor': stats.get('grad_norm_actor', 0.0),
            'grad_norm_critic': stats.get('grad_norm_critic', 0.0),
        }
        logger.save_stats(it, stats_dict)
        
        # 🎯 调试：保存每10次迭代的中间状态
        if it % 16 == 0:
            logger.save_state("rewards", rews, iteration=it)
            logger.save_state("values", vals, iteration=it)
            logger.save_state("advantages", adv, iteration=it)
            logger.log(f"中间状态已保存 (iter {it})")
        
        # 注意：环境重置已经在done=True时在rollout循环中完成，这里不需要再次重置

        # 详细训练日志
        if it % cfg['train']['log_interval'] == 0:
            logger.log("\n" + "="*80)
            logger.log(f"[Training] Iteration {it}/{total_iters}")
            logger.log(f"  Losses:     actor={stats['actor_loss']:.6f} | critic={stats['critic_loss']:.6f} | entropy={stats['entropy']:.6f}")
            logger.log(f"  Metrics:    KL={stats['approx_kl']:.6f} | clip_frac={stats['clip_fraction']:.4f} | value_clip_frac={stats.get('value_clip_frac', 0.0):.4f}")
            logger.log(f"  Rewards:    mean={mean_reward:.4f} | std={rews.std().item():.4f} | min={rews.min().item():.4f} | max={rews.max().item():.4f}")
            logger.log(f"  Values:     mean={mean_value:.4f} | std={vals.std().item():.4f}")
            logger.log(f"  Advantages: mean={mean_advantage:.4f} | std={std_advantage:.4f}")
            logger.log(f"  Episodes:   completed={episode_count} | success_rate={success_rate:.2%} | mean_length={mean_episode_length:.1f}")
            if 'grad_norm_actor' in stats:
                logger.log(f"  Gradients:  actor={stats['grad_norm_actor']:.4f} | critic={stats['grad_norm_critic']:.4f}")
            logger.log("="*80 + "\n")
            
            print("\n" + "="*80)
            print(f"[Training] Iteration {it}/{total_iters}")
            print(f"  Losses:     actor={stats['actor_loss']:.6f} | critic={stats['critic_loss']:.6f} | entropy={stats['entropy']:.6f}")
            print(f"  Metrics:    KL={stats['approx_kl']:.6f} | clip_frac={stats['clip_fraction']:.4f} | value_clip_frac={stats.get('value_clip_frac', 0.0):.4f}")
            print(f"  Rewards:    mean={mean_reward:.4f} | std={rews.std().item():.4f} | min={rews.min().item():.4f} | max={rews.max().item():.4f}")
            print(f"  Values:     mean={mean_value:.4f} | std={vals.std().item():.4f}")
            print(f"  Advantages: mean={mean_advantage:.4f} | std={std_advantage:.4f}")
            print(f"  Episodes:   completed={episode_count} | success_rate={success_rate:.2%} | mean_length={mean_episode_length:.1f}")
            if 'grad_norm_actor' in stats:
                print(f"  Gradients:  actor={stats['grad_norm_actor']:.4f} | critic={stats['grad_norm_critic']:.4f}")
            print("="*80 + "\n")
        else:
            # 简短进度显示（包含改进的统计）
            print(f"[Training] iter {it}/{total_iters} | loss_a={stats['actor_loss']:.4f} | loss_c={stats['critic_loss']:.4f} | reward={mean_reward:.2f} | success_rate={success_rate:.2%} | episodes={episode_count}")
            logger.log(f"[Training] iter {it}/{total_iters} | loss_a={stats['actor_loss']:.4f} | loss_c={stats['critic_loss']:.4f} | reward={mean_reward:.2f} | success_rate={success_rate:.2%} | episodes={episode_count}")

        if it % cfg['train']['save_interval'] == 0:
            os.makedirs(cfg['train']['checkpoint_dir'], exist_ok=True)
            checkpoint = {
                'actor': algo.actor.state_dict(),
                'critic': algo.critic.state_dict(),
                'iteration': it,
            }
            # 🎯 根据use_nav_style_features保存不同的特征提取器
            if not use_nav_style_features:
                checkpoint['backbone'] = algo.backbone.state_dict()
            else:
                checkpoint['lidar_cnn'] = algo.lidar_cnn.state_dict()
                checkpoint['dyn_obs_mlp'] = algo.dyn_obs_mlp.state_dict()
            checkpoint_path = os.path.join(cfg['train']['checkpoint_dir'], f'policy_{it}.pt')
            torch.save(checkpoint, checkpoint_path)
            logger.log(f"模型已保存: {checkpoint_path}")
            print(f"[Training] Model saved: {checkpoint_path}")

    # 保存最终模型
    os.makedirs(cfg['train']['checkpoint_dir'], exist_ok=True)
    final_checkpoint = {
        'actor': algo.actor.state_dict(),
        'critic': algo.critic.state_dict(),
        'iteration': total_iters,
    }
    # 🎯 根据use_nav_style_features保存不同的特征提取器
    if not use_nav_style_features:
        final_checkpoint['backbone'] = algo.backbone.state_dict()
    else:
        final_checkpoint['lidar_cnn'] = algo.lidar_cnn.state_dict()
        final_checkpoint['dyn_obs_mlp'] = algo.dyn_obs_mlp.state_dict()
    final_checkpoint_path = os.path.join(cfg['train']['checkpoint_dir'], 'policy_final.pt')
    torch.save(final_checkpoint, final_checkpoint_path)
    logger.log(f"最终模型已保存: {final_checkpoint_path}")
    
    logger.log("="*80)
    logger.log("训练完成！")
    logger.log(f"所有调试信息已保存到: {debug_dir}")
    logger.log("="*80)
    print(f"\n[Training] 训练完成！调试信息保存在: {debug_dir}")


if __name__ == '__main__':
    main()