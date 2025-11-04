"""
PPO (Proximal Policy Optimization) Implementation - Enhanced Version
增强版 PPO 算法实现
参考 isaac-training 的实现，添加了：
- LiDAR 的 CNN 特征提取
- 分离的优化器
- 梯度裁剪
- Value Normalization
- Critic Loss Clipping
"""
# IMPORTANT: Isaac Gym must be imported before PyTorch
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from typing import Union, Iterable

# 导入动作空间变换工具
from action_transform import ActionTransformer, vec_to_world, vec_to_new_frame


class ValueNorm(nn.Module):
    """
    价值归一化器 - 改进版
    使用指数移动平均（EMA）和偏差校正
    """
    
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,  # 意味着新的批次数据权重为0.5%，历史数据权重为99.5%
        epsilon=1e-5,  # 防止除零错误的小常数，数值稳定性常数
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        # 注册缓冲区（不会被视为模型参数，但会保存到状态字典）
        self.running_mean: torch.Tensor 
        self.running_mean_sq: torch.Tensor 
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        """重置统计量"""
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        """
        偏差校正的均值和方差
        
        在训练初期，运行统计量基于很少的样本，会偏向初始值0
        解决方案：通过偏差校正得到无偏估计
        """
        # 计算偏差校正的均值
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)

        # 计算偏差校正的平方均值
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)

        # 计算偏差校正的方差：Var[X] = E[X²] - E[X]²
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        """更新运行统计量（指数移动平均）"""
        assert input_vector.shape[-len(self.input_shape):] == self.input_shape
        
        # 计算需要平均的维度
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        
        # 计算当前批次的统计量
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        # 指数移动平均更新
        # running_mean = β × running_mean + (1-β) × batch_mean
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        
        # 偏差校正项：d_t = β × d_{t-1} + (1 - β) × 1 → d_t = 1 - β^t
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        """归一化输入"""
        assert input_vector.shape[-len(self.input_shape):] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        """反归一化输入"""
        assert input_vector.shape[-len(self.input_shape):] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out


class LiDARFeatureExtractor(nn.Module):
    """LiDAR CNN 特征提取器"""
    
    def __init__(self, lidar_h_beams, lidar_v_beams):
        super().__init__()
        
        # CNN 层处理 LiDAR 数据 [batch, 1, h_beams, v_beams]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 3), padding=(2, 1)),
            nn.ELU(),
            # stride 步长 
            # padding 填充
            nn.Conv2d(8, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ELU(),

            # 展平层
            nn.Flatten(),
        )
        
        # 计算输出维度
        # 创建一个临时环境 在此环境中的所有操作都不会计算梯度
        # 节省内存：不保存前向传播的中间结果
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, lidar_h_beams, lidar_v_beams)
            cnn_output_dim = self.cnn(dummy_input).shape[1]
        
        # 投影层
        # 投影层​​是一个将数据从​​高维空间映射到低维空间​​的神经网络层，目的是提取更紧凑、更有意义的特征表示
        self.proj = nn.Sequential(
            nn.Linear(cnn_output_dim, 128),
            nn.LayerNorm(128),
            nn.ELU()
        )
    
    def forward(self, lidar_data):
        # lidar_data: [batch, h_beams * v_beams]
        # reshape 为 [batch, 1, h_beams, v_beams]
        batch_size = lidar_data.shape[0]
        h_beams = 36  # 从配置推断
        v_beams = 4
        
        lidar_reshaped = lidar_data.reshape(batch_size, 1, h_beams, v_beams)
        features = self.cnn(lidar_reshaped)
        features = self.proj(features)
        
        return features


class ActorCritic(nn.Module):
    """PPO Actor-Critic 网络 - 增强版"""
    
    def __init__(self, obs_dim, action_dim, lidar_h_beams, lidar_v_beams, hidden_size=256):
        super().__init__()
        
        self.lidar_dim = lidar_h_beams * lidar_v_beams
        self.state_dim = obs_dim - self.lidar_dim  # 基础状态维度（不包括 LiDAR）
        
        # LiDAR 特征提取器
        self.lidar_extractor = LiDARFeatureExtractor(lidar_h_beams, lidar_v_beams)
        
        # 状态特征提取器
        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.LayerNorm(128),
            nn.ELU(),
        )
        
        # 融合后的共享网络
        self.shared_net = nn.Sequential(
            nn.Linear(256, hidden_size),  # 128 (lidar) + 128 (state) = 256
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
        )
        
        # Actor (策略网络) - 使用 Beta 分布
        # 输出 alpha 和 beta 参数（必须 > 0）
        self.actor_alpha = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softplus()  # 确保 alpha > 0
        )
        
        self.actor_beta = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softplus()  # 确保 beta > 0
        )
        
        # Critic (价值网络)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """正交初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, 0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)
    
    def forward(self, obs):
        # 分离 LiDAR 和状态
        state = obs[:, :self.state_dim]
        lidar = obs[:, self.state_dim:]
        
        # 特征提取
        state_features = self.state_mlp(state)
        lidar_features = self.lidar_extractor(lidar)
        
        # 融合特征
        combined_features = torch.cat([state_features, lidar_features], dim=1)
        features = self.shared_net(combined_features)
        
        # Actor - 输出 Beta 分布的参数
        alpha = self.actor_alpha(features) + 1.0  # alpha >= 1
        beta = self.actor_beta(features) + 1.0    # beta >= 1
        
        # Critic
        value = self.critic(features)
        
        return alpha, beta, value
    
    def get_action(self, obs):
        """
        采样动作
        Beta 分布输出 [0, 1]，需要转换到 [-1, 1]
        """
        alpha, beta, value = self.forward(obs)
        
        # Beta 分布（输出范围 [0, 1]）
        dist = Beta(alpha, beta)
        action_normalized = dist.sample()  # [0, 1]
        
        # 转换到 [-1, 1]
        action = 2.0 * action_normalized - 1.0
        
        # 计算对数概率（注意要考虑变换的雅可比）
        action_log_prob = dist.log_prob(action_normalized).sum(dim=1)
        
        return action, action_log_prob, value
    
    def evaluate(self, obs, action):
        """
        评估动作的概率和价值
        action 范围是 [-1, 1]，需要转换回 [0, 1] 来评估 Beta 分布
        """
        alpha, beta, value = self.forward(obs)
        
        # Beta 分布
        dist = Beta(alpha, beta)
        
        # 将 action 从 [-1, 1] 转换回 [0, 1]
        action_normalized = (action + 1.0) / 2.0
        action_normalized = torch.clamp(action_normalized, 1e-6, 1.0 - 1e-6)  # 避免边界问题
        
        # 计算对数概率
        action_log_prob = dist.log_prob(action_normalized).sum(dim=1)
        
        # 计算熵
        entropy = dist.entropy().sum(dim=1)
        
        return action_log_prob, value, entropy


class PPOTrainer:
    """PPO 训练器 - 增强版"""
    
    def __init__(self, env, config):
        self.config = config
        self.env = env
        
        # 设备
        self.device = torch.device(config.device)
        
        # 观测和动作维度
        # 基础观测: rel_pos(3) + distance(1) + target_found(1) + vel(6) + joints(num_dofs)
        # LiDAR: h_beams * v_beams (例如 36 * 4 = 144)
        lidar_dim = self.env.lidar_hbeams * self.env.lidar_vbeams
        obs_dim = 3 + 1 + 1 + 6 + self.env.num_dofs + lidar_dim  # +1 for target_found flag
        action_dim = 6  # [vx, vy, vz, wx, wy, wz]
        
        print(f"[PPOTrainer] 观测维度: {obs_dim} (基础: 17, LiDAR: {lidar_dim})")
        print(f"[PPOTrainer] 动作维度: {action_dim}")
        print(f"[PPOTrainer] 两阶段任务：探索 + 导航")
        
        # 创建网络（带 LiDAR CNN）
        print("[PPOTrainer] 创建增强版网络...")
        self.ac_network = ActorCritic(
            obs_dim, action_dim, 
            self.env.lidar_hbeams, 
            self.env.lidar_vbeams,
            config.hidden_size
        ).to(self.device)
        
        # 分离的优化器（参考 isaac-training）
        feature_lr = config.lr if not hasattr(config, 'feature_lr') else config.feature_lr
        actor_lr = config.lr if not hasattr(config, 'actor_lr') else config.actor_lr
        critic_lr = config.lr if not hasattr(config, 'critic_lr') else config.critic_lr
        
        self.feature_optimizer = optim.Adam([
            {'params': self.ac_network.lidar_extractor.parameters()},
            {'params': self.ac_network.state_mlp.parameters()},
            {'params': self.ac_network.shared_net.parameters()},
        ], lr=feature_lr)
        
        self.actor_optimizer = optim.Adam(
            list(self.ac_network.actor_alpha.parameters()) + list(self.ac_network.actor_beta.parameters()),
            lr=actor_lr
        )
        
        self.critic_optimizer = optim.Adam(
            self.ac_network.critic.parameters(),
            lr=critic_lr
        )
        
        # 训练参数
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.ppo_epochs = config.ppo_epochs
        self.batch_size = config.batch_size
        # PPO Clipping 参数（支持分离配置）
        self.actor_clip_ratio = config.actor_clip_ratio if hasattr(config, 'actor_clip_ratio') else config.clip_epsilon
        self.critic_clip_ratio = config.critic_clip_ratio if hasattr(config, 'critic_clip_ratio') else config.clip_epsilon
        self.clip_epsilon = config.clip_epsilon  # 向后兼容
        
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        
        print(f"[PPOTrainer] Actor Clip Ratio: {self.actor_clip_ratio}")
        print(f"[PPOTrainer] Critic Clip Ratio: {self.critic_clip_ratio}")
        
        # 梯度裁剪
        self.max_grad_norm = config.max_grad_norm if hasattr(config, 'max_grad_norm') else 5.0
        
        # Value Normalization（参考 isaac-training）
        self. use_value_norm = config.use_value_norm if hasattr(config, 'use_value_norm') else True
        if self.use_value_norm:
            value_norm_beta = config.value_norm_beta if hasattr(config, 'value_norm_beta') else 0.995
            self.value_norm = ValueNorm(1, beta=value_norm_beta).to(self.device)
            print(f"[PPOTrainer] Value Normalization 已启用 (beta={value_norm_beta})")
        
        # Critic Loss 类型
        self.use_huber_loss = config.use_huber_loss if hasattr(config, 'use_huber_loss') else True
        if self.use_huber_loss:
            self.critic_loss_fn = nn.HuberLoss(delta=10.0)
            print("[PPOTrainer] 使用 Huber Loss for Critic")
        else:
            self.critic_loss_fn = nn.MSELoss()
        
        # 统计信息
        self.episode_rewards = []
        self.episode_lengths = []
        self.discovery_times = []  # 发现目标的时间
        
        # 当前 episode 的累计信息
        self.current_episode_reward = torch.zeros(self.env.num_envs, device=self.device)
        self.current_episode_length = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        
        # 训练统计
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.grad_norms = []
        
        # 动作空间变换
        self.use_action_transform = config.use_action_transform if hasattr(config, 'use_action_transform') else False
        self.action_coordinate = config.action_coordinate if hasattr(config, 'action_coordinate') else 'world'
        self.action_limit = config.action_limit if hasattr(config, 'action_limit') else 1.0
        
        if self.use_action_transform:
            self.action_transformer = ActionTransformer(
                coordinate_type=self.action_coordinate,
                device=self.device
            )
            print(f"[PPOTrainer] 动作空间变换已启用: {self.action_coordinate} 坐标系")
        
        # 日志和调试配置
        self.log_interval = config.log_interval if hasattr(config, 'log_interval') else 10
        self.save_interval = config.save_interval if hasattr(config, 'save_interval') else 1000
        self.checkpoint_dir = config.checkpoint_dir if hasattr(config, 'checkpoint_dir') else "checkpoints"
        
        # 获取训练目标配置
        if hasattr(config, 'training') and hasattr(config.training, 'total_iterations'):
            self.target_iterations = config.training.total_iterations
        else:
            self.target_iterations = None  # 由 num_steps 决定
            
        print("[PPOTrainer] 增强版初始化完成")
        print(f"[PPOTrainer] 检查点保存目录: {self.checkpoint_dir}")
    
    def save_checkpoint(self, iteration, step, filepath=None, is_final=False):
        """保存训练检查点"""
        import os
        
        if filepath is None:
            # 确保检查点目录存在
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            if is_final:
                filepath = os.path.join(self.checkpoint_dir, f"ppo_final_iter_{iteration}_step_{step}.pt")
            else:
                filepath = os.path.join(self.checkpoint_dir, f"ppo_iter_{iteration}_step_{step}.pt")
        else:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'iteration': iteration,
            'step': step,
            'ac_network_state': self.ac_network.state_dict(),
            'feature_optimizer_state': self.feature_optimizer.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'discovery_times': self.discovery_times,
        }
        
        if self.use_value_norm:
            checkpoint['value_norm_state'] = self.value_norm.state_dict()
        
        torch.save(checkpoint, filepath)
        
        # 获取绝对路径并打印
        abs_filepath = os.path.abspath(filepath)
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"[Checkpoint] 已保存到: {abs_filepath} ({file_size:.2f} MB)")
    
    def train(self, num_steps):
        """训练主循环（增强版，带详细调试信息）"""
        # 确定训练目标（迭代次数或步数）
        if self.target_iterations is not None:
            # 如果配置了总迭代次数，则基于迭代次数来计算步数
            total_iterations = self.target_iterations
            num_steps = total_iterations * self.config.rollout_steps * self.env.num_envs
            training_mode = "iterations"
        else:
            # 否则基于步数来确定迭代次数
            total_iterations = num_steps // (self.config.rollout_steps * self.env.num_envs)
            training_mode = "steps"
        
        print("=" * 80)
        print(f"[PPOTrainer] 开始训练")
        print(f"  训练模式: {training_mode}")
        print(f"  总步数目标: {num_steps:,}")
        print(f"  总迭代次数: {total_iterations:,}")
        print(f"  Rollout 步数: {self.config.rollout_steps}")
        print(f"  环境数: {self.env.num_envs}")
        print(f"  每轮步数: {self.config.rollout_steps * self.env.num_envs}")
        print(f"  每 {self.log_interval} 次迭代打印日志")
        print(f"  每 {self.save_interval} 次迭代保存模型")
        print("=" * 80)
        
        obs = self.env.reset()
        obs = obs.to(self.device)  # 确保 obs 在 GPU 上
        step = 0
        iteration = 0  # 迭代计数器
        
        # 训练循环条件：基于迭代次数或步数
        should_continue = lambda: (training_mode == "iterations" and iteration < total_iterations) or \
                                   (training_mode == "steps" and step < num_steps)
        
        while should_continue():
            iteration += 1
            # 收集一批经验
            batch_obs, batch_actions, batch_rewards, batch_values, batch_log_probs, batch_dones = \
                self.collect_rollout(obs, self.config.rollout_steps)
            
            # 计算优势函数和回报（带 done 掩码）
            advantages, returns = self.compute_gae(batch_rewards, batch_values, batch_dones)
            
            # 归一化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Value Normalization
            if self.use_value_norm:
                # ValueNorm 期望 [batch, 1] 形状
                returns_reshaped = returns.unsqueeze(-1)  # [batch] -> [batch, 1]
                self.value_norm.update(returns_reshaped)
                returns_normalized = self.value_norm.normalize(returns_reshaped).squeeze(-1)  # [batch, 1] -> [batch]
            else:
                returns_normalized = returns
            
            # PPO 更新（多轮）
            update_info = {
                'actor_loss': 0,
                'critic_loss': 0,
                'entropy_loss': 0,
                'entropy': 0,  # 熵值（非损失）
                'actor_grad_norm': 0,
                'critic_grad_norm': 0,
                'feature_grad_norm': 0,
                'explained_var': 0,
                'approx_kl': 0,
                'clip_fraction': 0,
            }
            
            num_updates = 0
            for epoch in range(self.ppo_epochs):
                # 随机排列
                indices = torch.randperm(batch_obs.shape[0], device=self.device)
                
                # Mini-batch 更新
                for start in range(0, batch_obs.shape[0], self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    info = self.update(
                        batch_obs[batch_indices], 
                        batch_actions[batch_indices],
                        batch_log_probs[batch_indices],
                        returns_normalized[batch_indices],
                        batch_values[batch_indices],
                        advantages[batch_indices]
                    )
                    
                    # 累加统计信息
                    for key in update_info:
                        update_info[key] += info[key]
                    num_updates += 1
            
            # 平均统计信息
            for key in update_info:
                update_info[key] /= num_updates
            
            step += self.config.rollout_steps * self.env.num_envs
            
            # 保存检查点
            if iteration % self.save_interval == 0:
                self.save_checkpoint(iteration, step)
            
            # 打印详细的调试信息（使用配置的间隔）
            if iteration % self.log_interval == 0:
                num_episodes = len(self.episode_rewards)
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
                avg_discovery = np.mean(self.discovery_times[-100:]) if self.discovery_times else 0
                
                # 计算发现率
                recent_episodes = min(100, len(self.episode_lengths))
                discovery_rate = len(self.discovery_times[-recent_episodes:]) / recent_episodes if recent_episodes > 0 else 0
                
                # 计算进度百分比
                progress_pct = (step / num_steps) * 100
                
                print("\n" + "=" * 80)
                print(f"📊 训练进度报告 - 迭代 {iteration}/{total_iterations} ({progress_pct:.1f}%)")
                print("=" * 80)
                print(f"🔢 步数统计:")
                print(f"  当前步数: {step:,} / {num_steps:,}")
                print(f"  训练迭代: {iteration} / {total_iterations} (每次迭代收集{self.config.rollout_steps}步×{self.env.num_envs}环境)")
                print(f"  完成Episode: {num_episodes}")
                
                print(f"\n🎯 性能指标:")
                print(f"  平均奖励: {avg_reward:7.2f} (最近100个Episode)")
                print(f"  平均长度: {avg_length:6.1f} 步")
                print(f"  发现率: {discovery_rate*100:4.1f}% (最近100个Episode)")
                print(f"  平均发现时间: {avg_discovery:5.1f} 步")
                
                print(f"\n📉 损失信息:")
                print(f"  Actor Loss: {update_info['actor_loss']:7.4f}")
                print(f"  Critic Loss: {update_info['critic_loss']:7.4f}")
                print(f"  Entropy: {update_info['entropy']:7.4f}")
                print(f"  Entropy Loss: {update_info['entropy_loss']:7.4f}")
                
                print(f"\n🎓 训练指标:")
                print(f"  Approx KL: {update_info['approx_kl']:7.4f} (策略变化幅度)")
                print(f"  Clip Fraction: {update_info['clip_fraction']*100:5.1f}% (被裁剪样本比例)")
                print(f"  Explained Var: {update_info['explained_var']:7.4f} (Value拟合质量)")
                
                print(f"\n📐 梯度统计:")
                print(f"  Actor Grad Norm: {update_info['actor_grad_norm']:7.4f}")
                print(f"  Critic Grad Norm: {update_info['critic_grad_norm']:7.4f}")
                print(f"  Feature Grad Norm: {update_info['feature_grad_norm']:7.4f}")
                
                # 如果有环境的终止原因统计
                if hasattr(self.env, 'termination_reason'):
                    term_reasons = self.env.termination_reason.cpu().numpy()
                    reason_names = {0: '运行中', 1: '超时', 2: '成功', 3: '碰撞', 4: '越界', 5: '关节限位'}
                    print(f"\n⚠️  终止原因分布:")
                    for reason_id, reason_name in reason_names.items():
                        count = (term_reasons == reason_id).sum()
                        if count > 0:
                            print(f"  {reason_name}: {count}/{self.env.num_envs}")
                
                print("=" * 80 + "\n")
        
        # 训练完成时的总结
        print("\n" + "=" * 80)
        print("🎉 训练完成！")
        print("=" * 80)
        print(f"✅ 总迭代次数: {iteration:,} / {total_iterations:,}")
        print(f"✅ 总训练步数: {step:,} / {num_steps:,}")
        print(f"✅ 完成Episode数: {len(self.episode_rewards):,}")
        
        # 最终性能统计
        if self.episode_rewards:
            final_avg_reward = np.mean(self.episode_rewards[-100:])
            final_avg_length = np.mean(self.episode_lengths[-100:])
            final_discovery_rate = len(self.discovery_times[-100:]) / min(100, len(self.episode_lengths)) if self.discovery_times else 0
            
            print(f"\n📊 最终性能 (最近100个Episode):")
            print(f"  平均奖励: {final_avg_reward:7.2f}")
            print(f"  平均长度: {final_avg_length:6.1f} 步")
            print(f"  发现率: {final_discovery_rate*100:4.1f}%")
        
        # 保存最终模型
        print(f"\n💾 保存最终模型...")
        self.save_checkpoint(iteration, step, is_final=True)
        
        print("=" * 80 + "\n")
    
    def collect_rollout(self, obs, num_steps):
        """收集 rollout 数据"""
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_log_probs = []
        batch_dones = []  # 新增：收集 done 信息
        
        for _ in range(num_steps):
            with torch.no_grad():
                # 确保 obs 在正确的设备上
                obs = obs.to(self.device)
                action, log_prob, value = self.ac_network.get_action(obs)
                
                # 动作空间变换（仅在导航阶段 + 启用变换时）
                if self.use_action_transform:
                    action = self.action_transformer.transform(action, obs)
            
            batch_obs.append(obs)
            batch_actions.append(action)
            batch_values.append(value.squeeze())
            batch_log_probs.append(log_prob)
            
            obs, reward, done, _ = self.env.step(action)
            batch_rewards.append(reward)
            batch_dones.append(done)  # 新增：记录 done
            
            # 累计当前 episode 的奖励和长度
            self.current_episode_reward += reward.to(self.device)
            self.current_episode_length += 1
            
            # 统计完成的 episode
            done_cpu = done.cpu() if torch.is_tensor(done) else done
            for i in range(len(done_cpu)):
                if done_cpu[i]:
                    # 记录完成的 episode 的统计信息
                    ep_reward = self.current_episode_reward[i].item()
                    ep_length = self.current_episode_length[i].item()
                    
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    # 记录发现时间（如果有的话）
                    if hasattr(self.env, 'discovery_step') and self.env.target_discovered[i]:
                        discovery_step = self.env.discovery_step[i].item()
                        if discovery_step > 0:
                            self.discovery_times.append(discovery_step)
                    
                    # 重置该环境的累计信息
                    self.current_episode_reward[i] = 0.0
                    self.current_episode_length[i] = 0
        
        # 转换为张量并展平
        batch_obs = torch.cat(batch_obs, dim=0)  # [num_steps * num_envs, obs_dim]
        batch_actions = torch.cat(batch_actions, dim=0)  # [num_steps * num_envs, action_dim]
        batch_rewards = torch.stack(batch_rewards, dim=0)  # [num_steps, num_envs]
        batch_values = torch.stack(batch_values, dim=0)  # [num_steps, num_envs]
        batch_log_probs = torch.stack(batch_log_probs, dim=0)  # [num_steps, num_envs]
        batch_dones = torch.stack(batch_dones, dim=0)  # [num_steps, num_envs] - 新增
        
        # 展平为 [num_steps * num_envs]
        batch_rewards = batch_rewards.flatten()
        batch_values = batch_values.flatten()
        batch_log_probs = batch_log_probs.flatten()
        batch_dones = batch_dones.flatten()  # 新增
        
        return batch_obs, batch_actions, batch_rewards, batch_values, batch_log_probs, batch_dones
    
    def compute_gae(self, rewards, values, dones=None):
        """
        计算广义优势估计 (GAE) - 改进版
        
        Args:
            rewards: [num_steps * num_envs] - 奖励
            values: [num_steps * num_envs] - 价值估计
            dones: [num_steps * num_envs] - 终止标志（可选）
        
        Returns:
            advantages: [num_steps * num_envs] - 优势函数
            returns: [num_steps * num_envs] - 回报
        """
        rollout_steps = self.config.rollout_steps
        num_envs = self.env.num_envs
        
        # 确保在同一个设备上
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        
        rewards = rewards.reshape(rollout_steps, num_envs)
        values = values.reshape(rollout_steps, num_envs)
        
        # 处理 done 掩码
        if dones is not None:
            dones = dones.to(self.device)
            dones = dones.reshape(rollout_steps, num_envs)
            not_done = 1.0 - dones.float()  # 1 = 继续，0 = 终止
        else:
            # 如果没有提供 dones，假设所有步骤都继续（旧行为）
            not_done = torch.ones_like(rewards)
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        last_gae = torch.zeros(num_envs, device=self.device)  # 每个环境单独的 GAE
        
        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                next_value = torch.zeros(num_envs, device=self.device)
            else:
                next_value = values[t + 1]
            
            # 使用 not_done 掩码：episode 终止时，next_value 应该为 0
            delta = rewards[t] + self.gamma * next_value * not_done[t] - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * not_done[t] * last_gae
            returns[t] = advantages[t] + values[t]
            
            # 更新 last_gae，episode 终止时重置为 0
            last_gae = advantages[t] * not_done[t]
        
        return advantages.flatten(), returns.flatten()
    
    def update(self, obs, actions, old_log_probs, returns, old_values, advantages):
        """PPO 更新 - 改进版（参考 isaac-training）"""
        # ============ 1. 评估当前策略 ============
        log_probs, values, entropy = self.ac_network.evaluate(obs, actions)
        
        # ============ 2. 计算 Ratio（重要性采样比率）============
        # ratio = π_new(a|s) / π_old(a|s)
        ratio = torch.exp(log_probs - old_log_probs).unsqueeze(-1)  # [batch, 1]
        
        # ============ 3. Actor Loss（PPO-Clip）============
        # advantages 应该是 [batch] 或 [batch, 1]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)  # [batch, 1]
        
        surr1 = advantages * ratio
        surr2 = advantages * ratio.clamp(
            1.0 - self.actor_clip_ratio, 
            1.0 + self.actor_clip_ratio
        )
        actor_loss = -torch.mean(torch.min(surr1, surr2))  # 不乘以 action_dim
        
        # ============ 4. Critic Loss（Value Clipping）============
        # 将 values 和 returns 统一为 [batch] 形状
        values_flat = values.squeeze()  # [batch]
        returns_flat = returns if returns.dim() == 1 else returns.squeeze()  # [batch]
        old_values_flat = old_values if old_values.dim() == 1 else old_values.squeeze()  # [batch]
        
        if self.use_value_norm:
            # 反归一化以计算真实的 loss
            values_reshaped = values_flat.unsqueeze(-1)  # [batch, 1]
            returns_reshaped = returns_flat.unsqueeze(-1)  # [batch, 1]
            old_values_reshaped = old_values_flat.unsqueeze(-1)  # [batch, 1]
            
            values_denorm = self.value_norm.denormalize(values_reshaped).squeeze(-1)
            returns_denorm = self.value_norm.denormalize(returns_reshaped).squeeze(-1)
            old_values_denorm = self.value_norm.denormalize(old_values_reshaped).squeeze(-1)
            
            # Value Clipping（防止 critic 更新过大）
            value_clipped = old_values_denorm + (values_denorm - old_values_denorm).clamp(
                -self.critic_clip_ratio, self.critic_clip_ratio
            )
            
            # 取两个 loss 中的最大值（更保守的更新）
            critic_loss_original = self.critic_loss_fn(values_denorm, returns_denorm)
            critic_loss_clipped = self.critic_loss_fn(value_clipped, returns_denorm)
            critic_loss = torch.max(critic_loss_original, critic_loss_clipped)
        else:
            # 没有 value normalization 时，也使用 value clipping
            value_clipped = old_values_flat + (values_flat - old_values_flat).clamp(
                -self.critic_clip_ratio, self.critic_clip_ratio
            )
            critic_loss_original = self.critic_loss_fn(values_flat, returns_flat)
            critic_loss_clipped = self.critic_loss_fn(value_clipped, returns_flat)
            critic_loss = torch.max(critic_loss_original, critic_loss_clipped)
        
        # ============ 5. Entropy Loss（鼓励探索）============
        # 熵 = 策略分布的不确定性，熵越高越随机
        action_entropy = entropy.mean()
        entropy_loss = -self.entropy_coef * action_entropy  # 负号：最大化熵 = 最小化负熵
        
        # ============ 6. 总损失 ============
        total_loss = actor_loss + critic_loss + entropy_loss
        
        # ============ 7. 反向传播 ============
        self.feature_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        total_loss.backward()
        
        # ============ 8. 梯度裁剪（防止梯度爆炸）============
        # 当梯度向量的范数超过max_norm时，将整个梯度向量按比例缩小
        actor_grad_norm = nn.utils.clip_grad_norm_(
            list(self.ac_network.actor_alpha.parameters()) + 
            list(self.ac_network.actor_beta.parameters()),
            max_norm=self.max_grad_norm
        )
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.ac_network.critic.parameters(),
            max_norm=self.max_grad_norm
        )
        feature_grad_norm = nn.utils.clip_grad_norm_(
            list(self.ac_network.lidar_extractor.parameters()) +
            list(self.ac_network.state_mlp.parameters()) +
            list(self.ac_network.shared_net.parameters()),
            max_norm=self.max_grad_norm
        )
        
        # ============ 9. 参数更新 ============
        self.feature_optimizer.step()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # ============ 10. 统计信息 ============
        with torch.no_grad():
            # Explained Variance（解释方差）：value function 的拟合质量
            # 1.0 表示完美拟合，0.0 表示没有拟合，负值表示比均值还差
            if self.use_value_norm:
                explained_var = 1 - F.mse_loss(values_denorm, returns_denorm) / (returns_denorm.var() + 1e-8)
            else:
                explained_var = 1 - F.mse_loss(values_flat, returns_flat) / (returns_flat.var() + 1e-8)
            
            # Approximate KL Divergence（近似KL散度）
            # 衡量新旧策略的差异，用于监控策略变化幅度
            ratio_flat = ratio.squeeze()  # [batch]
            approx_kl = ((ratio_flat - 1) - torch.log(ratio_flat)).mean()
            
            # Clip Fraction（裁剪比例）
            # 有多少比例的样本被 clip 了，用于监控约束的作用
            clip_fraction = ((ratio_flat - 1).abs() > self.actor_clip_ratio).float().mean()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': action_entropy.item(),
            'entropy_loss': entropy_loss.item(),
            'actor_grad_norm': actor_grad_norm.item(),
            'critic_grad_norm': critic_grad_norm.item(),
            'feature_grad_norm': feature_grad_norm.item(),
            'explained_var': explained_var.item(),
            'approx_kl': approx_kl.item(),
            'clip_fraction': clip_fraction.item(),
        }

