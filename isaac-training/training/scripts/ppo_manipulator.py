"""
PPO for Manipulator Navigation
机械臂导航的PPO实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentBeta, BetaActor, GAE, make_batch


class PPO(TensorDictModuleBase):
    # TensorDictModuleBase：基础类，提供 TensorDict 支持的模块
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # ========== 深度图特征提取器 (替换激光雷达CNN) ==========
        depth_feature_network = nn.Sequential(
            # 输入: [N, 1, depth_w, depth_h] 例如 [N, 1, 80, 60]
            nn.LazyConv2d(out_channels=8, kernel_size=5, stride=2, padding=2), 
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            Rearrange("n c w h -> n (c w h)"), #展平操作 将空间特征转换为一维向量

            nn.LazyLinear(128), # 全连接层 自动学习最重要的特征组合 每个输入神经元与每个输出神经元都相连
            nn.LayerNorm(128), # 归一化计算：对每个样本的128个特征进行归一化
        ).to(self.device)
        
        # ========== 关节状态编码器 ==========
        joint_encoder = nn.Sequential(
            nn.LazyLinear(32),
            nn.ELU(),
            nn.LayerNorm(32),
        ).to(self.device)
        
        # ========== 动态障碍物编码器 ==========
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)
        
        # ========== 特征融合 ==========
        # TensorDictSequential：类似 nn.Sequential，但专门用于 TensorDict
        # 按顺序执行多个 TensorDictModule，自动处理输入输出键的传递

        """
        TensorDictModule:将普通PyTorch模块包装成能处理 TensorDict 的模块
        第一个参数：普通的 nn.Module
        in_keys：指定从 TensorDict 中读取哪些键
        out_keys：指定将结果写入哪些键
        """

        self.feature_extractor = TensorDictSequential(
            TensorDictModule(
                depth_feature_network,
                [("agents", "observation", "depth")],
                ["_depth_feature"]
            ),
            TensorDictModule(
                joint_encoder,
                [("agents", "observation", "joint_pos")],
                ["_joint_feature"]
            ),
            TensorDictModule(
                dynamic_obstacle_network,
                [("agents", "observation", "dynamic_obstacle")],
                ["_dynamic_obstacle_feature"]
            ),
            CatTensors([
                "_depth_feature",
                ("agents", "observation", "state"),
                "_joint_feature",
                "_dynamic_obstacle_feature"
            ], "_feature", del_keys=False),
            TensorDictModule(
                make_mlp([256, 256]),
                ["_feature"],
                ["_feature"]
            ),
        ).to(self.device)
        
        # ========== Actor: 6D末端速度 ==========
        self.n_agents, self.action_dim = action_spec.shape  # (1, 6)
        # 概率性策略
        self.actor = ProbabilisticActor(
            TensorDictModule(
                BetaActor(self.action_dim),
                ["_feature"],
                ["alpha", "beta"]
            ),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)
        
        # ========== Critic ==========
        self.critic = TensorDictModule(
            nn.LazyLinear(1),
            ["_feature"],
            ["state_value"]
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)
        
        # ========== 损失函数 ==========
        self.gae = GAE(0.99, 0.95)
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        
        # ========== 优化器 ==========
        self.feature_extractor_optim = torch.optim.Adam(
            self.feature_extractor.parameters(), 
            lr=cfg.feature_extractor.learning_rate
        )
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), 
            lr=cfg.actor.learning_rate
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), 
            lr=cfg.critic.learning_rate
        )
        
        # ========== 初始化 ==========
        dummy_input = observation_spec.zero()
        self.__call__(dummy_input)
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)
    
    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        
        # 动作缩放: [0, 1] → [-limit, limit]
        actions = (2 * tensordict["agents", "action_normalized"] - 1.0) * self.cfg.actor.action_limit
        
        # 末端速度限制
        # 线速度: vx, vy, vz (±0.3 m/s)
        # 角速度: wx, wy, wz (±0.5 rad/s)
        actions_scaled = actions.clone()
        actions_scaled[..., :3] = actions[..., :3] * 0.3   # 线速度缩放
        actions_scaled[..., 3:] = actions[..., 3:] * 0.5   # 角速度缩放
        
        tensordict["agents", "action"] = actions_scaled
        return tensordict
    
    def train(self, tensordict):
        """训练方法（与无人机版本相同）"""
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]
        
        rewards = tensordict["next", "agents", "reward"]
        dones = tensordict["next", "terminated"]
        values = tensordict["state_value"]
        
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)
        
        # GAE
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = (adv - adv.mean()) / adv.std().clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        
        # 训练
        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        
        infos = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}
    
    def _update(self, tensordict):
        """策略更新（与无人机版本相同）"""
        self.feature_extractor(tensordict)
        
        action_dist = self.actor.get_dist(tensordict)
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")])
        
        # Entropy
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)
        
        # Actor Loss (PPO)
        advantage = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(
            1.0 - self.cfg.actor.clip_ratio,
            1.0 + self.cfg.actor.clip_ratio
        )
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        
        # Critic Loss
        b_value = tensordict["state_value"]
        ret = tensordict["ret"]
        value = self.critic(tensordict)["state_value"]
        value_clipped = b_value + (value - b_value).clamp(
            -self.cfg.critic.clip_ratio,
            self.cfg.critic.clip_ratio
        )
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)
        
        # Total Loss
        loss = entropy_loss + actor_loss + critic_loss
        
        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        
        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        
        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])

