import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 支持相对导入和绝对导入
try:
    from .utils import ValueNorm
except ImportError:
    from isaac_gym_manipulator.active_training.utils import ValueNorm


class EfficientVoxelBackbone3D(nn.Module):
    """
    精简高效的3D体素骨干网络
    
    改进点：
    1. 2个阶段（而不是4层），减少50%计算量
    2. 使用stride=2卷积下采样（而不是MaxPool），保持梯度流动
    3. BatchNorm稳定训练
    4. 轻量级注意力机制聚焦重要区域
    5. 特征压缩到128维（而不是8192维），减少98%过拟合风险
    """
    def __init__(self, in_channels=1):
        super().__init__()
        
        # 🎯 阶段1: 基础特征 [32→16]
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1),  # 下采样
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),  # 特征增强
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
        )  # 输出: [B, 32, 16, 16, 16]
        
        # 🎯 阶段2: 语义特征 [16→8]
        self.stage2_conv1 = nn.Conv3d(32, 64, 3, stride=2, padding=1)  # 下采样
        self.stage2_bn1 = nn.BatchNorm3d(64)
        self.stage2_attention = self._create_attention(64)  # 轻量级注意力
        self.stage2_conv2 = nn.Conv3d(64, 64, 3, padding=1)
        self.stage2_bn2 = nn.BatchNorm3d(64)
        
        # 🎯 特征压缩到固定维度
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # [B, 64, 1, 1, 1]
        self.output_proj = nn.Linear(64, 128)  # 固定输出128维
        self.out_dim = 128
        
        # 初始化权重
        self._init_weights()
    
    def _create_attention(self, channels):
        """简化的空间注意力机制"""
        return nn.Sequential(
            nn.Conv3d(channels, 1, 1),  # 压缩到单通道重要性图
            nn.Sigmoid(),  # 重要性权重0-1
        )
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight, gain=0.01)  # 适中的初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, 32, 32, 32] 体素数据
        
        Returns:
            feat: [B, 128] 压缩后的特征向量
        """
        # 输入检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[EfficientVoxelBackbone3D] Warning: input contains NaN/inf, replacing with zeros")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        # 归一化输入（体素占据概率应该在 [0, 1]）
        x = torch.clamp(x, 0.0, 1.0)
        
        # 阶段1: 基础特征提取
        x = self.stage1(x)  # [B, 32, 16, 16, 16]
        
        # 阶段2: 语义特征提取 + 注意力
        x = self.stage2_conv1(x)  # [B, 64, 8, 8, 8]
        x = self.stage2_bn1(x)
        x = F.elu(x, inplace=True)
        
        # 应用注意力机制
        attention_map = self.stage2_attention(x)  # [B, 1, 8, 8, 8]
        x = x * attention_map  # 空间注意力加权
        
        x = self.stage2_conv2(x)
        x = self.stage2_bn2(x)
        x = F.elu(x, inplace=True)  # [B, 64, 8, 8, 8]
        
        # 全局池化 + 特征压缩
        x = self.global_pool(x)  # [B, 64, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]
        x = self.output_proj(x)  # [B, 128]
        
        # 输出检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[EfficientVoxelBackbone3D] Warning: output contains NaN/inf, replacing with zeros")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        # 限制输出范围
        x = torch.clamp(x, -10.0, 10.0)
        
        return x


# 保持向后兼容：别名
VoxelBackbone3D = EfficientVoxelBackbone3D


class ActorHeadGaussian(nn.Module):
    """
    高斯分布策略头（更适合机械臂控制）
    
    改进点：
    1. 输出均值和对数标准差（对数标准差可学习或固定）
    2. 高斯分布对称，更适合[-limit, +limit]范围的动作
    3. 适度容量 + 防过拟合
    """
    def __init__(self, in_dim: int, action_dim: int, log_std_init: float = -0.5):
        super().__init__()
        # 🎯 适度容量 + 防过拟合
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
        )
        # 均值输出（tanh限制到[-1, 1]，后续会缩放）
        self.mean_head = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        # 对数标准差（可学习参数）
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, fused_feat):
        # 检查输入是否有 NaN/inf
        if torch.isnan(fused_feat).any() or torch.isinf(fused_feat).any():
            print(f"[ActorHeadGaussian] Warning: fused_feat contains NaN/inf, replacing with zeros")
            fused_feat = torch.where(torch.isnan(fused_feat) | torch.isinf(fused_feat), 
                                    torch.zeros_like(fused_feat), fused_feat)
        
        h = self.mlp(fused_feat)
        
        # 检查 MLP 输出
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"[ActorHeadGaussian] Warning: MLP output contains NaN/inf, replacing with zeros")
            h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
        
        # 均值：[-1, 1]
        mean = self.mean_head(h)
        
        # 对数标准差：限制在合理范围
        log_std = torch.clamp(self.log_std, min=-2.0, max=0.5)
        
        return mean, log_std


# 保持向后兼容：Beta版本（已废弃，但保留以防需要）
class ActorHeadBeta(nn.Module):
    """
    稳健的策略头设计（改进版）- 已废弃，改用ActorHeadGaussian
    """
    def __init__(self, in_dim: int, action_dim: int):
        super().__init__()
        # 🎯 适度容量 + 防过拟合
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
        )
        self.alpha = nn.Sequential(nn.Linear(64, action_dim), nn.Softplus())
        self.beta = nn.Sequential(nn.Linear(64, action_dim), nn.Softplus())
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)  # 小初始化，防止初始输出过大
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, fused_feat):
        # 检查输入是否有 NaN/inf
        if torch.isnan(fused_feat).any() or torch.isinf(fused_feat).any():
            print(f"[ActorHeadBeta] Warning: fused_feat contains NaN/inf, replacing with zeros")
            fused_feat = torch.where(torch.isnan(fused_feat) | torch.isinf(fused_feat), 
                                    torch.zeros_like(fused_feat), fused_feat)
        
        h = self.mlp(fused_feat)
        
        # 检查 MLP 输出
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"[ActorHeadBeta] Warning: MLP output contains NaN/inf, replacing with zeros")
            h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
        
        # 更稳定的初始化：避免过小的alpha/beta，防止数值不稳定
        alpha_raw = self.alpha(h)
        beta_raw = self.beta(h)
        
        # 检查原始输出
        if torch.isnan(alpha_raw).any() or torch.isinf(alpha_raw).any():
            print(f"[ActorHeadBeta] Error: alpha_raw contains NaN/inf")
            alpha_raw = torch.where(torch.isnan(alpha_raw) | torch.isinf(alpha_raw), 
                                   torch.ones_like(alpha_raw) * 0.9, alpha_raw)
        if torch.isnan(beta_raw).any() or torch.isinf(beta_raw).any():
            print(f"[ActorHeadBeta] Error: beta_raw contains NaN/inf")
            beta_raw = torch.where(torch.isnan(beta_raw) | torch.isinf(beta_raw), 
                                  torch.ones_like(beta_raw) * 0.9, beta_raw)
        
        alpha = alpha_raw + 1.1  # 最小1.1，避免接近0
        beta = beta_raw + 1.1
        
        # 添加数值稳定性：限制alpha/beta范围
        alpha = torch.clamp(alpha, min=1.01, max=100.0)
        beta = torch.clamp(beta, min=1.01, max=100.0)
        return alpha, beta


class CriticHead(nn.Module):
    """
    稳健的价值函数头（改进版）
    
    改进点：
    1. 适度容量（128→64→1），避免过拟合
    2. BatchNorm稳定训练
    3. Dropout防止过拟合
    """
    def __init__(self, in_dim: int):
        super().__init__()
        # 🎯 适度容量 + 防过拟合
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1),
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)  # 小初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, fused_feat):
        return self.mlp(fused_feat).squeeze(-1)


class PPO:
    def __init__(self, cfg, obs_shapes, action_dim, device):
        self.cfg = cfg
        self.device = device
        voxel_shape, aux_dim = obs_shapes
        voxel_channels = voxel_shape[0]
        
        # 动作缩放参数（从env配置读取，如果没有则使用默认值）
        self.action_limit = cfg.get('action_limit', 0.0189)  # 默认值改为0.0189（参考ur5e_DDPG）
        # 注意：已移除两步缩放，不再需要linear_vel_scale和angular_vel_scale

        # Modules - 使用改进的网络架构
        self.backbone = EfficientVoxelBackbone3D(voxel_channels).to(device)
        
        # 🎯 改进的特征融合：辅助信息处理
        self.auxiliary_net = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 64),
        ).to(device)
        
        # 🎯 特征融合层
        # 视觉特征128维 + 辅助特征64维 = 192维
        fusion_dim = self.backbone.out_dim + 64  # 128 + 64 = 192
        self.fusion = nn.Sequential(
            nn.Linear(self.backbone.out_dim + 64, 256),  # 192 → 256
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256),
        ).to(device)
        
        # 使用融合后的特征维度
        fused_dim = 256
        
        # 🎯 改用高斯分布（更适合机械臂控制）
        self.actor = ActorHeadGaussian(fused_dim, action_dim).to(device)
        self.critic = CriticHead(fused_dim).to(device)

        # Optimizers（分离 feature/actor/critic + 新增的auxiliary和fusion）
        # 注意：cfg 已经是 ppo 配置字典（从 train.py 传入 cfg['ppo'].copy()）
        base_lr = cfg['lr']
        feat_lr = cfg.get('feature_lr', base_lr)
        actor_lr = cfg.get('actor_lr', base_lr)
        critic_lr = cfg.get('critic_lr', base_lr)
        
        # 🎯 将新添加的模块参数也加入优化器
        self.feature_optim = optim.Adam(
            list(self.backbone.parameters()) + 
            list(self.auxiliary_net.parameters()) + 
            list(self.fusion.parameters()), 
            lr=feat_lr
        )
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # PPO hyperparams
        self.clip_eps = cfg['clip_eps']
        self.entropy_coef = cfg['entropy_coef']
        self.value_coef = cfg['value_coef']
        self.max_grad_norm = cfg['max_grad_norm']
        self.epochs = cfg['epochs']
        self.batch_size = cfg['batch_size']
        # ValueNorm beta值（参考isaac-training：beta=0.995，更新更慢，避免过度归一化）
        # 注意：beta越大，历史数据权重越大，归一化更新越慢
        value_norm_beta = cfg.get('value_norm_beta', 0.995)  # 默认0.995（isaac-training标准）
        self.value_norm = ValueNorm(beta=value_norm_beta).to(device)
        self.critic_clip_ratio = cfg.get('critic_clip_ratio', self.clip_eps)
        # 使用HuberLoss替代MSE（参考isaac-training，更稳定）
        self.critic_loss_fn = nn.HuberLoss(delta=10.0)
        
        # 训练稳定性参数
        self.target_kl = cfg.get('target_kl', 0.02)  # KL早停阈值
        self.adv_clip = cfg.get('adv_clip', 10.0)  # Advantage裁剪范围
        
        # 学习率调度器（可选）
        if cfg.get('use_lr_scheduler', False):
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.feature_optim, T_max=cfg.get('lr_decay_steps', 1000)
            )
        else:
            self.lr_scheduler = None

    def _fuse(self, voxel, aux):
        """
        🎯 改进的特征融合：使用独立的辅助网络 + 融合层
        
        Args:
            voxel: [B, 1, 32, 32, 32] 体素数据
            aux: [B, aux_dim] 辅助信息
        
        Returns:
            fused: [B, 256] 融合后的特征
        """
        # 检查输入是否有 NaN 或 inf
        if torch.isnan(voxel).any() or torch.isinf(voxel).any():
            print(f"[PPO] Warning: voxel contains NaN/inf, replacing with zeros")
            voxel = torch.where(torch.isnan(voxel) | torch.isinf(voxel), torch.zeros_like(voxel), voxel)
        
        # 检查 voxel 数值范围，防止过大值
        if voxel.abs().max() > 100.0:
            print(f"[PPO] Warning: voxel values too large (max={voxel.abs().max()}), clamping")
            voxel = torch.clamp(voxel, -10.0, 10.0)
        
        if torch.isnan(aux).any() or torch.isinf(aux).any():
            print(f"[PPO] Warning: aux contains NaN/inf, replacing with zeros")
            aux = torch.where(torch.isnan(aux) | torch.isinf(aux), torch.zeros_like(aux), aux)
        
        # 限制 aux 数值范围
        if aux.abs().max() > 100.0:
            print(f"[PPO] Warning: aux values too large (max={aux.abs().max()}), clamping")
            aux = torch.clamp(aux, -10.0, 10.0)
        
        # 🎯 3D视觉特征提取
        voxel_feat = self.backbone(voxel)  # [B, 128]
        
        # 🎯 辅助信息处理
        aux_feat = self.auxiliary_net(aux)  # [B, 64]
        
        # 🎯 特征融合
        combined = torch.cat([voxel_feat, aux_feat], dim=1)  # [B, 192]
        fused = self.fusion(combined)  # [B, 256]
        
        # 检查融合后的特征
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            print(f"[PPO] Error: fused features contain NaN/inf, replacing with zeros")
            fused = torch.where(torch.isnan(fused) | torch.isinf(fused), torch.zeros_like(fused), fused)
        
        # 最终限制
        fused = torch.clamp(fused, -10.0, 10.0)
        
        return fused

    @torch.inference_mode()  # 比torch.no_grad()更高效，自动处理BatchNorm
    def act(self, voxel, aux):
        """
        Actor输出动作（使用高斯分布，更适合机械臂控制）
        
        Returns:
            a: [batch, 6] 关节角度增量 [dq1, dq2, dq3, dq4, dq5, dq6]（弧度，已缩放）
            logp: [batch] 动作的对数概率
            v: [batch] 价值函数估计
        """
        # 🎯 优化：不需要手动切换eval/train模式！
        # inference_mode() 已经自动处理了BatchNorm的行为
        # 移除所有 .eval() 和 .train() 调用，提升性能
        
        fused = self._fuse(voxel, aux) # 自动使用eval模式
        mean, log_std = self.actor(fused)
        
        # 检查输出是否有 NaN/inf
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print(f"[PPO.act] Warning: mean contains NaN/inf, replacing with zeros")
            mean = torch.where(torch.isnan(mean) | torch.isinf(mean), torch.zeros_like(mean), mean)
        if torch.isnan(log_std).any() or torch.isinf(log_std).any():
            print(f"[PPO.act] Warning: log_std contains NaN/inf, replacing with default")
            log_std = torch.where(torch.isnan(log_std) | torch.isinf(log_std), 
                                 torch.zeros_like(log_std) - 0.5, log_std)
        
        # 计算标准差
        std = torch.exp(log_std.clamp(-1.0, 1.0))  # 限制std范围 [0.368, 2.718] - 增大标准差范围以允许更大的探索
        
        # 创建高斯分布（在无界空间）
        dist = torch.distributions.Normal(mean, std)
        
        # 重参数化采样（支持梯度反向传播）
        action_raw = dist.rsample()  # [batch, 6] ∈ 无界范围
        
        # 🎯 使用tanh将无界动作映射到有界范围 [-action_limit, action_limit]
        # 这是标准PPO处理有界动作空间的方式（参考SAC、PPO论文）
        action_tanh = torch.tanh(action_raw)  # [batch, 6] ∈ [-1, 1]
        action = action_tanh * self.action_limit  # [batch, 6] ∈ [-action_limit, action_limit]
        
        # 🎯 计算log_prob（需要考虑tanh变换的雅可比行列式）
        # log_prob = log_prob_gaussian - log(1 - tanh^2(action_raw)) * action_limit
        # 这是因为 tanh 变换改变了概率密度
        logp_raw = dist.log_prob(action_raw).sum(-1)  # 高斯分布的log_prob
        # tanh的雅可比行列式修正项：log|d tanh(x)/dx| = log(1 - tanh^2(x))
        tanh_correction = torch.log(1.0 - action_tanh.pow(2) + 1e-6).sum(-1)  # 避免log(0)
        logp = logp_raw - tanh_correction  # 修正后的log_prob
        
        # 价值函数估计
        v = self.critic(fused)
        
        return action, logp, v

    def evaluate(self, voxel, aux, a):
        """
        评估动作（使用高斯分布）
        
        Args:
            a: [batch, 6] 关节角度增量 [dq1, dq2, dq3, dq4, dq5, dq6]（已缩放）
        
        Returns:
            logp: [batch] 动作的对数概率
            ent: [batch] 分布的熵
            v: [batch] 价值函数估计
        """
        # 🎯 评估时保持训练模式（用于计算梯度）
        fused = self._fuse(voxel, aux)
        mean, log_std = self.actor(fused)
        
        # 检查输出是否有 NaN/inf
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.where(torch.isnan(mean) | torch.isinf(mean), torch.zeros_like(mean), mean)
        if torch.isnan(log_std).any() or torch.isinf(log_std).any():
            log_std = torch.where(torch.isnan(log_std) | torch.isinf(log_std), 
                                 torch.zeros_like(log_std) - 0.5, log_std)
        
        # 计算标准差
        std = torch.exp(log_std.clamp(-1.0, 1.0))  # 限制std范围 [0.368, 2.718] - 增大标准差范围以允许更大的探索
        
        # 🎯 将动作转换回无界空间（逆tanh变换）
        # 在act中：action = tanh(action_raw) * action_limit
        # 所以：action_tanh = action / action_limit = tanh(action_raw)
        # 反向：action_raw = atanh(action_tanh)
        action_normalized = a / self.action_limit  # [batch, 6] ∈ [-1, 1]，即 tanh(action_raw)
        
        # 计算 atanh（tanh的逆函数）
        # atanh(x) = 0.5 * ln((1+x)/(1-x))，需要限制x在[-1+eps, 1-eps]范围内
        action_normalized_clamped = torch.clamp(action_normalized, -0.999, 0.999)  # 避免atanh的数值问题
        action_raw = 0.5 * torch.log((1.0 + action_normalized_clamped) / (1.0 - action_normalized_clamped + 1e-8))
        
        # 创建高斯分布
        dist = torch.distributions.Normal(mean, std)
        
        # 🎯 计算log_prob（需要考虑tanh变换的雅可比行列式）
        logp_raw = dist.log_prob(action_raw).sum(-1)  # 高斯分布的log_prob
        # tanh的雅可比行列式修正项：log|d tanh(x)/dx| = log(1 - tanh^2(x))
        tanh_correction = torch.log(1.0 - action_normalized_clamped.pow(2) + 1e-6).sum(-1)  # 避免log(0)
        logp = logp_raw - tanh_correction  # 修正后的log_prob
        
        # 计算熵
        ent = dist.entropy().sum(-1)
        
        # 价值函数估计
        v = self.critic(fused)
        
        return logp, ent, v

    def update(self, traj):
        adv = traj['adv']
        
        # 检查 advantage 是否有效
        if torch.isnan(adv).any() or torch.isinf(adv).any():
            print(f"[PPO] Warning: advantage contains NaN/inf, replacing with zeros")
            adv = torch.where(torch.isnan(adv) | torch.isinf(adv), torch.zeros_like(adv), adv)
        
        # 更稳定的Advantage标准化
        adv_mean = adv.mean()
        adv_std = adv.std()
        if torch.isnan(adv_mean) or torch.isnan(adv_std) or adv_std < 1e-7:
            print(f"[PPO] Warning: advantage stats invalid (mean={adv_mean}, std={adv_std}), using zeros")
            adv = torch.zeros_like(adv)
        else:
            adv = (adv - adv_mean) / (adv_std + 1e-7)  # 使用更小的epsilon避免除零
        
        # Advantage裁剪（参考isaac-training：他们不裁剪advantage，只标准化）
        # 如果adv_clip很大（如999.0），相当于不裁剪
        clip_fraction_adv = 0.0
        if self.adv_clip < 100.0:  # 只在adv_clip较小时才裁剪
            adv_before_clip = adv.clone()
            adv = torch.clamp(adv, -self.adv_clip, self.adv_clip)
            clip_fraction_adv = ((adv_before_clip.abs() >= self.adv_clip).float().mean().item())
            if clip_fraction_adv > 0.1:  # 如果超过10%被裁剪，打印警告
                print(f"[PPO] Warning: {clip_fraction_adv*100:.1f}% advantage clipped (adv_clip={self.adv_clip}, adv_mean={adv_mean.item():.2f}, adv_std={adv_std.item():.2f})")
        
        B = traj['voxel'].shape[0]
        idx = torch.randperm(B, device=self.device)

        # ValueNorm on returns（参考isaac-training实现）
        # isaac-training的做法：
        # 1. 先denormalize values计算GAE（在train.py的GAE计算中已完成）
        # 2. 然后normalize returns用于critic loss
        returns = traj['ret'].unsqueeze(-1)  # returns已经是denormalized的（从GAE计算）
        with torch.no_grad():
            self.value_norm.update(returns)
            ret_norm = self.value_norm.normalize(returns).squeeze(-1)  # normalized returns用于critic loss
        
        # 将val_den从denormalized转换为normalized（用于critic loss）
        # train.py中vals_den存储的是denormalized的，但critic loss在normalized space计算
        if 'val_den' in traj:
            traj['val_den'] = self.value_norm.normalize(traj['val_den'].unsqueeze(-1)).squeeze(-1)

        last_actor_loss = last_value_loss = last_ent = 0.0
        clip_fraction_v_list = []
        grad_norm_actor_final = 0.0
        grad_norm_critic_final = 0.0
        
        # 梯度范数计算辅助函数
        def compute_grad_norm(module):
            total_norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5
        
        for epoch in range(self.epochs):
            # 检查轨迹数据是否有效（在每个epoch开始时）
            if epoch == 0:
                for key in ['voxel', 'aux', 'act', 'logp', 'adv', 'ret', 'val_den']:
                    if key in traj:
                        if torch.isnan(traj[key]).any() or torch.isinf(traj[key]).any():
                            print(f"[PPO] Warning: traj['{key}'] contains NaN/inf in epoch {epoch}, replacing with zeros")
                            traj[key] = torch.where(torch.isnan(traj[key]) | torch.isinf(traj[key]), 
                                                   torch.zeros_like(traj[key]), traj[key])
            
            for s in range(0, B, self.batch_size):
                b = idx[s:s + self.batch_size]
                logp, ent, v = self.evaluate(traj['voxel'][b], traj['aux'][b], traj['act'][b])
                
                # 检查 evaluate 输出是否有效
                if torch.isnan(logp).any() or torch.isinf(logp).any():
                    if epoch == 0 and s == 0:
                        print(f"[PPO] Warning: logp from evaluate contains NaN/inf, clamping")
                    logp = torch.clamp(logp, -10.0, 10.0)
                if torch.isnan(ent).any() or torch.isinf(ent).any():
                    ent = torch.zeros_like(ent)
                if torch.isnan(v).any() or torch.isinf(v).any():
                    v = torch.zeros_like(v)
                
                # 计算 ratio，防止数值溢出
                logp_diff = logp - traj['logp'][b]
                logp_diff = torch.clamp(logp_diff, -10.0, 10.0)  # 防止 exp 溢出
                ratio = logp_diff.exp()
                
                # 检查 ratio 是否有效
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    if epoch == 0 and s == 0:
                        print(f"[PPO] Warning: ratio contains NaN/inf, fixing...")
                    ratio = torch.where(torch.isnan(ratio) | torch.isinf(ratio), torch.ones_like(ratio), ratio)
                
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[b]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 检查 actor_loss 是否有效
                if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                    if epoch == 0 and s == 0:
                        print(f"[PPO] Warning: actor_loss is NaN/inf, skipping this batch")
                    continue

                # Critic Loss（参考isaac-training：在normalized space计算）
                # isaac-training直接在normalized space计算，更稳定
                b_value = traj['val_den'][b]  # old value (already normalized)
                ret = ret_norm[b]  # returns (normalized)
                value = v  # new value (already normalized)
                
                # Value clipping in normalized space
                value_clipped = b_value + (value - b_value).clamp(
                    -self.critic_clip_ratio,
                    self.critic_clip_ratio
                )
                
                # 使用HuberLoss（参考isaac-training）
                critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
                critic_loss_original = self.critic_loss_fn(ret, value)
                critic_loss = torch.max(critic_loss_clipped, critic_loss_original)
                
                # 检查 critic_loss 是否有效
                if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                    if epoch == 0 and s == 0:
                        print(f"[PPO] Warning: critic_loss is NaN/inf, skipping this batch")
                    continue
                
                # Critic价值裁剪监控（在normalized space）
                with torch.no_grad():
                    clip_ratio_v = (value - b_value).abs() / (b_value.abs() + 1e-8)
                    clip_fraction_v = (clip_ratio_v > self.critic_clip_ratio).float().mean()
                    clip_fraction_v_list.append(clip_fraction_v.item())

                entropy_loss = -ent.mean()
                if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
                    entropy_loss = torch.tensor(0.0, device=self.device)
                
                # Total Loss（参考isaac-training：直接相加，不使用value_coef）
                loss = entropy_loss + actor_loss + critic_loss
                
                # 检查总损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    if epoch == 0 and s == 0:
                        print(f"[PPO] Warning: total loss is NaN/inf, skipping this batch")
                    continue

                # 分离优化步骤
                self.feature_optim.zero_grad()
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                # 梯度裁剪前检查梯度是否有效
                for name, param in self.backbone.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"[PPO] Warning: backbone.{name} gradient contains NaN/inf, zeroing...")
                            param.grad.zero_()
                
                for name, param in self.actor.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"[PPO] Warning: actor.{name} gradient contains NaN/inf, zeroing...")
                            param.grad.zero_()
                
                for name, param in self.critic.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"[PPO] Warning: critic.{name} gradient contains NaN/inf, zeroing...")
                            param.grad.zero_()
                
                # 梯度裁剪（参考isaac-training：max_norm=5.0）
                max_grad_norm_isaac = 5.0  # isaac-training使用5.0，而不是配置中的1.0
                grad_norm_backbone = nn.utils.clip_grad_norm_(self.backbone.parameters(), max_grad_norm_isaac)
                grad_norm_actor_batch = nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm_isaac)
                grad_norm_critic_batch = nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm_isaac)
                
                # 检查梯度范数是否异常
                if torch.isnan(grad_norm_backbone) or torch.isinf(grad_norm_backbone) or grad_norm_backbone > 1000:
                    print(f"[PPO] Warning: backbone grad_norm={grad_norm_backbone}, skipping update")
                    self.feature_optim.zero_grad()
                else:
                    self.feature_optim.step()
                
                if torch.isnan(grad_norm_actor_batch) or torch.isinf(grad_norm_actor_batch) or grad_norm_actor_batch > 1000:
                    print(f"[PPO] Warning: actor grad_norm={grad_norm_actor_batch}, skipping update")
                    self.actor_optim.zero_grad()
                else:
                    self.actor_optim.step()
                
                if torch.isnan(grad_norm_critic_batch) or torch.isinf(grad_norm_critic_batch) or grad_norm_critic_batch > 1000:
                    print(f"[PPO] Warning: critic grad_norm={grad_norm_critic_batch}, skipping update")
                    print(f"[PPO] Debug: critic_loss={critic_loss.item():.2f}, value_range=[{value.min().item():.2f}, {value.max().item():.2f}], ret_range=[{ret.min().item():.2f}, {ret.max().item():.2f}]")
                    self.critic_optim.zero_grad()
                else:
                    self.critic_optim.step()
                
                # 更新后检查权重是否有效
                for name, param in self.backbone.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"[PPO] Critical: backbone.{name} weight contains NaN/inf after update, reinitializing...")
                        if 'weight' in name:
                            nn.init.orthogonal_(param, gain=0.01)
                        else:
                            nn.init.constant_(param, 0.0)
                
                # 在step之前记录梯度范数（最后一个batch）
                if s + self.batch_size >= B:  # 最后一个batch
                    grad_norm_actor_final = compute_grad_norm(self.actor)
                    grad_norm_critic_final = compute_grad_norm(self.critic)

                last_actor_loss = actor_loss.item()
                last_value_loss = critic_loss.item()
                last_ent = ent.mean().item()
            
            # KL早停检查（在每个epoch后）
            with torch.no_grad():
                logp_new, _, _ = self.evaluate(traj['voxel'], traj['aux'], traj['act'])
                ratio_all = (logp_new - traj['logp']).exp()
                approx_kl = ((ratio_all - 1) - (logp_new - traj['logp'])).mean()
                
                # 如果KL散度超过阈值，提前终止训练
                if approx_kl > self.target_kl:
                    break
        
        # 学习率调度（如果启用）
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # 计算最终监控指标
        with torch.no_grad():
            logp_new, ent_new, _ = self.evaluate(traj['voxel'], traj['aux'], traj['act'])
            ratio_all = (logp_new - traj['logp']).exp()
            approx_kl = ((ratio_all - 1) - (logp_new - traj['logp'])).mean()
            clip_fraction = ((ratio_all - 1.0).abs() > self.clip_eps).float().mean()

        return {
            'actor_loss': float(last_actor_loss),
            'critic_loss': float(last_value_loss),
            'entropy': float(last_ent),
            'approx_kl': float(approx_kl.item()),
            'clip_fraction': float(clip_fraction.item()),
            'value_clip_frac': float(sum(clip_fraction_v_list) / len(clip_fraction_v_list)) if clip_fraction_v_list else 0.0,
            'avg_ratio': float(ratio_all.mean().item()),
            'max_ratio': float(ratio_all.max().item()),
            'min_ratio': float(ratio_all.min().item()),
            'grad_norm_actor': float(grad_norm_actor_final),
            'grad_norm_critic': float(grad_norm_critic_final),
        }
