import torch
import numpy as np
from typing import Optional


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def angle_shaping(prev_deg: torch.Tensor, curr_deg: torch.Tensor, coef: float, cap_deg: float):
    if prev_deg is None or prev_deg.numel() == 0:
        return torch.zeros_like(curr_deg)
    has_prev = torch.isfinite(prev_deg)
    delta = torch.zeros_like(curr_deg)
    delta[has_prev] = prev_deg[has_prev] - curr_deg[has_prev]
    improve = torch.clamp(delta, min=0.0, max=cap_deg)
    return coef * improve


class ValueNorm(torch.nn.Module):
    def __init__(self, beta: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('mean_sq', torch.zeros(1))
        self.register_buffer('debias', torch.zeros(1))
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        dim = tuple(range(x.dim() - 1))
        b_mean = x.mean(dim=dim)
        b_mean_sq = (x ** 2).mean(dim=dim)
        self.mean.mul_(self.beta).add_(b_mean * (1 - self.beta))
        self.mean_sq.mul_(self.beta).add_(b_mean_sq * (1 - self.beta))
        self.debias.mul_(self.beta).add_(1 - self.beta)

    def stats(self):
        m = self.mean / self.debias.clamp_min(self.eps)
        msq = self.mean_sq / self.debias.clamp_min(self.eps)
        # 注意：clamp_min(1e-2)可能太大，导致方差过小，value std=0
        # 参考isaac-training：他们使用clamp_min(1e-2)，但可能需要更大的最小值
        var = (msq - m * m).clamp_min(1e-4)  # 从1e-2降低到1e-4，允许更小的方差
        return m, var

    def normalize(self, x: torch.Tensor):
        m, var = self.stats()
        return (x - m) / torch.sqrt(var)

    def denormalize(self, x: torch.Tensor):
        m, var = self.stats()
        return x * torch.sqrt(var) + m


def compute_observation_confidence(d: float, max_range: float, angle_deg: float = 0.0) -> float:
    """
    根据距离和角度计算观测置信度
    
    Args:
        d: 深度距离（米）
        max_range: 最大观测范围（米）
        angle_deg: 相对于相机光轴的偏角（度），0表示正前方
    
    Returns:
        confidence: 占据概率置信度 [0.0, 1.0]
    """
    # 距离因子：距离越近置信度越高
    # 使用指数衰减：d/max_range 越小，置信度越高
    distance_factor = torch.exp(-d / (max_range * 0.5))  # 0.5为衰减系数
    
    # 角度因子：角度越大（边缘视野），置信度越低
    # 假设FOV=87度，边缘约43.5度
    angle_factor = torch.exp(-(angle_deg / 43.5) ** 2)  # 高斯衰减
    
    # 基础置信度：近距离高置信度（0.8-0.9），远距离低置信度（0.5-0.6）
    base_confidence = 0.5 + 0.4 * distance_factor.item()
    
    # 综合置信度
    confidence = base_confidence * angle_factor.item()
    
    # 限制在合理范围
    return max(0.4, min(0.9, confidence))


def build_voxel_from_depth(
    depth: torch.Tensor,
    cam_T_w: torch.Tensor,
    intrinsics: dict,
    grid_origin: torch.Tensor,
    grid_size: torch.Tensor,
    voxel_res: torch.Tensor,
    max_range: float,
    subsample: int = 4,
    free_step: float = 0.05,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    将深度图通过射线投射构建占据体素：射线经过体素标记为自由，射线终点标记为占据。
    - depth: [H, W] depth in meters
    - cam_T_w: [4,4] camera pose (world from camera) 或 world_T_cam？此处约定 world_from_cam
    - intrinsics: {fx, fy, cx, cy}
    - grid_origin: [3] 世界坐标下体素网格最小角点（米）
    - grid_size: [3] 体素数量 [Nx, Ny, Nz]
    - voxel_res: [3] 每个体素尺寸（米）
    - 返回占据概率体素 [1, 1, Nx, Ny, Nz]（格式：[batch, channels, Vx, Vy, Vz]）
    """
    H, W = depth.shape
    Nx, Ny, Nz = map(int, grid_size.tolist())
    voxel = torch.zeros((1, Nx, Ny, Nz), device=device)

    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

    # 采样像素
    ys = torch.arange(0, H, subsample, device=device)
    xs = torch.arange(0, W, subsample, device=device)

    # 相机原点（世界）
    cam_origin = cam_T_w[:3, 3]
    R = cam_T_w[:3, :3]

    for yi in ys:
        for xi in xs:
            d = depth[int(yi.item()), int(xi.item())]
            if not torch.isfinite(d) or d <= 0 or d > max_range:
                continue
            # 像素到相机坐标（Z 前）
            x_cam = (xi - cx) * d / fx
            y_cam = (yi - cy) * d / fy
            z_cam = d
            p_cam = torch.tensor([x_cam, y_cam, z_cam], device=device)
            # 方向：从相机到终点
            dir_cam = torch.nn.functional.normalize(p_cam, dim=0)
            dir_w = R @ dir_cam
            end_w = cam_origin + dir_w * d

            # 🎯 修复1：光线步进，主动标记自由空间（清除之前误标记的占据）
            num_steps = max(1, int(d / free_step))
            ts = torch.linspace(0.0, d - 1e-3, steps=num_steps, device=device)
            pts = cam_origin.unsqueeze(0) + ts.unsqueeze(1) * dir_w.unsqueeze(0)
            idx = ((pts - grid_origin) / voxel_res).long()
            mask = (idx[:, 0] >= 0) & (idx[:, 0] < Nx) & (idx[:, 1] >= 0) & (idx[:, 1] < Ny) & (idx[:, 2] >= 0) & (idx[:, 2] < Nz)
            idx = idx[mask]
            # 主动标记为自由空间（清除之前误标记的占据）
            voxel[0, idx[:, 0], idx[:, 1], idx[:, 2]] = 0.0  # 自由空间

            # 🎯 修复2：根据距离和角度计算占据置信度
            end_idx = ((end_w - grid_origin) / voxel_res).long()
            if 0 <= end_idx[0] < Nx and 0 <= end_idx[1] < Ny and 0 <= end_idx[2] < Nz:
                # 计算像素到光轴中心的距离（角度近似）
                dx = (xi.item() - cx) / fx
                dy = (yi.item() - cy) / fy
                angle_deg = torch.sqrt(torch.tensor(dx**2 + dy**2)) * 180.0 / torch.pi  # 近似角度
                
                # 使用动态置信度而不是固定0.7
                confidence = compute_observation_confidence(d.item(), max_range, angle_deg.item())
                
                # 融合新观测（累加并截断）
                voxel[0, end_idx[0], end_idx[1], end_idx[2]] = torch.clamp(
                    voxel[0, end_idx[0], end_idx[1], end_idx[2]] + confidence, 0.0, 1.0
                )

    # 添加通道维度：[1, Nx, Ny, Nz] -> [1, 1, Nx, Ny, Nz]（VoxelBackbone3D 期望的格式）
    return voxel.unsqueeze(1)


def update_voxel_from_depth(
    global_voxel: torch.Tensor,
    depth: torch.Tensor,
    cam_T_w: torch.Tensor,
    intrinsics: dict,
    grid_origin: torch.Tensor,
    grid_size: torch.Tensor,
    voxel_res: torch.Tensor,
    max_range: float,
    subsample: int = 4,
    free_step: float = 0.05,
    decay_factor: float = 0.95,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    增量更新全局体素地图（体素构建优化）
    只更新当前视角下的体素，而不是重新计算整个地图
    
    Args:
        global_voxel: [1, 1, Nx, Ny, Nz] 全局体素地图（会被原地更新）
        depth: [H, W] 深度图
        cam_T_w: [4,4] 相机位姿
        intrinsics: {fx, fy, cx, cy}
        grid_origin: [3] 网格原点
        grid_size: [3] 网格尺寸
        voxel_res: [3] 体素分辨率
        max_range: 最大范围
        subsample: 像素采样步长
        free_step: 自由空间步进距离
        decay_factor: 占据概率衰减因子（用于融合多次观测）
        device: 设备
    
    Returns:
        更新后的体素地图 [1, 1, Nx, Ny, Nz]（格式：[batch, channels, Vx, Vy, Vz]）
    """
    H, W = depth.shape
    Nx, Ny, Nz = map(int, grid_size.tolist())
    
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    
    # 采样像素
    ys = torch.arange(0, H, subsample, device=device)
    xs = torch.arange(0, W, subsample, device=device)
    
    # 相机原点（世界）
    cam_origin = cam_T_w[:3, 3]
    R = cam_T_w[:3, :3]
    
    # 创建一个临时体素用于标记本次更新的区域
    # 如果 global_voxel 没有通道维度，添加一个
    if global_voxel.dim() == 4:
        # [1, Nx, Ny, Nz] -> [1, 1, Nx, Ny, Nz]
        global_voxel = global_voxel.unsqueeze(1)
    
    # 🎯 修复：创建自由空间标记图（用于清除误标记的占据）
    free_space_mask = torch.zeros_like(global_voxel)
    new_occupied_voxel = torch.zeros_like(global_voxel)
    
    for yi in ys:
        for xi in xs:
            d = depth[int(yi.item()), int(xi.item())]
            if not torch.isfinite(d) or d <= 0 or d > max_range:
                continue
            
            # 像素到相机坐标（Z 前）
            x_cam = (xi - cx) * d / fx
            y_cam = (yi - cy) * d / fy
            z_cam = d
            p_cam = torch.tensor([x_cam, y_cam, z_cam], device=device)
            dir_cam = torch.nn.functional.normalize(p_cam, dim=0)
            dir_w = R @ dir_cam
            end_w = cam_origin + dir_w * d
            
            # 🎯 修复1：光线步进，主动标记自由空间
            num_steps = max(1, int(d / free_step))
            ts = torch.linspace(0.0, d - 1e-3, steps=num_steps, device=device)
            pts = cam_origin.unsqueeze(0) + ts.unsqueeze(1) * dir_w.unsqueeze(0)
            idx = ((pts - grid_origin) / voxel_res).long()
            mask = (idx[:, 0] >= 0) & (idx[:, 0] < Nx) & (idx[:, 1] >= 0) & (idx[:, 1] < Ny) & (idx[:, 2] >= 0) & (idx[:, 2] < Nz)
            idx = idx[mask]
            # 标记自由空间（用于清除之前误标记的占据）
            free_space_mask[0, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
            
            # 🎯 修复2：根据距离和角度计算占据置信度
            end_idx = ((end_w - grid_origin) / voxel_res).long()
            if 0 <= end_idx[0] < Nx and 0 <= end_idx[1] < Ny and 0 <= end_idx[2] < Nz:
                # 计算像素到光轴中心的距离（角度近似）
                dx = (xi.item() - cx) / fx
                dy = (yi.item() - cy) / fy
                angle_deg = torch.sqrt(torch.tensor(dx**2 + dy**2)) * 180.0 / torch.pi  # 近似角度
                
                # 使用动态置信度而不是固定0.7
                confidence = compute_observation_confidence(d.item(), max_range, angle_deg.item())
                new_occupied_voxel[0, 0, end_idx[0], end_idx[1], end_idx[2]] = confidence
    
    # 🎯 修复：融合新观测
    # 1. 先衰减全局地图中不在当前视野的占据
    global_voxel = global_voxel * decay_factor
    
    # 2. 清除自由空间（光线经过的体素）
    global_voxel[free_space_mask > 0] = 0.0
    
    # 3. 融合新观测的占据（取最大值）
    global_voxel = torch.maximum(global_voxel, new_occupied_voxel)
    global_voxel = torch.clamp(global_voxel, 0.0, 1.0)
    
    return global_voxel  # [1, 1, Nx, Ny, Nz]


class GAE:
    """
    Generalized Advantage Estimation (仿照 isaac-training)
    输入:
      rewards: [T, N]
      dones:   [T, N]  (bool)
      values:  [T, N]
      next_values: [T, N] 或 [N]（t=T-1 时的 next_v）
    输出:
      adv, ret: [T, N]
    """
    def __init__(self, gamma: float = 0.99, lam: float = 0.95, device: Optional[torch.device] = None):
        self.gamma = gamma
        self.lam = lam
        self.device = device

    def __call__(self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor):
        T, N = rewards.shape
        device = rewards.device
        adv = torch.zeros_like(rewards)
        ret = torch.zeros_like(rewards)
        last_adv = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            nv = next_values if next_values.dim() == 1 else next_values[t]
            not_done = (~dones[t]).float()
            delta = rewards[t] + self.gamma * nv * not_done - values[t]
            last_adv = delta + self.gamma * self.lam * not_done * last_adv
            adv[t] = last_adv
            ret[t] = adv[t] + values[t]
        return adv, ret
