import torch
import torch.nn as nn
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
import numpy as np

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        return alpha, beta

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) 
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]



def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)
    # print("vec: ", vec.shape)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # goal direction y
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coodirnate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel

# State transformation
def get_robot_state(pos, goal, vel, target_dir, device="cuda"):
    rpos = np.zeros(3)
    rpos[:2] = goal - pos
    vel3 = np.zeros(3)
    vel3[:2] = vel

    distance = np.linalg.norm(rpos)
    distance_2d = np.linalg.norm(rpos[:2])
    distance_z = 0

    target_dir_2d = np.zeros(3)
    target_dir_2d[:2] = target_dir

    rpos_clipped = rpos / max(distance, 1e-6)

    rpos_clipped_g = vec_to_new_frame(torch.tensor(rpos_clipped, dtype=torch.float),
                                      torch.tensor(target_dir_2d, dtype=torch.float))
    vel_g = vec_to_new_frame(torch.tensor(vel3, dtype=torch.float),
                             torch.tensor(target_dir_2d, dtype=torch.float))

    d2 = torch.tensor(distance_2d, dtype=torch.float).view(1, 1, 1)
    dz = torch.tensor(distance_z, dtype=torch.float).view(1, 1, 1)

    return torch.cat([rpos_clipped_g, d2, dz, vel_g], dim=-1).squeeze(0).to(device)

# Raycasting (geometry-based)
def ray_cast_distance(robot_pos, angle, obstacles, max_range=4.0, safety_margin=0.1):
    dx = np.cos(angle)
    dy = np.sin(angle)
    min_dist = max_range

    for ox, oy, r in obstacles:
        cx = ox - robot_pos[0]
        cy = oy - robot_pos[1]

        proj = cx * dx + cy * dy
        if proj < 0 or proj > max_range:
            continue

        closest_x = robot_pos[0] + proj * dx
        closest_y = robot_pos[1] + proj * dy

        dist_to_center = np.hypot(ox - closest_x, oy - closest_y)
        if dist_to_center <= r + safety_margin:
            adjusted_dist = max(proj - r - safety_margin, 0.0)
            min_dist = min(min_dist, adjusted_dist)

    return min_dist

def get_ray_cast(robot_pos, obstacles, max_range=4.0,
                          hres_deg=10.0,
                          vfov_angles_deg=[-10.0, 0.0, 10.0, 20.0],
                          start_angle_deg=0.0,
                          device="cuda"):
    num_h = int(360 / hres_deg)
    num_v = len(vfov_angles_deg)

    range_matrix = np.full((num_h, num_v), max_range)
    v0_idx = vfov_angles_deg.index(0.0)
    ray_segments_2d = []

    for h in range(num_h):
        h_angle_deg = start_angle_deg + h * hres_deg
        h_angle_rad = np.deg2rad(h_angle_deg)

        dist = ray_cast_distance(robot_pos, h_angle_rad, obstacles, max_range, 0.0)
        range_matrix[h, 1:4] = dist

        x_end = robot_pos[0] + dist * np.cos(h_angle_rad)
        y_end = robot_pos[1] + dist * np.sin(h_angle_rad)
        ray_segments_2d.append(((robot_pos[0], robot_pos[1]), (x_end, y_end)))

    static_obs_input = np.maximum(range_matrix, 0.1)
    static_obs_input = max_range - static_obs_input
    static_obs_input = torch.tensor(static_obs_input, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
    return static_obs_input, range_matrix, ray_segments_2d


def get_dyn_obs_state(pos, vel, robot_positions, robot_velocities, target_dir, robot_size=0.25, max_range=4.0, max_num=5, device="cuda"):
    """
    pos:              torch.Tensor (2,)     - current robot position
    vel:              torch.Tensor (2,)     - current robot velocity
    robot_positions:  List[np.ndarray]      - positions of all robots
    robot_velocities: List[np.ndarray]      - velocities of all robots
    """

    # Convert input
    pos = torch.tensor(pos, dtype=torch.float, device=device)
    vel = torch.tensor(vel, dtype=torch.float, device=device)
    others_pos = torch.tensor(robot_positions, dtype=torch.float, device=device)
    others_vel = torch.tensor(robot_velocities, dtype=torch.float, device=device)

    # Filter out self by checking if position matches
    dists = torch.norm(others_pos - pos, dim=-1)
    mask = dists > 1e-4  # exclude self
    others_pos = others_pos[mask]
    others_vel = others_vel[mask]
    dists = dists[mask]

    # Keep only those within range
    in_range_mask = dists < max_range
    others_pos = others_pos[in_range_mask]
    others_vel = others_vel[in_range_mask]
    dists = dists[in_range_mask]
    if len(others_pos) == 0:
        return torch.zeros((1, 1, max_num, 10), dtype=torch.float, device=device)

    # Sort by distance
    sorted_indices = torch.argsort(dists)
    others_pos = others_pos[sorted_indices]
    others_vel = others_vel[sorted_indices]

    # Select top-k
    num_dyn = min(max_num, others_pos.shape[0])
    closest_pos = others_pos[:num_dyn]
    closest_vel = others_vel[:num_dyn]

    # Relative position (3D) and velocity (3D)
    rel_pos = torch.zeros((num_dyn, 1, 3), device=device)
    rel_vel = torch.zeros((num_dyn, 1, 3), device=device)
    rel_pos[:, :, :2] = (closest_pos.squeeze(1) - pos).unsqueeze(1)
    rel_vel[:, :, :2] = closest_vel.unsqueeze(1)
    target_dir_3d = torch.zeros(num_dyn, 3, device=device)
    target_dir_3d[:, :2] = target_dir[:, :, :2]

    # Transform to local frame
    rel_pos_g = vec_to_new_frame(rel_pos, target_dir_3d)
    rel_vel_g = vec_to_new_frame(rel_vel, target_dir_3d)

    # Distance components
    dist_2d = rel_pos_g[:, :, :2].norm(dim=-1, keepdim=True)
    dist_z = torch.zeros(num_dyn, 1, dtype=torch.float, device=device).unsqueeze(-1)
    rel_pos_gn = rel_pos_g / rel_pos.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Width and height (for now fixed or placeholder)
    width = torch.zeros((num_dyn, 1), device=device)
    height = torch.zeros((num_dyn, 1), device=device)


    # Compose state
    dyn_state = torch.cat([
        rel_pos_gn,         # (x, y, z) unit vec
        dist_2d,            # [dx, dy]
        dist_z,             # scalar
        rel_vel_g,          # (vx, vy, vz)
        width.unsqueeze(1), height.unsqueeze(1)       # size hints
    ], dim=-1).squeeze(1)

    # Pad if needed
    if num_dyn < max_num:
        padding = torch.zeros((max_num - num_dyn, 10), device=device)
        dyn_state = torch.cat([dyn_state, padding], dim=0)

    return dyn_state.unsqueeze(0).unsqueeze(0) # [1, 1, max_num, 10]
