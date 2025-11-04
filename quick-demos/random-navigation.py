import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from utils import get_robot_state, get_ray_cast
import torch
import random
from matplotlib.patches import Wedge
from env import generate_obstacles_grid, sample_free_start, sample_free_goal
from agent import Agent

# === Set random seed ===
SEED = 0 
random.seed(SEED)
np.random.seed(SEED)

# === Device ===
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# === Constants ===
MAP_HALF_SIZE = 16
OBSTACLE_REGION_MIN = -15
OBSTACLE_REGION_MAX = 15
MIN_RADIUS = 0.3
MAX_RADIUS = 0.5
MAX_RAY_LENGTH = 4.0
DT = 0.1
GOAL_REACHED_THRESHOLD = 0.3
HRES_DEG = 10.0
VFOV_ANGLES_DEG = [-10.0, 0.0, 10.0, 20.0]
GRID_DIV = 10


# === Setup ===
obstacles = generate_obstacles_grid(GRID_DIV, OBSTACLE_REGION_MIN, OBSTACLE_REGION_MAX, MIN_RADIUS, MAX_RADIUS)
robot_vel = np.array([0.0, 0.0])
goal = sample_free_goal(obstacles, obstacle_region_min=OBSTACLE_REGION_MIN, obstacle_region_max=OBSTACLE_REGION_MAX)
robot_pos = sample_free_start(obstacles, goal, obstacle_region_min=OBSTACLE_REGION_MIN, obstacle_region_max=OBSTACLE_REGION_MAX)
start_pos = robot_pos.copy()
target_dir = goal - robot_pos 
trajectory = []

# === NavRL Agent ===
agent = Agent(device=device)


# === Visualization setup ===
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#fefcfb')  # Light warm figure background
ax.set_facecolor('#fdf6e3')         # Slightly warm off-white axes background
ax.set_xlim(-MAP_HALF_SIZE, MAP_HALF_SIZE)
ax.set_ylim(-MAP_HALF_SIZE, MAP_HALF_SIZE)
ax.set_aspect('equal')
ax.set_title("NavRL Dynamic Goal Navigation")
# ax.add_patch(Rectangle((-MAP_HALF_SIZE, -MAP_HALF_SIZE),
#                        2*MAP_HALF_SIZE,
#                        2*MAP_HALF_SIZE,
#                        edgecolor='black', facecolor='lightgray', alpha=0.1))
robot_dot, = ax.plot([], [], 'o', markersize=6, color="royalblue" , label='Robot', zorder=5)
velocity_arrow = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1.5, width=0.005, color='purple', zorder=4)
goal_dot, = ax.plot([], [], marker='*', markersize=15, color='red', linestyle='None', label='Goal')
start_dot, = ax.plot([], [], marker='s', markersize=8, color='navy', label='Start', linestyle='None', zorder=3)
trajectory_line, = ax.plot([], [], '-', linewidth=1.5, color="lime", label='Trajectory')
ray_lines = [ax.plot([], [], 'r--', linewidth=0.5)[0] for _ in range(int(360 / HRES_DEG))]
ax.legend(loc='upper left')

# Store obstacle patches for color updates
obstacle_patches = []
for obs in obstacles:
    patch = Circle((obs[0], obs[1]), obs[2], color='gray')
    ax.add_patch(patch)
    obstacle_patches.append(patch)

perception_wedge = Wedge(center=(0, 0), r=MAX_RAY_LENGTH, theta1=0, theta2=120,
                         color='cyan', alpha=0.2)
ax.add_patch(perception_wedge)


# === Simulation update ===
def update(frame):
    global robot_pos, robot_vel, goal, trajectory, target_dir, start_pos

    # Goal reach check
    to_goal = goal - robot_pos
    dist = np.linalg.norm(to_goal)
    if dist < GOAL_REACHED_THRESHOLD:
        start_pos = goal.copy()
        goal[:] = sample_free_goal(obstacles, obstacle_region_min=OBSTACLE_REGION_MIN, obstacle_region_max=OBSTACLE_REGION_MAX)
        trajectory = []
        velocity = np.array([0.0, 0.0])
        target_dir = goal - robot_pos 
        return


    # Get robot internal states
    robot_state = get_robot_state(robot_pos, goal, robot_vel, target_dir, device=device)

    # Get static obstacle representations
    static_obs_input, range_matrix, ray_segments = get_ray_cast(robot_pos, obstacles, max_range=MAX_RAY_LENGTH,
                                                       hres_deg=HRES_DEG,
                                                       vfov_angles_deg=VFOV_ANGLES_DEG,
                                                       start_angle_deg=np.degrees(np.arctan2(target_dir[1], target_dir[0])),
                                                       device=device)
    # Get dynamic obstacle representations (assume zero)
    dyn_obs_input = torch.zeros((1, 1, 5, 10), dtype=torch.float, device=device)

    # Target direction in tensor
    target_dir_tensor = torch.tensor(np.append(target_dir[:2], 0.0), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    # Output the planned velocity
    velocity = agent.plan(robot_state, static_obs_input, dyn_obs_input, target_dir_tensor)

    # ---Visualizaton update---
    robot_dot.set_data([robot_pos[0]], [robot_pos[1]])
    start_dot.set_data([start_pos[0]], [start_pos[1]])
    goal_dot.set_data([goal[0]], [goal[1]])
    trajectory.append(robot_pos.copy())
    trajectory_np = np.array(trajectory)
    trajectory_line.set_data(trajectory_np[:, 0], trajectory_np[:, 1])

    perception_center = robot_pos.copy()
    direction_angle_deg = np.degrees(np.arctan2(target_dir[1], target_dir[0]))
    cover_degree = 180
    start_angle = direction_angle_deg - cover_degree
    end_angle = direction_angle_deg + cover_degree

    perception_wedge.set_center((perception_center[0], perception_center[1]))
    perception_wedge.set_theta1(start_angle)
    perception_wedge.set_theta2(end_angle)

    # Highlight obstacles inside both angle and distance bounds
    for patch, (ox, oy, r) in zip(obstacle_patches, obstacles):
        dx, dy = ox - robot_pos[0], oy - robot_pos[1]
        dist_to_robot = np.hypot(dx, dy)
        angle_to_obs = np.degrees(np.arctan2(dy, dx))
        angle_diff = (angle_to_obs - direction_angle_deg + 180) % 360 - 180

        if abs(angle_diff) <= cover_degree and dist_to_robot <= MAX_RAY_LENGTH + r:
            patch.set_color('orange')  # Inside semisphere
        else:
            patch.set_color('gray')    # Outside semisphere

    v0_idx = VFOV_ANGLES_DEG.index(0.0)  # make sure this is consistent with get_ray_cast_3d_style()

    for i, (line, seg) in enumerate(zip(ray_lines, ray_segments)):
        # Compute angle of the current ray (relative to robot heading)
        ray_angle_deg = direction_angle_deg + i * HRES_DEG
        angle_diff = (ray_angle_deg - direction_angle_deg + 180) % 360 - 180

        ray_range = range_matrix[i, v0_idx]

        if abs(angle_diff) <= cover_degree and ray_range < MAX_RAY_LENGTH:
            line.set_data([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]])
            line.set_visible(True)
            line.set_color('blue' if i == 0 else 'red')
            line.set_linewidth(1.0)
        else:
            line.set_visible(False)

    velocity_arrow.set_offsets([robot_pos])
    velocity_arrow.set_UVC(robot_vel[0], robot_vel[1])
    # ---Visualizaton update end---


    # Update simulation states
    robot_pos += velocity * DT
    robot_vel = velocity.copy()

    return [robot_dot, goal_dot, trajectory_line, perception_wedge, start_dot, velocity_arrow] + ray_lines

ani = animation.FuncAnimation(fig, update, frames=300, interval=20, blit=False)
plt.show()