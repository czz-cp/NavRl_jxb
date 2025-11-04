import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
import torch
import random
from env import generate_obstacles_grid
from utils import get_robot_state, get_ray_cast, get_dyn_obs_state
from agent import Agent
from matplotlib import cm


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
MAP_HALF_SIZE = 20
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
NUM_ROBOTS = 8

# === Setup ===
obstacles = generate_obstacles_grid(GRID_DIV, OBSTACLE_REGION_MIN, OBSTACLE_REGION_MAX, MIN_RADIUS, MAX_RADIUS)
robot_xs = np.linspace(OBSTACLE_REGION_MIN + 3, OBSTACLE_REGION_MAX - 3, NUM_ROBOTS)
robot_positions = [np.array([x, -18.0]) for x in robot_xs]
robot_velocities = [np.zeros(2) for _ in range(NUM_ROBOTS)]
goals = [np.array([x, 18.0]) for x in robot_xs]
target_dirs = [goals[i] - robot_positions[i] for i in range(len(goals))]
trajectories = [[p.copy()] for p in robot_positions]

# === NavRL Agent ===
agent = Agent(device=device)

# === Visualization setup ===
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#fefcfb')  # Light warm figure background
ax.set_facecolor('#fdf6e3')         # Slightly warm off-white axes background
ax.set_xlim(-MAP_HALF_SIZE, MAP_HALF_SIZE)
ax.set_ylim(-MAP_HALF_SIZE, MAP_HALF_SIZE)
ax.set_aspect('equal')
ax.set_title("Multi-Robot NavRL Simulation")
# ax.add_patch(Rectangle(
#                     (-MAP_HALF_SIZE, OBSTACLE_REGION_MIN),
#                     2 * MAP_HALF_SIZE,
#                     OBSTACLE_REGION_MAX - OBSTACLE_REGION_MIN,
#                     edgecolor='black',
#                     facecolor='lightgray',
#                     alpha=0.3,          # Slightly stronger for contrast
#                     linewidth=1.0,      # Thinner edge
#                     linestyle='--'      # Dotted border looks more natural
#                 ))

for obs in obstacles:
    ax.add_patch(Circle((obs[0], obs[1]), obs[2], color='gray'))

# Use a colormap to assign unique colors to each robot
color_map = cm.get_cmap('tab10', NUM_ROBOTS)  # or 'tab20'
robot_colors = [color_map(i) for i in range(NUM_ROBOTS)]
robot_dots = [ax.plot([], [], 'o', color=robot_colors[i])[0] for i in range(NUM_ROBOTS)]
trajectory_lines = [ax.plot([], [], '-', linewidth=1.5, color=robot_colors[i])[0] for i in range(NUM_ROBOTS)]
goal_dots = [ax.plot(goal[0], goal[1], '*', color=robot_colors[i], markersize=10)[0] for i, goal in enumerate(goals)]
start_positions = [pos.copy() for pos in robot_positions]
start_dots = [ax.plot(pos[0], pos[1], 's', markersize=8, color=robot_colors[i], label=f'Start {i}')[0]
              for i, pos in enumerate(start_positions)]

# === Simulation update ===
def update(frame):
    global target_dirs
    artists = []

    for i in range(NUM_ROBOTS):
        pos = robot_positions[i]
        vel = robot_velocities[i]
        goal = goals[i]

        to_goal = goal - pos
        dist = np.linalg.norm(to_goal)
        if dist < GOAL_REACHED_THRESHOLD:
            continue

        target_dir = target_dirs[i]

        # Get robot internal states
        robot_state = get_robot_state(pos, goal, vel, target_dir, device=device)

        # Get static obstacle representations
        static_obs_input, range_matrix, _ = get_ray_cast(
            pos, obstacles, max_range=MAX_RAY_LENGTH,
            hres_deg=HRES_DEG,
            vfov_angles_deg=VFOV_ANGLES_DEG,
            start_angle_deg=np.degrees(np.arctan2(target_dir[1], target_dir[0])),
            device=device
        )
        
        # Target direction in tensor
        target_tensor = torch.tensor(np.append(target_dir[:2], 0.0), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

        # Get dynamic obstacle representations
        dyn_obs_input = get_dyn_obs_state(pos, vel, robot_positions, robot_velocities, target_tensor, device=device)

        # Output the planned velocity
        velocity = agent.plan(robot_state, static_obs_input, dyn_obs_input, target_tensor)

        # Update positions
        robot_positions[i] += velocity * DT
        robot_velocities[i] = velocity.copy()
        trajectories[i].append(robot_positions[i].copy())

        # Draw
        robot_dots[i].set_data([robot_positions[i][0]], [robot_positions[i][1]])
        trajectory_lines[i].set_data(*zip(*trajectories[i]))

        artists += [robot_dots[i], trajectory_lines[i]]
        artists += start_dots

    return artists

ani = animation.FuncAnimation(fig, update, frames=300, interval=20, blit=False)
plt.show()