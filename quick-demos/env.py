import numpy as np
import random

# Generate grid-based obstacles
def generate_obstacles_grid(grid_div, region_min, region_max, min_radius, max_radius, min_clearance=1.0):
    cell_size = (region_max - region_min) / grid_div
    obstacles = []

    for i in range(grid_div):
        for j in range(grid_div):
            for _ in range(10):  # Try up to 10 times per cell
                radius = random.uniform(min_radius, max_radius)
                margin = radius + 0.2
                x = np.random.uniform(region_min + i * cell_size + margin,
                                      region_min + (i + 1) * cell_size - margin)
                y = np.random.uniform(region_min + j * cell_size + margin,
                                      region_min + (j + 1) * cell_size - margin)

                # Check clearance from existing obstacles
                too_close = False
                for ox, oy, oradius in obstacles:
                    dist = np.hypot(x - ox, y - oy)
                    min_dist = radius + oradius + min_clearance
                    if dist < min_dist:
                        too_close = True
                        break

                if not too_close:
                    obstacles.append((x, y, radius))
                    break  # Accepted, go to next cell

    return obstacles

# Sample collision-free start
def sample_free_start(obstacles, goal, obstacle_region_min, obstacle_region_max, min_clearance=1.0):
    while True:
        x = np.random.uniform(obstacle_region_min, obstacle_region_max)
        y = np.random.uniform(obstacle_region_min, obstacle_region_max)

        # Ensure clearance from obstacles
        too_close = False
        for ox, oy, r in obstacles:
            if np.hypot(x - ox, y - oy) <= r + min_clearance:
                too_close = True
                break

        # Ensure not too close to the goal
        if np.hypot(x - goal[0], y - goal[1]) < 3.0:
            too_close = True

        if not too_close:
            return np.array([x, y])

# Sample collision-free goal
def sample_free_goal(obstacles, obstacle_region_min, obstacle_region_max):
    while True:
        x = np.random.uniform(obstacle_region_min, obstacle_region_max)
        y = np.random.uniform(obstacle_region_min, obstacle_region_max)
        safe = True
        for ox, oy, r in obstacles:
            if np.hypot(x - ox, y - oy) <= r + 1.5:
                safe = False
                break
        if safe:
            return np.array([x, y])