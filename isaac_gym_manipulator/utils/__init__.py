"""
Isaac Gym Manipulator - 工具模块

模块组织：
- math_utils: 数学工具（四元数、位姿误差等）
- collision: 碰撞检测（连杆距离、多源检测）
- rewards: 奖励函数（探索、导航、惩罚）
- sensors: 传感器（LiDAR、摄像头）
"""

from .math_utils import (
    quat_rotate_vector_batch,
    quat_rotate_vector,
    compute_pose_error_three_point,
    compute_point_to_segment_distance,
    quat_conjugate,
    quat_multiply,
    euler_to_quat,
)

from .collision import (
    compute_link_obstacle_distances,
    detect_collision_multisource,
    compute_collision_penalty,
    check_collision,
)

from .rewards import (
    compute_downward_search_reward,
    compute_systematic_scan_reward,
    compute_active_exploration_reward,
    compute_static_penalty,
    compute_z_exploration_reward,
)

from .sensors import (
    compute_lidar_ray_directions,
    check_target_in_camera_view,
)

__all__ = [
    # Math utils
    'quat_rotate_vector_batch',
    'quat_rotate_vector',
    'compute_pose_error_three_point',
    'compute_point_to_segment_distance',
    'quat_conjugate',
    'quat_multiply',
    'euler_to_quat',
    # Collision
    'compute_link_obstacle_distances',
    'detect_collision_multisource',
    'compute_collision_penalty',
    'check_collision',
    # Rewards
    'compute_downward_search_reward',
    'compute_systematic_scan_reward',
    'compute_active_exploration_reward',
    'compute_static_penalty',
    'compute_z_exploration_reward',
    # Sensors
    'compute_lidar_ray_directions',
    'check_target_in_camera_view',
]


