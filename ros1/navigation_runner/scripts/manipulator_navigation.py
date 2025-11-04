#!/usr/bin/env python3
"""
Manipulator Navigation Node
UR10e机械臂导航节点 - 使用NavRL策略
"""
import rospy
import numpy as np
import torch
import tf2_ros
import tf.transformations
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Twist, Vector3, Point
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from map_manager.srv import RayCast
from onboard_detector.srv import GetDynamicObstacles
from navigation_runner.srv import GetSafeAction
from ppo import PPO
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from utils import vec_to_new_frame
import os
import threading


class ManipulatorNavigation:
    """
    UR10e机械臂导航控制器
    
    功能:
    - 从TF获取末端位姿和ArUco目标
    - 使用RL策略生成末端速度
    - 集成动态避障
    - 发布速度命令到UR控制器
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        # ========== TF监听器 ==========
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        
        # ========== 状态变量 ==========
        self.joint_state = None
        self.ee_pose = None
        self.target_pose = None
        self.target_received = False
        self.joint_received = False
        self.depth_points = []
        self.dynamic_obstacles = []
        
        # 安全控制
        self.safety_stop = False
        self.robot_radius = 0.1  # 机械臂半径（用于碰撞检测）
        
        # 深度图参数
        self.depth_w = cfg.sensor.depth_w
        self.depth_h = cfg.sensor.depth_h
        self.depth_range = cfg.sensor.depth_range
        
        # ========== 订阅话题 ==========
        self.joint_sub = rospy.Subscriber(
            '/joint_states',
            JointState,
            self.joint_callback
        )
        
        self.target_sub = rospy.Subscriber(
            '/rl_navigation/aruco_target',
            PoseStamped,
            self.target_callback
        )
        
        # ========== 发布话题 ==========
        # UR10e末端速度控制
        self.ee_vel_pub = rospy.Publisher(
            '/ur10e_velocity_controller/command',
            Twist,
            queue_size=10
        )
        
        # 或关节速度控制（备选）
        self.joint_vel_pub = rospy.Publisher(
            '/scaled_pos_joint_traj_controller/command',
            Float64MultiArray,
            queue_size=10
        )
        
        # 可视化
        self.depth_vis_pub = rospy.Publisher(
            '/rl_navigation/depth_visualization',
            MarkerArray,
            queue_size=10
        )
        self.cmd_vis_pub = rospy.Publisher(
            '/rl_navigation/cmd_visualization',
            MarkerArray,
            queue_size=10
        )
        self.target_vis_pub = rospy.Publisher(
            '/rl_navigation/target_visualization',
            Marker,
            queue_size=10
        )
        
        # ========== 加载RL策略 ==========
        self.policy = self.init_model()
        self.policy.eval()
        print("[Manipulator Nav]: RL Policy loaded.")
        
        # ========== 安全线程 ==========
        safety_thread = threading.Thread(target=self.safety_check)
        safety_thread.daemon = True
        safety_thread.start()
        
        # ========== 定时器 ==========
        rospy.Timer(rospy.Duration(0.05), self.raycast_callback)  # 20Hz
        rospy.Timer(rospy.Duration(0.05), self.dynamic_obstacle_callback)  # 20Hz
        rospy.Timer(rospy.Duration(0.05), self.control_callback)  # 20Hz
        rospy.Timer(rospy.Duration(0.1), self.visualization_callback)  # 10Hz
        
        print("[Manipulator Nav]: Node initialized. Waiting for ArUco target...")
    
    def init_model(self):
        """初始化RL模型"""
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec(
                        (observation_dim,), device=self.cfg.device
                    ),
                    "depth": UnboundedContinuousTensorSpec(
                        (1, self.depth_w, self.depth_h), device=self.cfg.device
                    ),
                    "joint_pos": UnboundedContinuousTensorSpec(
                        (6,), device=self.cfg.device
                    ),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec(
                        (1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state),
                        device=self.cfg.device
                    ),
                }),
            }).expand(1)
        }, shape=[1], device=self.cfg.device)
        
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((6,), device=self.cfg.device),
            })
        }).expand(1, 6).to(self.cfg.device)
        
        policy = PPO(self.cfg.algo, observation_spec, action_spec, self.cfg.device)
        
        # 加载检查点
        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")
        checkpoint = "manipulator_checkpoint.pt"
        policy.load_state_dict(
            torch.load(os.path.join(file_dir, checkpoint), map_location=self.cfg.device)
        )
        
        return policy
    
    def safety_check(self):
        """安全停止线程"""
        while not rospy.is_shutdown():
            if not self.safety_stop:
                input("[Manipulator Nav]: Press Enter to STOP!\n")
                self.safety_stop = True
            else:
                input("[Manipulator Nav]: Press Enter to CONTINUE!\n")
                self.safety_stop = False
    
    def joint_callback(self, msg):
        """关节状态回调"""
        self.joint_state = msg
        self.joint_received = True
    
    def target_callback(self, msg):
        """ArUco目标回调"""
        self.target_pose = msg
        if not self.target_received:
            print("[Manipulator Nav]: ArUco target received!")
            self.target_received = True
    
    def get_ee_pose(self):
        """从TF获取末端位姿"""
        try:
            trans = self.tfBuffer.lookup_transform(
                'base_link',
                'ee_link',
                rospy.Time(0),
                rospy.Duration(0.1)
            )
            return trans
        except:
            return None
    
    def raycast_callback(self, event):
        """获取深度数据"""
        ee_pose = self.get_ee_pose()
        if ee_pose is None:
            return
        
        try:
            raycast = rospy.ServiceProxy('occupancy_map/raycast', RayCast)
            
            pos_msg = Point()
            pos_msg.x = ee_pose.transform.translation.x
            pos_msg.y = ee_pose.transform.translation.y
            pos_msg.z = ee_pose.transform.translation.z
            
            # 获取末端朝向
            quat = [
                ee_pose.transform.rotation.x,
                ee_pose.transform.rotation.y,
                ee_pose.transform.rotation.z,
                ee_pose.transform.rotation.w
            ]
            _, _, yaw = tf.transformations.euler_from_quaternion(quat)
            
            # 调用raycast（模拟Realsense深度图）
            response = raycast(
                pos_msg,
                yaw,  # 起始角度
                self.depth_range,  # 3.0m
                -29,  # vfov_min
                29,   # vfov_max
                self.depth_h,  # 60
                self.cfg.sensor.depth_fov_h / self.depth_w  # hres
            )
            
            # 重塑为深度图
            num_points = len(response.points) // 3
            points = np.array(response.points).reshape(num_points, 3)
            self.depth_points = points
            
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(1.0, f"Raycast failed: {e}")
    
    def dynamic_obstacle_callback(self, event):
        """获取动态障碍物"""
        ee_pose = self.get_ee_pose()
        if ee_pose is None:
            return
        
        try:
            get_obstacle = rospy.ServiceProxy(
                'onboard_detector/get_dynamic_obstacles',
                GetDynamicObstacles
            )
            
            pos_msg = Point()
            pos_msg.x = ee_pose.transform.translation.x
            pos_msg.y = ee_pose.transform.translation.y
            pos_msg.z = ee_pose.transform.translation.z
            
            response = get_obstacle(pos_msg, 2.0)  # 2米范围
            
            # 转换为张量
            dyn_obs_num = self.cfg.algo.feature_extractor.dyn_obs_num
            dynamic_obstacle_pos = torch.zeros(dyn_obs_num, 3, device=self.cfg.device)
            dynamic_obstacle_vel = torch.zeros(dyn_obs_num, 3, device=self.cfg.device)
            dynamic_obstacle_size = torch.zeros(dyn_obs_num, 3, device=self.cfg.device)
            
            for i in range(min(dyn_obs_num, len(response.position))):
                dynamic_obstacle_pos[i] = torch.tensor([
                    response.position[i].x,
                    response.position[i].y,
                    response.position[i].z
                ], device=self.cfg.device)
                dynamic_obstacle_vel[i] = torch.tensor([
                    response.velocity[i].x,
                    response.velocity[i].y,
                    response.velocity[i].z
                ], device=self.cfg.device)
                dynamic_obstacle_size[i] = torch.tensor([
                    response.size[i].x,
                    response.size[i].y,
                    response.size[i].z
                ], device=self.cfg.device)
            
            self.dynamic_obstacles = (
                dynamic_obstacle_pos,
                dynamic_obstacle_vel,
                dynamic_obstacle_size
            )
            
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(1.0, f"Dynamic obstacle query failed: {e}")
    
    def control_callback(self, event):
        """主控制循环 - 20Hz"""
        # ========== 前置检查 ==========
        if not self.joint_received:
            return
        
        ee_pose = self.get_ee_pose()
        if ee_pose is None:
            return
        
        if not self.target_received or len(self.depth_points) == 0:
            self.publish_zero_velocity()
            return
        
        if self.safety_stop:
            self.publish_zero_velocity()
            rospy.logwarn_throttle(1.0, "Safety stop activated!")
            return
        
        # ========== 获取状态 ==========
        # 末端位置
        ee_pos = np.array([
            ee_pose.transform.translation.x,
            ee_pose.transform.translation.y,
            ee_pose.transform.translation.z
        ])
        
        # 目标位置
        target_pos = np.array([
            self.target_pose.pose.position.x,
            self.target_pose.pose.position.y,
            self.target_pose.pose.position.z
        ])
        
        # 关节位置
        joint_pos = np.array(self.joint_state.position[:6])
        
        # ========== 构建观测 ==========
        obs = self.build_observation(ee_pos, target_pos, joint_pos)
        
        # ========== RL推理 ==========
        with torch.no_grad():
            with set_exploration_type(ExplorationType.MEAN):
                output = self.policy(obs)
        
        ee_vel_cmd = output["agents", "action"].squeeze().cpu().numpy()
        
        # ========== 安全检查 ==========
        ee_vel_safe = self.safety_filter(ee_vel_cmd, ee_pos, target_pos)
        
        # ========== 发布命令 ==========
        self.publish_ee_velocity(ee_vel_safe)
        
        # 存储用于可视化
        self.cmd_vel_rl = ee_vel_cmd
        self.cmd_vel_safe = ee_vel_safe
    
    def build_observation(self, ee_pos, target_pos, joint_pos):
        """构建RL观测"""
        # ========== 1. 机械臂状态 ==========
        rpos = target_pos - ee_pos
        distance = np.linalg.norm(rpos)
        distance_xy = np.linalg.norm(rpos[:2])
        distance_z = rpos[2]
        
        # 目标方向
        target_dir = torch.tensor(rpos, device=self.cfg.device)
        target_dir[2] = 0
        target_dir = target_dir / target_dir.norm().clamp(1e-6)
        
        # 归一化方向
        rpos_normalized = rpos / max(distance, 1e-6)
        rpos_normalized_g = vec_to_new_frame(
            torch.tensor(rpos_normalized, device=self.cfg.device).unsqueeze(0).unsqueeze(0),
            target_dir.unsqueeze(0).unsqueeze(0)
        ).squeeze()
        
        # 末端速度（从关节速度估计）
        ee_vel = self.estimate_ee_velocity()
        ee_vel_g = vec_to_new_frame(
            torch.tensor(ee_vel[:3], device=self.cfg.device).unsqueeze(0).unsqueeze(0),
            target_dir.unsqueeze(0).unsqueeze(0)
        ).squeeze()
        
        # 拼接状态
        robot_state = torch.cat([
            rpos_normalized_g,
            torch.tensor([distance_xy], device=self.cfg.device),
            torch.tensor([distance_z], device=self.cfg.device),
            ee_vel_g
        ]).unsqueeze(0)  # [1, 8]
        
        # ========== 2. 深度图 ==========
        if len(self.depth_points) > 0:
            depth_tensor = torch.tensor(self.depth_points, device=self.cfg.device)
            ee_pos_tensor = torch.tensor(ee_pos, device=self.cfg.device)
            
            # 计算距离
            distances = (depth_tensor - ee_pos_tensor).norm(dim=-1)
            distances = distances.clamp_max(self.depth_range)
            
            # 重塑为图像
            depth_image = (self.depth_range - distances).reshape(
                1, 1, self.depth_w, self.depth_h
            )
        else:
            depth_image = torch.zeros(
                1, 1, self.depth_w, self.depth_h,
                device=self.cfg.device
            )
        
        # ========== 3. 关节位置 ==========
        joint_pos_tensor = torch.tensor(
            joint_pos, device=self.cfg.device, dtype=torch.float
        ).unsqueeze(0)
        
        # ========== 4. 动态障碍物 ==========
        if len(self.dynamic_obstacles) > 0:
            dyn_obs_pos, dyn_obs_vel, dyn_obs_size = self.dynamic_obstacles
            
            # 相对位置
            ee_pos_tensor = torch.tensor(ee_pos, device=self.cfg.device)
            dyn_obs_rpos = dyn_obs_pos - ee_pos_tensor
            
            # 转换到目标坐标系
            dyn_obs_rpos_g = vec_to_new_frame(
                dyn_obs_rpos.unsqueeze(0),
                target_dir.unsqueeze(0).unsqueeze(0)
            ).squeeze(0)
            
            dyn_obs_distance = dyn_obs_rpos.norm(dim=-1, keepdim=True)
            dyn_obs_distance_2d = dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            dyn_obs_distance_z = dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            dyn_obs_rpos_gn = dyn_obs_rpos_g / dyn_obs_distance.clamp(1e-6)
            
            # 速度
            dyn_obs_vel_g = vec_to_new_frame(
                dyn_obs_vel.unsqueeze(0),
                target_dir.unsqueeze(0).unsqueeze(0)
            ).squeeze(0)
            
            # 尺寸编码
            dyn_obs_width = torch.max(dyn_obs_size[:, 0], dyn_obs_size[:, 1])
            dyn_obs_width_cat = (dyn_obs_width / 0.25 - 1.0).clamp(0, 3).unsqueeze(-1)
            dyn_obs_height_cat = torch.ones_like(dyn_obs_width_cat)
            
            dyn_obs_states = torch.cat([
                dyn_obs_rpos_gn,
                dyn_obs_distance_2d,
                dyn_obs_distance_z,
                dyn_obs_vel_g,
                dyn_obs_width_cat,
                dyn_obs_height_cat
            ], dim=-1).unsqueeze(0).unsqueeze(0)
        else:
            dyn_obs_states = torch.zeros(
                1, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10,
                device=self.cfg.device
            )
        
        # ========== 5. 组装TensorDict ==========
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": robot_state,
                    "depth": depth_image,
                    "joint_pos": joint_pos_tensor,
                    "dynamic_obstacle": dyn_obs_states
                })
            })
        })
        
        return obs
    
    def estimate_ee_velocity(self):
        """估计末端速度"""
        # 简化版本：使用数值微分
        # 完整版本需要雅可比矩阵
        if self.joint_state is None or len(self.joint_state.velocity) < 6:
            return np.zeros(6)
        
        # 粗略估计：关节速度的加权和
        joint_vel = np.array(self.joint_state.velocity[:6])
        
        # 简化的雅可比近似（需要改进）
        ee_vel_approx = np.zeros(6)
        ee_vel_approx[:3] = joint_vel[:3] * 0.5  # 粗略估计线速度
        ee_vel_approx[3:] = joint_vel[3:] * 0.3  # 粗略估计角速度
        
        return ee_vel_approx
    
    def safety_filter(self, ee_vel, ee_pos, target_pos):
        """安全过滤"""
        ee_vel_safe = ee_vel.copy()
        
        # 1. 深度检查
        if len(self.depth_points) > 0:
            min_depth = np.min(np.linalg.norm(self.depth_points - ee_pos, axis=1))
            if min_depth < 0.15:  # 15cm内有障碍物
                ee_vel_safe *= 0.3  # 减速到30%
                rospy.logwarn_throttle(0.5, f"Near obstacle! min_depth={min_depth:.3f}")
        
        # 2. 速度限制
        linear_vel_norm = np.linalg.norm(ee_vel_safe[:3])
        if linear_vel_norm > 0.3:
            ee_vel_safe[:3] = ee_vel_safe[:3] / linear_vel_norm * 0.3
        
        angular_vel_norm = np.linalg.norm(ee_vel_safe[3:])
        if angular_vel_norm > 0.5:
            ee_vel_safe[3:] = ee_vel_safe[3:] / angular_vel_norm * 0.5
        
        # 3. 接近目标减速
        distance = np.linalg.norm(target_pos - ee_pos)
        if distance < 0.2:
            scale = distance / 0.2
            ee_vel_safe *= scale
        
        if distance < 0.05:
            ee_vel_safe *= 0.0  # 停止
        
        return ee_vel_safe
    
    def publish_ee_velocity(self, ee_vel):
        """发布末端速度"""
        vel_msg = Twist()
        vel_msg.linear.x = ee_vel[0]
        vel_msg.linear.y = ee_vel[1]
        vel_msg.linear.z = ee_vel[2]
        vel_msg.angular.x = ee_vel[3]
        vel_msg.angular.y = ee_vel[4]
        vel_msg.angular.z = ee_vel[5]
        
        self.ee_vel_pub.publish(vel_msg)
    
    def publish_zero_velocity(self):
        """发布零速度"""
        vel_msg = Twist()
        self.ee_vel_pub.publish(vel_msg)
    
    def visualization_callback(self, event):
        """可视化"""
        if not self.target_received:
            return
        
        # 目标点
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "aruco_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = self.target_pose.pose
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.target_vis_pub.publish(marker)
        
        # 速度箭头
        if hasattr(self, 'cmd_vel_rl'):
            self.publish_cmd_visualization()
    
    def publish_cmd_visualization(self):
        """发布速度箭头可视化"""
        ee_pose = self.get_ee_pose()
        if ee_pose is None:
            return
        
        msg = MarkerArray()
        
        # RL速度（红色）
        rl_arrow = Marker()
        rl_arrow.header.frame_id = "base_link"
        rl_arrow.header.stamp = rospy.Time.now()
        rl_arrow.ns = "rl_cmd"
        rl_arrow.id = 0
        rl_arrow.type = Marker.ARROW
        rl_arrow.action = Marker.ADD
        
        start = Point()
        start.x = ee_pose.transform.translation.x
        start.y = ee_pose.transform.translation.y
        start.z = ee_pose.transform.translation.z
        
        end = Point()
        end.x = start.x + self.cmd_vel_rl[0]
        end.y = start.y + self.cmd_vel_rl[1]
        end.z = start.z + self.cmd_vel_rl[2]
        
        rl_arrow.points = [start, end]
        rl_arrow.scale.x = 0.02
        rl_arrow.scale.y = 0.04
        rl_arrow.color.r = 1.0
        rl_arrow.color.a = 1.0
        
        msg.markers.append(rl_arrow)
        
        # 安全速度（绿色）
        safe_arrow = Marker()
        safe_arrow.header = rl_arrow.header
        safe_arrow.ns = "safe_cmd"
        safe_arrow.id = 1
        safe_arrow.type = Marker.ARROW
        safe_arrow.action = Marker.ADD
        
        end_safe = Point()
        end_safe.x = start.x + self.cmd_vel_safe[0]
        end_safe.y = start.y + self.cmd_vel_safe[1]
        end_safe.z = start.z + self.cmd_vel_safe[2]
        
        safe_arrow.points = [start, end_safe]
        safe_arrow.scale.x = 0.02
        safe_arrow.scale.y = 0.04
        safe_arrow.color.g = 1.0
        safe_arrow.color.a = 1.0
        
        msg.markers.append(safe_arrow)
        
        self.cmd_vis_pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('manipulator_navigation')
    
    # 加载配置
    import sys
    import yaml
    
    config_file = rospy.get_param('~config', 'manipulator_nav.yaml')
    cfg_dict = yaml.safe_load(open(config_file))
    
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg_dict)
    
    nav = ManipulatorNavigation(cfg)
    
    rospy.loginfo("[Manipulator Nav]: Node ready. Waiting for ArUco target...")
    rospy.spin()

