#!/usr/bin/env python3
"""
ArUco Target Provider
将ArUco检测结果转换为导航目标
"""
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker


class ArucoTargetProvider:
    """
    功能:
    - 从TF树读取ArUco位置
    - 发布为导航目标
    - 可视化目标点
    """
    
    def __init__(self):
        # TF监听
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        
        # 参数
        self.aruco_frame = rospy.get_param('~aruco_frame', 'aruco_marker_0')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        
        # 发布目标
        self.target_pub = rospy.Publisher(
            '/rl_navigation/aruco_target',
            PoseStamped,
            queue_size=10
        )
        
        # 可视化
        self.marker_pub = rospy.Publisher(
            '/rl_navigation/aruco_marker',
            Marker,
            queue_size=10
        )
        
        # 定时器
        rospy.Timer(rospy.Duration(0.05), self.update_target)
        
        rospy.loginfo(f"[ArUco Provider]: Looking for {self.aruco_frame} in {self.base_frame}")
    
    def update_target(self, event):
        """更新目标位置"""
        try:
            # 从TF获取ArUco位置
            trans = self.tfBuffer.lookup_transform(
                self.base_frame,
                self.aruco_frame,
                rospy.Time(0),
                rospy.Duration(0.1)
            )
            
            # 转换为PoseStamped
            target = PoseStamped()
            target.header.stamp = rospy.Time.now()
            target.header.frame_id = self.base_frame
            target.pose.position.x = trans.transform.translation.x
            target.pose.position.y = trans.transform.translation.y
            target.pose.position.z = trans.transform.translation.z
            target.pose.orientation = trans.transform.rotation
            
            # 发布
            self.target_pub.publish(target)
            
            # 可视化
            self.publish_marker(target)
            
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"[ArUco Provider]: ArUco not detected - {e}")
    
    def publish_marker(self, target):
        """发布可视化标记"""
        marker = Marker()
        marker.header = target.header
        marker.ns = "aruco_target"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = target.pose
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.lifetime = rospy.Duration(0.2)
        
        self.marker_pub.publish(marker)


if __name__ == '__main__':
    rospy.init_node('aruco_target_provider')
    provider = ArucoTargetProvider()
    rospy.loginfo("[ArUco Provider]: Node started")
    rospy.spin()

