#!/usr/bin/env python3

import math
from enum import Enum

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import UInt8

class DetectSign(Node):

    def __init__(self):
        super().__init__('detect_sign')

        # Get Detect Param
        self.fnDeclareParameter()

        # Publisher
        self.pub_image = self.create_publisher(
            CompressedImage,
            '/camera/output/compressed',
            10
        )
        self.pub_init = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )
        self.pub_goal = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        # Subscription
        self.sub_image = self.create_subscription(
            CompressedImage,
            '/camera/input/compressed',
            self.cbFindSign,
            10
        )
        self.sub_mission = self.create_subscription(
            UInt8,
            '/mission',
            self.cbMission,
            1
        )

        self.mission = Enum('mission', 'TrafficLight Intersection Construction Parking LevelCrossing Tunnel')
        self.cvBridge = CvBridge()
        self.fnPreproc()
        
        self.off_tunnel = False
        self.go_nav = False

        self.step = 0
        self.step_count = 0
        self.counter = 1

        self.timer = self.create_timer(1.0, self.cbTimer)

    def cbTimer(self):
        if not self.go_nav:
            return
        
        if self.step == 0:
            self.step_count += 1
            if self.step_count > 5:
                self.step = 1
                self.step_count = 0

        elif self.step == 1:
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'

            msg.pose.pose.position.x = self.init_position_x
            msg.pose.pose.position.y = self.init_position_y
            msg.pose.pose.position.z = self.init_position_z

            yaw = math.radians(self.init_orientation_yaw)
            msg.pose.pose.orientation.x = self.init_orientation_x
            msg.pose.pose.orientation.y = self.init_orientation_y
            msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

            msg.pose.covariance = [0.0] * 36

            self.pub_init.publish(msg)

            self.step = 2

        elif self.step == 2 :
            self.step_count += 1
            if self.step_count > 5:
                self.step = 3
                self.step_count = 0

        elif self.step == 3:
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'

            goal_msg.pose.position.x = self.goal_position_x
            goal_msg.pose.position.y = self.goal_position_y
            goal_msg.pose.position.z = self.goal_position_z

            yaw = math.radians(self.goal_orientation_yaw)
            goal_msg.pose.orientation.x = self.goal_orientation_x
            goal_msg.pose.orientation.y = self.goal_orientation_y
            goal_msg.pose.orientation.z = math.sin(yaw / 2.0)
            goal_msg.pose.orientation.w = math.cos(yaw / 2.0)

            self.pub_goal.publish(goal_msg)

            self.step = 4

        elif self.step == 4:
            self.step_count += 1
            if self.step_count > 5:
                self.go_nav = False
                self.step = 0
                self.step_count = 0
                self.get_logger().info('END Tunnel')

    def fnPreproc(self):
        self.sift = cv2.SIFT_create()

        dir_path = '/workspace/src/autorace_idminer/image'

        self.img = cv2.imread(dir_path + '/tunnel.png', 0)

        self.kp, self.des = self.sift.detectAndCompute(
            self.img, None
        )

        FLANN_INDEX_KDTREE = 0
        index_params = {
            'algorithm': FLANN_INDEX_KDTREE,
            'trees': 5
        }

        search_params = {
            'checks': 50
        }

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def fnCalcMSE(self, arr1, arr2):
        squared_diff = (arr1 - arr2) ** 2
        total_sum = np.sum(squared_diff)
        num_all = arr1.shape[0] * arr1.shape[1] 
        err = total_sum / num_all
        return err
    
    def cbMission(self, msg):
        if msg.data == self.mission.Tunnel.value:
            self.off_tunnel = False
            self.go_nav = False
            self.step = 0
            self.step_count = 0
            self.get_logger().info('START Tunnel')
        else:
            self.off_tunnel = True
            self.go_nav = False
            self.step = 0
            self.step_count = 0

    def cbFindSign(self, msg):

        if self.off_tunnel:
            return
        
        # drop the frame to 1/5 (6fps) because of the processing speed. This is up to your computer's operating power.
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        MIN_MATCH_COUNT = 5
        MIN_MSE_DECISION = 70000

        kp1, des1 = self.sift.detectAndCompute(cv_image_input, None)

        matches = self.flann.knnMatch(des1, self.des, k=2)

        image_out_num = 1

        if not self.off_tunnel:
            good = []
            for m, n in matches:
                if m.distance < 0.5*n.distance:
                    good.append(m)
            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([
                    self.kp[m.trainIdx].pt for m in good
                ]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matches = mask.ravel().tolist()

                mse = self.fnCalcMSE(src_pts, dst_pts)
                if mse < MIN_MSE_DECISION:
                    self.off_tunnel = True
                    self.go_nav = True

                    image_out_num = 2
                    
                    self.get_logger().info('Detect tunnel sign')
            else:
                matches = None

        if image_out_num == 1:
            self.pub_image.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(
                    cv_image_input,
                    'jpg'
                )
            )

        elif image_out_num == 2:
            draw_params = {
                'matchColor': (255, 0, 0),  
                'singlePointColor': None,
                'matchesMask': matches,
                'flags': 2
            }
            final = cv2.drawMatches(
                cv_image_input,
                kp1,
                self.img,
                self.kp,
                good,
                None,
                **draw_params
            )
            self.pub_image.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(
                    final,
                    'jpg'
                )
            )

    def fnDeclareParameter(self):
        self.declare_parameter(
            'init_pose.position.x', 0.0)
        self.declare_parameter(
            'init_pose.position.y', 0.0)
        self.declare_parameter(
            'init_pose.position.z', 0.0)
        self.declare_parameter(
            'init_pose.orientation.x', 0.0)
        self.declare_parameter(
            'init_pose.orientation.y', 0.0)
        self.declare_parameter(
            'init_pose.orientation.yaw', 0)
        self.declare_parameter(
            'goal_pose.position.x', 0.0)
        self.declare_parameter(
            'goal_pose.position.y', 0.0)
        self.declare_parameter(
            'goal_pose.position.z', 0.0)
        self.declare_parameter(
            'goal_pose.orientation.x', 0.0)
        self.declare_parameter(
            'goal_pose.orientation.y', 0.0)
        self.declare_parameter(
            'goal_pose.orientation.yaw', 0)

        self.init_position_x = self.get_parameter(
            'init_pose.position.x').value
        self.init_position_y = self.get_parameter(
            'init_pose.position.y').value
        self.init_position_z = self.get_parameter(
            'init_pose.position.z').value
        self.init_orientation_x = self.get_parameter(
            'init_pose.orientation.x').value
        self.init_orientation_y = self.get_parameter(
            'init_pose.orientation.y').value
        self.init_orientation_yaw = self.get_parameter(
            'init_pose.orientation.yaw').value
        self.goal_position_x = self.get_parameter(
            'goal_pose.position.x').value
        self.goal_position_y = self.get_parameter(
            'goal_pose.position.y').value
        self.goal_position_z = self.get_parameter(
            'goal_pose.position.z').value
        self.goal_orientation_x = self.get_parameter(
            'goal_pose.orientation.x').value
        self.goal_orientation_y = self.get_parameter(
            'goal_pose.orientation.y').value
        self.goal_orientation_yaw = self.get_parameter(
            'goal_pose.orientation.yaw').value

def main(args=None):
    rclpy.init(args=args)
    node = DetectSign()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()