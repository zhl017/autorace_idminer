#!/usr/bin/env python3

from enum import Enum

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import UInt8, Bool

class DetectSign(Node):

    def __init__(self):
        super().__init__('detect_sign')

        # Publisher
        self.pub_image = self.create_publisher(
            CompressedImage,
            '/camera/output/compressed',
            10
        )
        self.pub_lane_type = self.create_publisher(
            UInt8,
            '/detect/lane_type',
            1
        )
        self.pub_mission = self.create_publisher(
            UInt8,
            '/mission',
            1
        )
        self.pub_avoid_control = self.create_publisher(
            Twist,
            '/avoid_control',
            1
        )
        self.pub_avoid_active = self.create_publisher(
            Bool,
            '/avoid_active',
            1
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
        self.sub_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.cbScan,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        )
        
        self.mission = Enum('mission', 'TrafficLight Intersection Construction Parking LevelCrossing Tunnel')
        self.cvBridge = CvBridge()
        self.fnPreproc()

        self.off_parking = True
        self.off_lidar = True
        self.go_parking = False
        self.r_side = False
        self.l_side = False

        self.step = 0
        self.step_count = 0
        self.counter = 1

        self.timer = self.create_timer(0.1, self.cbTimer)

    def cbMission(self, msg):
        if msg.data == self.mission.Parking.value:
            self.off_parking = False
            self.off_lidar = True
            self.go_parking = False
            self.r_side = False
            self.l_side = False
            self.step = 0
            self.step_count = 0
            self.get_logger().info('START Parking')
        else:
            self.off_parking = True
            self.off_lidar = True
            self.go_parking = False
            self.r_side = False
            self.l_side = False
            self.step = 0
            self.step_count = 0

    def cbTimer(self):
        if not self.go_parking:
            return
        
        active = Bool()
        control = Twist()

        if self.step == 0:
            active.data = True

            if self.r_side:
                control.linear.x = 0.13
                control.angular.z = -0.5
            elif self.l_side:
                control.linear.x = 0.13
                control.angular.z = 0.5

            self.step_count += 1
            if self.step_count > 35:
                self.step = 1
                self.step_count = 0

        elif self.step == 1:

            active.data = True
            control.linear.x = 0.0
            control.angular.z = 0.0

            self.step_count += 1
            if self.step_count > 50:
                self.step = 2
                self.step_count = 0

        elif self.step == 2:

            active.data = True
            control.linear.x =  -0.13
            control.angular.z = 0.0

            self.step_count += 1
            if self.step_count > 20:
                self.step = 3
                self.step_count = 0

        elif self.step == 3:

            active.data = True
            if self.r_side:
                control.linear.x = 0.0
                control.angular.z = -0.5
            elif self.l_side:
                control.linear.x = 0.0
                control.angular.z = 0.5

            self.step_count += 1
            if self.step_count > 30:
                self.step = 4
                self.step_count = 0

        elif self.step == 4:

            active.data = True
            control.linear.x = 0.1
            control.angular.z = 0.0

            self.step_count += 1
            if self.step_count > 30:
                self.step = 5
                self.step_count = 0

        elif self.step == 5:

            active.data = False
            control = Twist()

            self.step_count += 1
            if self.step_count > 50:
                self.step = 6
                self.step_count = 0

        elif self.step == 6:
            self.go_parking = False

            active.data = False
            control = Twist()

            lane_type = UInt8()
            lane_type.data = 0
            self.pub_lane_type.publish(lane_type)

            mission = UInt8()
            mission.data = self.mission.LevelCrossing.value
            self.pub_mission.publish(mission)

            self.get_logger().info('END Parking')

        
        self.pub_avoid_active.publish(active)
        self.pub_avoid_control.publish(control)

    def cbScan(self, msg):
        if self.off_lidar:
            return
        
        self.step_count += 1
        if self.step_count < 100:
            self.get_logger().info(f'{self.step_count}')
            return
        
        self.get_logger().info(f'scan data : {len(msg.ranges)}')
        
        angle_scan = 30

        scan_start_left = 45 - angle_scan
        scan_end_left = 45 + angle_scan

        scan_start_right = 315 - angle_scan
        scan_end_right = 315 + angle_scan

        threshold_distance = 0.5

        for i in range(scan_start_left, scan_end_left):
            if msg.ranges[i] < threshold_distance and msg.ranges[i] > 0.3:
                self.r_side = True
                self.go_parking = True
                self.off_lidar = True
                self.step_count = 0

                self.get_logger().info('RIGHT SIDE EMPTY')


        for i in range(scan_start_right, scan_end_right):
            if msg.ranges[i] < threshold_distance and msg.ranges[i] > 0.3:
                self.l_side = True
                self.go_parking = True
                self.off_lidar = True
                self.step_count = 0

                self.get_logger().info('LEFT SIDE EMPTY')

    def fnPreproc(self):
        self.sift = cv2.SIFT_create()

        dir_path = '/workspace/src/autorace_idminer/image'

        self.img = cv2.imread(dir_path + '/parking.png', 0)

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

    def cbFindSign(self, msg):

        if self.off_parking:
            return
        
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

        if not self.off_parking:
            good = []
            for m, n in matches:
                if m.distance < 0.7*n.distance:
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
                    self.off_parking = True
                    self.off_lidar = False
                    
                    image_out_num = 2

                    lane_type = UInt8()
                    lane_type.data = 1
                    self.pub_lane_type.publish(lane_type)

                    self.get_logger().info('Detect parking sign')
                    
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