#!/usr/bin/env python3

from enum import Enum

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import UInt8

class DetectSign(Node):

    def __init__(self):
        super().__init__('detect_sign')

        # Publisher
        self.pub_image = self.create_publisher(
            CompressedImage,
            '/camera/output/compressed',
            1
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

        # Subscription
        self.sub_image = self.create_subscription(
            CompressedImage,
            '/camera/input/compressed',
            self.cbFindSign,
            1
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

        self.off_intersection = True
        self.off_left_right = True
        self.off_exit_timer = True

        self.counter = 1
        self.left_count = 0
        self.right_count = 0
        self.exit_count = 0

        self.timer = self.create_timer(0.5, self.cbTimer)

    def cbMission(self, msg):
        if msg.data == self.mission.Intersection.value:
            self.off_intersection = False
            self.off_left_right = False
            self.left_count = 0
            self.right_count = 0
            self.exit_count = 0
            self.get_logger().info('START Intersection')
        else:
            self.off_intersection = True
            self.off_left_right = True
            self.off_exit_timer = True
            self.left_count = 0
            self.right_count = 0
            self.exit_count = 0

    def cbTimer(self):
        if self.off_exit_timer:
            return
        
        self.exit_count += 1

        if self.exit_count > 20:
            self.off_exit_timer = True
            self.exit_count = 0

            lane_type = UInt8()
            lane_type.data = 0
            self.pub_lane_type.publish(lane_type)

            mission = UInt8()
            mission.data = self.mission.Construction.value
            self.pub_mission.publish(mission)

            self.get_logger().info('END Intersection')

    def fnPreproc(self):
        self.sift = cv2.SIFT_create()

        dir_path = '/workspace/src/autorace_idminer/image'

        self.img_intersection = cv2.imread(dir_path + '/intersection.png', 0)
        self.img_left = cv2.imread(dir_path + '/left.png', 0)
        self.img_right = cv2.imread(dir_path + '/right.png', 0)

        self.kp_intersection, self.des_intersection = self.sift.detectAndCompute(
            self.img_intersection, None
        )
        self.kp_left, self.des_left = self.sift.detectAndCompute(self.img_left, None)
        self.kp_right, self.des_right = self.sift.detectAndCompute(self.img_right, None)

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

        if self.off_left_right and self.off_intersection:
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

        matches_intersection = self.flann.knnMatch(des1, self.des_intersection, k=2)
        matches_left = self.flann.knnMatch(des1, self.des_left, k=2)
        matches_right = self.flann.knnMatch(des1, self.des_right, k=2)

        image_out_num = 1

        if not self.off_intersection:
            good_intersection = []
            for m, n in matches_intersection:
                if m.distance < 0.7*n.distance:
                    good_intersection.append(m)
            if len(good_intersection) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_intersection]).reshape(-1, 1, 2)
                dst_pts = np.float32([
                    self.kp_intersection[m.trainIdx].pt for m in good_intersection
                ]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matches_intersection = mask.ravel().tolist()

                mse = self.fnCalcMSE(src_pts, dst_pts)
                if mse < MIN_MSE_DECISION:
                    self.off_intersection = True
                    image_out_num = 2
                    self.get_logger().info('Detect intersection sign')

        if not self.off_left_right:
            good_left = []
            for m, n in matches_left:
                if m.distance < 0.5*n.distance:
                    good_left.append(m)
            if len(good_left) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_left]).reshape(-1, 1, 2)
                dst_pts = np.float32([
                    self.kp_left[m.trainIdx].pt for m in good_left
                ]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matches_left = mask.ravel().tolist()

                mse = self.fnCalcMSE(src_pts, dst_pts)
                if mse < MIN_MSE_DECISION:

                    self.left_count += 1

                    if self.left_count >= 3:

                        self.off_left_right = True
                        self.off_exit_timer = False
                        image_out_num = 3

                        lane_type = UInt8()
                        lane_type.data = 1
                        self.pub_lane_type.publish(lane_type)

                        self.get_logger().info('Detect left sign')

            else:
                matches_left = None

        if not self.off_left_right:
            good_right = []
            for m, n in matches_right:
                if m.distance < 0.5*n.distance:
                    good_right.append(m)
            if len(good_right) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_right]).reshape(-1, 1, 2)
                dst_pts = np.float32([
                    self.kp_right[m.trainIdx].pt for m in good_right
                ]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matches_right = mask.ravel().tolist()

                mse = self.fnCalcMSE(src_pts, dst_pts)
                if mse < MIN_MSE_DECISION:

                    self.right_count += 1

                    if self.right_count >= 3:

                        self.off_left_right = True
                        self.off_exit_timer = False
                        image_out_num = 4

                        lane_type = UInt8()
                        lane_type.data = 2
                        self.pub_lane_type.publish(lane_type)

                        self.get_logger().info('Detect right sign')

            else:
                matches_right = None

        if image_out_num == 1:
            self.pub_image.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(
                    cv_image_input,
                    'jpg'
                )
            )

        elif image_out_num == 2:
            draw_params_intersection = {
                'matchColor': (255, 0, 0),  
                'singlePointColor': None,
                'matchesMask': matches_intersection,
                'flags': 2
            }
            final_intersection = cv2.drawMatches(
                cv_image_input,
                kp1,
                self.img_intersection,
                self.kp_intersection,
                good_intersection,
                None,
                **draw_params_intersection
            )
            self.pub_image.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(
                    final_intersection,
                    'jpg'
                )
            )

        elif image_out_num == 3:
            draw_params_left = {
                'matchColor': (255, 0, 0),  
                'singlePointColor': None,
                'matchesMask': matches_left,
                'flags': 2
            }
            final_left = cv2.drawMatches(
                cv_image_input,
                kp1,
                self.img_left,
                self.kp_left,
                good_left,
                None,
                **draw_params_left
            )
            self.pub_image.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(
                    final_left,
                    'jpg'
                )
            )

        elif image_out_num == 4:
            draw_params_right = {
                'matchColor': (255, 0, 0),  
                'singlePointColor': None,
                'matchesMask': matches_right,
                'flags': 2
            }
            final_right = cv2.drawMatches(
                cv_image_input,
                kp1,
                self.img_right,
                self.kp_right,
                good_right,
                None,
                **draw_params_right
            )
            self.pub_image.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(
                    final_right,
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