#!/usr/bin/env python3

import math
from enum import Enum

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import UInt8, Float64

def fnCalcDistanceDot2Line(a, b, c, x0, y0):
    distance = abs(x0*a + y0*b + c)/math.sqrt(a*a + b*b)
    return distance

def fnCalcDistanceDot2Dot(x1, y1, x2, y2):
    distance = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
    return distance

def fnArrangeIndexOfPoint(arr):
    new_arr = arr[:]
    arr_idx = [0] * len(arr)
    for i in range(len(arr)):
        arr_idx[i] = i

    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] < arr[j]:
                buffer = arr_idx[j]
                arr_idx[j] = arr_idx[i]
                arr_idx[i] = buffer
                buffer = new_arr[j]
                new_arr[j] = new_arr[i]
                new_arr[i] = buffer
    return arr_idx

def fnCheckLinearity(point1, point2, point3):
    threshold_linearity = 50
    x1, y1 = point1
    x2, y2 = point3
    if x2-x1 != 0:
        a = (y2-y1)/(x2-x1)
    else:
        a = 1000
    b = -1
    c = y1 - a*x1
    err = fnCalcDistanceDot2Line(a, b, c, point2[0], point2[1])

    if err < threshold_linearity:
        return True
    else:
        return False

def fnCheckDistanceIsEqual(point1, point2, point3):

    threshold_distance_equality = 3
    distance1 = fnCalcDistanceDot2Dot(point1[0], point1[1], point2[0], point2[1])
    distance2 = fnCalcDistanceDot2Dot(point2[0], point2[1], point3[0], point3[1])
    std = np.std([distance1, distance2])

    if std < threshold_distance_equality:
        return True
    else:
        return False

class DetectLevel(Node):

    def __init__(self):
        super().__init__('detect_level')

        # Get Detect Param
        self.fnDeclareParameter()

        # Publisher
        self.pub_image = self.create_publisher(
            CompressedImage,
            '/camera/output/compressed',
            1
        )
        self.sub_mission = self.create_subscription(
            UInt8,
            '/mission',
            self.cbMission,
            1
        )
        self.pub_mission = self.create_publisher(
            UInt8,
            '/mission',
            1
        )
        self.pub_max_vel = self.create_publisher(
            Float64,
            '/control/max_vel',
            1
        )
        if self.calibration:
            self.pub_image_level = self.create_publisher(
                CompressedImage,
                '/detect/level/red/compressed',
                1
            )

        # Subscription
        self.sub_image = self.create_subscription(
            CompressedImage,
            '/camera/input/compressed',
            self.cbGetImage,
            1
        )

        self.mission = Enum('mission', 'TrafficLight Intersection Construction Parking LevelCrossing Tunnel')
        self.cvBridge = CvBridge()
        
        self.off_level = True
        self.get_image = False
        self.is_level_detected = False
        self.is_level_close = False
        self.is_level_opened = False
        self.counter = 1

        self.timer = self.create_timer(0.1, self.cbTimer)

    def cbMission(self, msg):
        if msg.data == self.mission.LevelCrossing.value:
            self.off_level = False
            self.get_image = False
            self.is_level_detected = False
            self.is_level_close = False
            self.is_level_opened = False
            self.get_logger().info('START LevelCrossing')
        else:
            self.off_level = True
            self.get_image = False
            self.is_level_detected = False
            self.is_level_close = False
            self.is_level_opened = False

    def cbTimer(self):
        if self.get_image:

            self.fnFindLevel()

            max_vel = Float64()

            if self.is_level_opened:
                self.off_level = True
                self.get_image = False

                max_vel.data = 0.1
                self.pub_max_vel.publish(max_vel)

                mission = UInt8()
                mission.data = self.mission.Tunnel.value
                self.pub_mission.publish(mission)

                self.get_logger().info('END LevelCrossing')
                return

            elif self.is_level_close:
                max_vel.data = 0.0
                self.pub_max_vel.publish(max_vel)
                return

            elif self.is_level_detected:
                max_vel.data = 0.05
                self.pub_max_vel.publish(max_vel)
                return

    def cbGetImage(self, msg):

        if self.off_level:
            return

        # drop the frame to 1/5 (6fps) because of the processing speed. This is up to your computer's operating power.
        # if self.counter % 3 != 0:
        #     self.counter += 1
        #     return
        # else:
        #     self.counter = 1

        np_arr = np.frombuffer(msg.data, np.uint8)
        self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        self.get_image = True

    def fnFindLevel(self):
        cv_image_mask = self.fnMaskRedOfLevel()
        cv_image_mask = cv2.GaussianBlur(cv_image_mask, (5, 5), 0)

        self.fnFindRectOfLevel(cv_image_mask)
    
    def fnMaskRedOfLevel(self):
        image = np.copy(self.cv_image)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Hue_l = self.hue_red_l
        Hue_h = self.hue_red_h
        Saturation_l = self.saturation_red_l
        Saturation_h = self.saturation_red_h
        Lightness_l = self.lightness_red_l
        Lightness_h = self.lightness_red_h

        lower_red = np.array([Hue_l, Saturation_l, Lightness_l])
        upper_red = np.array([Hue_h, Saturation_h, Lightness_h])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        mask = cv2.bitwise_not(mask)

        if self.calibration:
            self.pub_image_level.publish(self.cvBridge.cv2_to_compressed_imgmsg(mask, "jpg"))

        return mask

    def fnFindRectOfLevel(self, mask):

        params=cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 200
        params.maxArea = 3000
        params.filterByCircularity = True
        params.minCircularity = 0.3
        params.filterByConvexity = True
        params.minConvexity = 0.9

        det=cv2.SimpleBlobDetector_create(params)
        keypts=det.detect(mask)
        frame=cv2.drawKeypoints(self.cv_image,keypts,np.array([]),(0,255,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        mean_x = 0.0
        mean_y = 0.0

        if len(keypts) >= 3:
            for i in range(3):
                mean_x = mean_x + keypts[i].pt[0]/3
                mean_y = mean_y + keypts[i].pt[1]/3
            arr_distances = [0]*3
            for i in range(3):
                arr_distances[i] = fnCalcDistanceDot2Dot(mean_x, mean_y, keypts[i].pt[0], keypts[i].pt[1])

            idx1, idx2, idx3 = fnArrangeIndexOfPoint(arr_distances)
            frame = cv2.line(frame, (int(keypts[idx1].pt[0]), int(keypts[idx1].pt[1])), (int(keypts[idx2].pt[0]), int(keypts[idx2].pt[1])), (255, 0, 0), 5)
            frame = cv2.line(frame, (int(mean_x), int(mean_y)), (int(mean_x), int(mean_y)), (255, 255, 0), 5)
            point1 =  [int(keypts[idx1].pt[0]), int(keypts[idx1].pt[1]-1)]
            point2 = [int(keypts[idx3].pt[0]), int(keypts[idx3].pt[1]-1)]
            point3 = [int(keypts[idx2].pt[0]), int(keypts[idx2].pt[1]-1)]

            is_rects_linear = fnCheckLinearity(point1, point2, point3)
            is_rects_dist_equal = fnCheckDistanceIsEqual(point1, point2, point3)

            if (is_rects_linear == True or is_rects_dist_equal == True) and self.is_level_opened == False:
                distance_bar2car = 50 / fnCalcDistanceDot2Dot(point1[0], point1[1], point2[0], point2[1])
                if distance_bar2car > 1.2:
                    self.get_logger().info('detect')
                    self.is_level_detected = True

                else:
                    self.get_logger().info('stop')
                    self.is_level_close = True
                    self.is_level_detected = False

        elif len(keypts) < 1 and (self.is_level_detected == True or self.is_level_close == True):
            self.stop_bar_count = 0
            self.is_level_opened = True
            self.get_logger().info('go')

        self.pub_image.publish(self.cvBridge.cv2_to_compressed_imgmsg(frame, "jpg"))    

    def fnDeclareParameter(self):
            parameter_descriptor_hue = ParameterDescriptor(
                        description='hue parameter range',
                        integer_range=[IntegerRange(
                            from_value = 0,
                            to_value = 179,
                            step = 1)]
                    )

            parameter_descriptor_saturation_lightness = ParameterDescriptor(
                        description='saturation and lightness range',
                        integer_range=[IntegerRange(
                            from_value = 0,
                            to_value = 255,
                            step = 1)]
                    )

            self.declare_parameter(
                'red.hue_l', 0, parameter_descriptor_hue)
            self.declare_parameter(
                'red.hue_h', 179, parameter_descriptor_hue)
            self.declare_parameter(
                'red.saturation_l', 0, parameter_descriptor_saturation_lightness)
            self.declare_parameter(
                'red.saturation_h', 255, parameter_descriptor_saturation_lightness)
            self.declare_parameter(
                'red.lightness_l', 0, parameter_descriptor_saturation_lightness)
            self.declare_parameter(
                'red.lightness_h', 255, parameter_descriptor_saturation_lightness)

            self.declare_parameter('calibration', False)

            self.hue_red_l = self.get_parameter(
                'red.hue_l').get_parameter_value().integer_value
            self.hue_red_h = self.get_parameter(
                'red.hue_h').get_parameter_value().integer_value
            self.saturation_red_l = self.get_parameter(
                'red.saturation_l').get_parameter_value().integer_value
            self.saturation_red_h = self.get_parameter(
                'red.saturation_h').get_parameter_value().integer_value
            self.lightness_red_l = self.get_parameter(
                'red.lightness_l').get_parameter_value().integer_value
            self.lightness_red_h = self.get_parameter(
                'red.lightness_h').get_parameter_value().integer_value

            self.calibration = self.get_parameter(
                'calibration').get_parameter_value().bool_value
            
            if self.calibration:
                self.add_on_set_parameters_callback(self.GetParam)

    def GetParam(self, parameters):
        for param in parameters:
            if param.name == 'red.hue_l':
                self.hue_red_l = param.value
                self.get_logger().info(f'red.hue_l set to: {param.value}')
            elif param.name == 'red.hue_h':
                self.hue_red_h = param.value
                self.get_logger().info(f'red.hue_h set to: {param.value}')
            elif param.name == 'red.saturation_l':
                self.saturation_red_l = param.value
                self.get_logger().info(f'red.saturation_l set to: {param.value}')
            elif param.name == 'red.saturation_h':
                self.saturation_red_h = param.value
                self.get_logger().info(f'red.saturation_h set to: {param.value}')
            elif param.name == 'red.lightness_l':
                self.lightness_red_l = param.value
                self.get_logger().info(f'red.lightness_l set to: {param.value}')
            elif param.name == 'red.lightness_h':
                self.lightness_red_h = param.value
                self.get_logger().info(f'red.lightness_h set to: {param.value}')

        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = DetectLevel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()