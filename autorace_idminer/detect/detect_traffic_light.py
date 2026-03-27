#!/usr/bin/env python3

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
from std_msgs.msg import Bool, UInt8

class DetectTrafficLight(Node):

    def __init__(self):
        super().__init__('detect_traffic_light')

        # Get Detect Param
        self.fnDeclareParameter()

        # Publisher
        self.pub_image = self.create_publisher(
            CompressedImage,
            '/camera/output/compressed',
            1
        )
        self.pub_avoid_active = self.create_publisher(
            Bool,
            '/avoid_active',
            1
        )
        self.pub_mission = self.create_publisher(
            UInt8,
            '/mission',
            1
        )
        if self.calibration:
            self.pub_red = self.create_publisher(
                CompressedImage,
                '/detect/traffic_light/red/compressed',
                1
            )
            self.pub_yellow = self.create_publisher(
                CompressedImage,
                '/detect/traffic_light/yellow/compressed',
                1
            )
            self.pub_green = self.create_publisher(
                CompressedImage,
                '/detect/traffic_light/green/compressed',
                1
            )

        # Subscription
        self.sub_image = self.create_subscription(
            CompressedImage,
            '/camera/input/compressed',
            self.cbGetImage,
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

        self.off_traffic = True
        self.get_image = False

        self.counter = 1
        self.stop_count = 0
        self.green_count = 0
        self.yellow_count = 0
        self.red_count = 0

        self.timer = self.create_timer(0.1, self.cbTimer)

    def cbMission(self, msg):
        if msg.data == self.mission.TrafficLight.value:
            self.off_traffic = False
            self.get_image = False
            self.stop_count = 0
            self.green_count = 0
            self.yellow_count = 0
            self.red_count = 0
            self.get_logger().info('START TrafficLight')
        else:
            self.off_traffic = True
            self.get_image = False
            self.stop_count = 0
            self.green_count = 0
            self.yellow_count = 0
            self.red_count = 0

    def cbTimer(self):
        if self.get_image and not self.off_traffic:
            self.fnFindTrafficLight()

    def cbGetImage(self, msg):
        # drop the frame to 1/5 (6fps) because of the processing speed. This is up to your computer's operating power.
        # if self.counter % 3 != 0:
        #     self.counter += 1
        #     return
        # else:
        #     self.counter = 1

        if self.off_traffic:
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        self.get_image = True

    def fnFindTrafficLight(self):

        cv_image_mask = self.fnMaskGreen()
        cv_image_mask = cv2.GaussianBlur(cv_image_mask, (5, 5), 0)
        green = self.fnFindCircle(cv_image_mask, 'green')
        if green == 1 or green == 5:
            self.stop_count = 0
            self.green_count += 1

        else:
            self.green_count = 0

            cv_image_mask = self.fnMaskYellow()
            cv_image_mask = cv2.GaussianBlur(cv_image_mask, (5, 5), 0)
            yellow = self.fnFindCircle(cv_image_mask, 'yellow')
            if yellow == 2:
                self.yellow_count += 1
            else:
                self.yellow_count = 0

                cv_image_mask = self.fnMaskRed()
                cv_image_mask = cv2.GaussianBlur(cv_image_mask, (5, 5), 0)

                red = self.fnFindCircle(cv_image_mask, 'red')
                if red == 3:
                    self.red_count += 1
                elif red == 4:
                    self.red_count = 0
                    self.stop_count += 1
                else:
                    self.red_count = 0
                    self.stop_count = 0

        if self.green_count >=5:
            if not self.calibration:
                self.off_traffic = True

                avoid_active = Bool()
                avoid_active.data = False
                self.pub_avoid_active.publish(avoid_active)

                mission = UInt8()
                mission.data = self.mission.Intersection.value
                self.pub_mission.publish(mission)

                self.get_logger().info('END TrafficLight')

            cv2.putText(self.cv_image,"GREEN", (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
            
            
        if self.yellow_count >= 8:
            cv2.putText(self.cv_image,"YELLOW", (self.point_x, self.point_y)
                        ,cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))
            
        if self.red_count >= 8:
            cv2.putText(self.cv_image,"RED", (self.point_x, self.point_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            
        self.pub_image.publish(self.cvBridge.cv2_to_compressed_imgmsg(self.cv_image, 'jpg'))

    def fnMaskGreen(self):
        image = np.copy(self.cv_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([self.hue_green_l, self.saturation_green_l, self.lightness_green_l])
        upper = np.array([self.hue_green_h, self.saturation_green_h, self.lightness_green_h])

        mask = cv2.inRange(hsv, lower, upper)
        
        if self.calibration:
            self.pub_green.publish(self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))

        mask = cv2.bitwise_not(mask)
        return mask
    
    def fnMaskYellow(self):
        image = np.copy(self.cv_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([self.hue_yellow_l, self.saturation_yellow_l, self.lightness_yellow_l])
        upper = np.array([self.hue_yellow_h, self.saturation_yellow_h, self.lightness_yellow_h])

        mask = cv2.inRange(hsv, lower, upper)
        
        if self.calibration:
            self.pub_yellow.publish(self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))

        mask = cv2.bitwise_not(mask)
        return mask
    
    def fnMaskRed(self):
        image = np.copy(self.cv_image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array([self.hue_red_l, self.saturation_red_l, self.lightness_red_l])
        upper = np.array([self.hue_red_h, self.saturation_red_h, self.lightness_red_h])

        mask = cv2.inRange(hsv, lower, upper)
        
        if self.calibration:
            self.pub_red.publish(self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg'))

        mask = cv2.bitwise_not(mask)
        return mask
    
    def fnFindCircle(self, mask, color):
        detect_result = False
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 600
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.filterByConvexity = True
        params.minConvexity = 0.6
        detector = cv2.SimpleBlobDetector_create(params)
        keypts = detector.detect(mask)

        col1 = 180
        col2 = 270
        col3 = 305

        low1 = 50
        low2 = 170
        low3 = 170

        for i in range(len(keypts)):
            self.point_x = int(keypts[i].pt[0])
            self.point_y = int(keypts[i].pt[1])
            if self.point_x > col1 and self.point_x < col2 and self.point_y > low1 and self.point_y < low2:
                if color == 'green':
                    detect_result = 1
                elif color == 'yellow':
                    detect_result = 2
                elif color == 'red':
                    detect_result = 3
            elif self.point_x > col2 and self.point_x < col3 and self.point_y > low1 and self.point_y < low3:
                if color == 'red':
                    detect_result = 4
                elif color == 'green':
                    detect_result = 5
            else:
                detect_result = 6

        return detect_result
    
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

        self.declare_parameter(
            'yellow.hue_l', 0, parameter_descriptor_hue)
        self.declare_parameter(
            'yellow.hue_h', 179, parameter_descriptor_hue)
        self.declare_parameter(
            'yellow.saturation_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.lightness_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'yellow.lightness_h', 255, parameter_descriptor_saturation_lightness)

        self.declare_parameter(
            'green.hue_l', 0, parameter_descriptor_hue)
        self.declare_parameter(
            'green.hue_h', 179, parameter_descriptor_hue)
        self.declare_parameter(
            'green.saturation_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'green.saturation_h', 255, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'green.lightness_l', 0, parameter_descriptor_saturation_lightness)
        self.declare_parameter(
            'green.lightness_h', 255, parameter_descriptor_saturation_lightness)

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

        self.hue_yellow_l = self.get_parameter(
            'yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter(
            'yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter(
            'yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter(
            'yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = self.get_parameter(
            'yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = self.get_parameter(
            'yellow.lightness_h').get_parameter_value().integer_value

        self.hue_green_l = self.get_parameter(
            'green.hue_l').get_parameter_value().integer_value
        self.hue_green_h = self.get_parameter(
            'green.hue_h').get_parameter_value().integer_value
        self.saturation_green_l = self.get_parameter(
            'green.saturation_l').get_parameter_value().integer_value
        self.saturation_green_h = self.get_parameter(
            'green.saturation_h').get_parameter_value().integer_value
        self.lightness_green_l = self.get_parameter(
            'green.lightness_l').get_parameter_value().integer_value
        self.lightness_green_h = self.get_parameter(
            'green.lightness_h').get_parameter_value().integer_value

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

            elif param.name == 'yellow.hue_l':
                self.hue_yellow_l = param.value
                self.get_logger().info(f'yellow.hue_l set to: {param.value}')
            elif param.name == 'yellow.hue_h':
                self.hue_yellow_h = param.value
                self.get_logger().info(f'yellow.hue_h set to: {param.value}')
            elif param.name == 'yellow.saturation_l':
                self.saturation_yellow_l = param.value
                self.get_logger().info(f'yellow.saturation_l set to: {param.value}')
            elif param.name == 'yellow.saturation_h':
                self.saturation_yellow_h = param.value
                self.get_logger().info(f'yellow.saturation_h set to: {param.value}')
            elif param.name == 'yellow.lightness_l':
                self.lightness_yellow_l = param.value
                self.get_logger().info(f'yellow.lightness_l set to: {param.value}')
            elif param.name == 'yellow.lightness_h':
                self.lightness_yellow_h = param.value
                self.get_logger().info(f'yellow.lightness_h set to: {param.value}')

            elif param.name == 'green.hue_l':
                self.hue_green_l = param.value
                self.get_logger().info(f'green.hue_l set to: {param.value}')
            elif param.name == 'green.hue_h':
                self.hue_green_h = param.value
                self.get_logger().info(f'green.hue_h set to: {param.value}')
            elif param.name == 'green.saturation_l':
                self.saturation_green_l = param.value
                self.get_logger().info(f'green.saturation_l set to: {param.value}')
            elif param.name == 'green.saturation_h':
                self.saturation_green_h = param.value
                self.get_logger().info(f'green.saturation_h set to: {param.value}')
            elif param.name == 'green.lightness_l':
                self.lightness_green_l = param.value
                self.get_logger().info(f'green.lightness_l set to: {param.value}')
            elif param.name == 'green.lightness_h':
                self.lightness_green_h = param.value
                self.get_logger().info(f'green.lightness_h set to: {param.value}')

        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = DetectTrafficLight()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
