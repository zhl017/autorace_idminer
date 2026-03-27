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
from std_msgs.msg import UInt8, Float64

class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')

        # Get Detect Param
        self.fnDeclareParameter()

        # Publisher
        self.pub_image = self.create_publisher(
            CompressedImage,
            '/camera/output/compressed',
            1
        )
        self.pub_lane = self.create_publisher(
            Float64,
            '/detect/lane',
            1
        )
        self.pub_yellow_line_reliability = self.create_publisher(
            UInt8,
            '/detect/yellow_line_reliability',
            1
            )
        self.pub_white_line_reliability = self.create_publisher(
            UInt8,
            '/detect/white_line_reliability',
            1
            )
        if self.calibration:
            self.pub_image_calib = self.create_publisher(
                CompressedImage,
                '/detect/image_calib/compressed',
                1
            )
            self.pub_image_projected = self.create_publisher(
                CompressedImage,
                '/detect/image_projected/compressed',
                1
            )
            self.pub_white_lane = self.create_publisher(
                CompressedImage,
                '/detect/white_lane/compressed',
                1
            )    
            self.pub_yellow_lane = self.create_publisher(
                CompressedImage,
                '/detect/yellow_lane/compressed',
                1
            )

        self.sub_image = self.create_subscription(
            CompressedImage,
            '/camera/input/compressed',
            self.cbImageProjection,
            1
        )
        self.sub_lane_type = self.create_subscription(
            UInt8,
            '/detect/lane_type',
            self.cbLaneType,
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

        self.counter = 1

        self.mov_avg_left = np.empty((0, 3))
        self.mov_avg_right = np.empty((0, 3))

        self.reliability_white_line = 100
        self.reliability_yellow_line = 100

        self.lane_type = 0

    def cbMission(self, msg):
        return

    def cbLaneType(self, msg):
        self.lane_type = msg.data

    def cbImageProjection(self, msg):
        # drop the frame to 1/5 (6fps) because of the processing speed. This is up to your computer's operating power.
        # if self.counter % 2 != 0:
        #     self.counter += 1
        #     return
        # else:
        #     self.counter = 1

        np_image_original = np.frombuffer(msg.data, np.uint8)
        cv_image_original = cv2.imdecode(np_image_original, cv2.IMREAD_COLOR)

        top_x = self.top_x
        top_y = self.top_y
        bottom_x = self.bottom_x
        bottom_y = self.bottom_y

        if self.calibration:
            cv_image_calib = np.copy(cv_image_original)

            cv_image_calib = cv2.line(
                cv_image_calib,
                (160 - top_x, 180 - top_y),
                (160 + top_x, 180 - top_y),
                (0, 0, 255),
                1
            )
            cv_image_calib = cv2.line(
                cv_image_calib,
                (160 - bottom_x, 120 + bottom_y),
                (160 + bottom_x, 120 + bottom_y),
                (0, 0, 255),
                1
            )
            cv_image_calib = cv2.line(
                cv_image_calib,
                (160 + bottom_x, 120 + bottom_y),
                (160 + top_x, 180 - top_y),
                (0, 0, 255),
                1
            )
            cv_image_calib = cv2.line(
                cv_image_calib,
                (160 - bottom_x, 120 + bottom_y),
                (160 - top_x, 180 - top_y),
                (0, 0, 255),
                1
            )

            self.pub_image_calib.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(
                    cv_image_calib,
                    'jpg'
                )
            )
            
        cv_image_original = cv2.GaussianBlur(cv_image_original, (5, 5), 0)

        pts_src = np.array([
            [160 - top_x, 180 - top_y],
            [160 + top_x, 180 - top_y],
            [160 + bottom_x, 120 + bottom_y],
            [160 - bottom_x, 120 + bottom_y]
        ])

        pts_dst = np.array([[200, 0], [800, 0], [800, 600], [200, 600]])

        h, status = cv2.findHomography(pts_src, pts_dst)

        cv_image_homography = cv2.warpPerspective(cv_image_original, h, (1000, 600))

        triangle1 = np.array([[0, 599], [0, 340], [200, 599]], np.int32)
        triangle2 = np.array([[999, 599], [999, 340], [799, 599]], np.int32)
        black = (0, 0, 0)
        cv_image_homography = cv2.fillPoly(cv_image_homography, [triangle1, triangle2], black)

        if self.calibration:
            self.pub_image_projected.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(
                    cv_image_homography,
                    'jpg'
                )
            )

        cv_image = np.copy(cv_image_homography)

        white_fraction, cv_white_lane = self.fnMaskWhiteLane(cv_image)
        yellow_fraction, cv_yellow_lane = self.fnMaskYellowLane(cv_image)

        try:
            if yellow_fraction > 4000:
                self.left_fitx, self.left_fit = self.fnFitFromLine(
                    self.left_fit, cv_yellow_lane)
                self.mov_avg_left = np.append(
                    self.mov_avg_left,np.array([self.left_fit]), axis=0
                    )
                
            if white_fraction > 4000:
                self.right_fitx, self.right_fit = self.fnFitFromLine(
                    self.right_fit, cv_white_lane)
                self.mov_avg_right = np.append(
                    self.mov_avg_right, np.array([self.right_fit]), axis=0
                    )
        except Exception:
            if yellow_fraction > 3000:
                self.left_fitx, self.left_fit = self.fnSlidingWindown(cv_yellow_lane, 'left')
                self.mov_avg_left = np.array([self.left_fit])

            if white_fraction > 3000:
                self.right_fitx, self.right_fit = self.fnSlidingWindown(cv_white_lane, 'right')
                self.mov_avg_right = np.array([self.right_fit])

        MOV_AVG_LENGTH = 5

        try:
            self.left_fit = np.array([
                np.mean(self.mov_avg_left[::-1][:, 0][0:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_left[::-1][:, 1][0:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_left[::-1][:, 2][0:MOV_AVG_LENGTH])
                ])
            
            if self.mov_avg_left.shape[0] > 1000:
                self.mov_avg_left = self.mov_avg_left[0:MOV_AVG_LENGTH]
        except: pass

        try:
            self.right_fit = np.array([
                np.mean(self.mov_avg_right[::-1][:, 0][0:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_right[::-1][:, 1][0:MOV_AVG_LENGTH]),
                np.mean(self.mov_avg_right[::-1][:, 2][0:MOV_AVG_LENGTH])
                ])
            
            if self.mov_avg_right.shape[0] > 1000:
                self.mov_avg_right = self.mov_avg_right[0:MOV_AVG_LENGTH]
        except: pass

        self.fnMakeLane(cv_image, white_fraction, yellow_fraction)

    def fnMaskWhiteLane(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Hue_l = self.hue_white_l
        Hue_h = self.hue_white_h
        Saturation_l = self.saturation_white_l
        Saturation_h = self.saturation_white_h
        Lightness_l = self.lightness_white_l
        Lightness_h = self.lightness_white_h

        lower_white = np.array([Hue_l, Saturation_l, Lightness_l])
        upper_white = np.array([Hue_h, Saturation_h, Lightness_h])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        fraction_num = np.count_nonzero(mask)

        how_much_short = 0

        for i in range(0, 600):
            if np.count_nonzero(mask[i, ::]) > 0:
                how_much_short += 1

        how_much_short = 600 - how_much_short

        if how_much_short > 100:
            if self.reliability_white_line >= 5:
                self.reliability_white_line -= 5
        elif how_much_short <= 100:
            if self.reliability_white_line <= 99:
                self.reliability_white_line += 5

        msg_white_line_reliability = UInt8()
        msg_white_line_reliability.data = self.reliability_white_line
        self.pub_white_line_reliability.publish(msg_white_line_reliability)

        if self.calibration:
            self.pub_white_lane.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(
                    mask,
                    'jpg'
                )
            )

        return fraction_num, mask
    
    def fnMaskYellowLane(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Hue_l = self.hue_yellow_l
        Hue_h = self.hue_yellow_h
        Saturation_l = self.saturation_yellow_l
        Saturation_h = self.saturation_yellow_h
        Lightness_l = self.lightness_yellow_l
        Lightness_h = self.lightness_yellow_h

        lower_yellow = np.array([Hue_l, Saturation_l, Lightness_l])
        upper_yellow = np.array([Hue_h, Saturation_h, Lightness_h])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        fraction_num = np.count_nonzero(mask)
        
        how_much_short = 0

        for i in range(0, 600):
            if np.count_nonzero(mask[i, ::]) > 0:
                how_much_short += 1

        how_much_short = 600 - how_much_short

        if how_much_short > 100:
            if self.reliability_yellow_line >= 5:
                self.reliability_yellow_line -= 5
        elif how_much_short <= 100:
            if self.reliability_yellow_line <= 99:
                self.reliability_yellow_line += 5

        msg_yellow_line_reliability = UInt8()
        msg_yellow_line_reliability.data = self.reliability_yellow_line
        self.pub_yellow_line_reliability.publish(msg_yellow_line_reliability)

        if self.calibration:
            self.pub_yellow_lane.publish(
                self.cvBridge.cv2_to_compressed_imgmsg(
                    mask,
                    'jpg'
                )
            )

        return fraction_num, mask
    
    def fnFitFromLine(self, lane_fit, image):
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 150 #100
        lane_inds = (
            (nonzerox >
                (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] - margin)) &
            (nonzerox <
                (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] + margin))
                )

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        lane_fit = np.polyfit(y, x, 2)

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit

    def fnSlidingWindown(self, img_w, left_or_right):
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)

        out_img = np.dstack((img_w, img_w, img_w)) * 255

        midpoint = np.int_(histogram.shape[0] / 2)

        if left_or_right == 'left':
            lane_base = np.argmax(histogram[:midpoint])
        elif left_or_right == 'right':
            lane_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 20

        window_height = np.int_(img_w.shape[0] / nwindows)

        nonzero = img_w.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        x_current = lane_base

        margin = 50

        minpix = 50

        lane_inds = []

        for window in range(nwindows):
            win_y_low = img_w.shape[0] - (window + 1) * window_height
            win_y_high = img_w.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            cv2.rectangle(
                out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            good_lane_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) &
                (nonzerox < win_x_high)
                ).nonzero()[0]

            lane_inds.append(good_lane_inds)

            if len(good_lane_inds) > minpix:
                x_current = np.int_(np.mean(nonzerox[good_lane_inds]))

        lane_inds = np.concatenate(lane_inds)

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        try:
            lane_fit = np.polyfit(y, x, 2)
            self.lane_fit_bef = lane_fit
        except Exception:
            lane_fit = self.lane_fit_bef

        ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit
    
    def fnMakeLane(self, cv_image, white_fraction, yellow_fraction):
        centerx = None
        warp_zero = np.zeros((cv_image.shape[0], cv_image.shape[1], 1), dtype=np.uint8)

        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])

        lane_state = UInt8()

        if yellow_fraction > 3000:
            pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, ploty])))])
            cv2.polylines(color_warp_lines, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=25)

        if white_fraction > 3000:
            pts_right = np.array([np.transpose(np.vstack([self.right_fitx, ploty]))])
            cv2.polylines(color_warp_lines, np.int_([pts_right]), isClosed=False, color=(255, 255, 0), thickness=25)
        
        self.is_center_x_exist = True

        if self.lane_type == 1:
            self.reliability_white_line = 0
        elif self.lane_type == 2:
            self.reliability_yellow_line = 0

        if self.reliability_white_line > 50 and self.reliability_yellow_line > 50:   
            if white_fraction > 3000 and yellow_fraction > 3000:
                centerx = np.mean([self.left_fitx, self.right_fitx], axis=0)
                pts = np.hstack((pts_left, pts_right))
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 2

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

                cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

            if white_fraction > 3000 and yellow_fraction <= 3000:
                centerx = np.subtract(self.right_fitx, 320)
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 3

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

            if white_fraction <= 3000 and yellow_fraction > 3000:
                centerx = np.add(self.left_fitx, 320)
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 1

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

        elif self.reliability_white_line <= 50 and self.reliability_yellow_line > 50:
            centerx = np.add(self.left_fitx, 240)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

            lane_state.data = 1

            cv2.polylines(
                color_warp_lines,
                np.int_([pts_center]),
                isClosed=False,
                color=(0, 255, 255),
                thickness=12
                )

        elif self.reliability_white_line > 50 and self.reliability_yellow_line <= 50:
            centerx = np.subtract(self.right_fitx, 260)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

            lane_state.data = 3

            cv2.polylines(
                color_warp_lines,
                np.int_([pts_center]),
                isClosed=False,
                color=(0, 255, 255),
                thickness=12
                )
            
        else:
            self.is_center_x_exist = False

            lane_state.data = 0

            pass

        self.get_logger().info(f'Lane state: {lane_state.data}')

        final = cv2.addWeighted(cv_image, 1, color_warp, 0.2, 0)
        final = cv2.addWeighted(final, 1, color_warp_lines, 1, 0)

        if self.is_center_x_exist and centerx is not None:
            msg_desired_center = Float64()
            msg_desired_center.data = centerx.item(350)
            self.pub_lane.publish(msg_desired_center)

        self.pub_image.publish(
            self.cvBridge.cv2_to_compressed_imgmsg(
                final,
                'jpg'
            )
        )

    def fnDeclareParameter(self):

        parameter_descriptor_top = ParameterDescriptor(
            description='projection range top.',
            integer_range=[IntegerRange(
                from_value = 0,
                to_value = 120,
                step = 1)]
        )

        parameter_descriptor_bottom = ParameterDescriptor(
            description='projection range bottom.',
            integer_range=[IntegerRange(
                from_value = 0,
                to_value = 320,
                step = 1)]
        )

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

        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera.top_x', 120, parameter_descriptor_top),
                ('camera.top_y', 20, parameter_descriptor_top),
                ('camera.bottom_x', 150, parameter_descriptor_bottom),
                ('camera.bottom_y', 100, parameter_descriptor_bottom),

                ('detect.lane.white.hue_l', 0,
                    parameter_descriptor_hue),
                ('detect.lane.white.hue_h', 179,
                    parameter_descriptor_hue),
                ('detect.lane.white.saturation_l', 230,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.saturation_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.lightness_l', 0,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.lightness_h', 50,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.hue_l', 10,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.hue_h', 127,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.saturation_l', 95,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.saturation_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.lightness_l', 70,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.lightness_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('calibration', False)
            ]
        )

        self.top_x = self.get_parameter(
            'camera.top_x').get_parameter_value().integer_value
        self.top_y = self.get_parameter(
            'camera.top_y').get_parameter_value().integer_value
        self.bottom_x = self.get_parameter(
            'camera.bottom_x').get_parameter_value().integer_value
        self.bottom_y = self.get_parameter(
            'camera.bottom_y').get_parameter_value().integer_value

        self.hue_white_l = self.get_parameter(
            'detect.lane.white.hue_l').get_parameter_value().integer_value
        self.hue_white_h = self.get_parameter(
            'detect.lane.white.hue_h').get_parameter_value().integer_value
        self.saturation_white_l = self.get_parameter(
            'detect.lane.white.saturation_l').get_parameter_value().integer_value
        self.saturation_white_h = self.get_parameter(
            'detect.lane.white.saturation_h').get_parameter_value().integer_value
        self.lightness_white_l = self.get_parameter(
            'detect.lane.white.lightness_l').get_parameter_value().integer_value
        self.lightness_white_h = self.get_parameter(
            'detect.lane.white.lightness_h').get_parameter_value().integer_value

        self.hue_yellow_l = self.get_parameter(
            'detect.lane.yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter(
            'detect.lane.yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter(
            'detect.lane.yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter(
            'detect.lane.yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = self.get_parameter(
            'detect.lane.yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = self.get_parameter(
            'detect.lane.yellow.lightness_h').get_parameter_value().integer_value
        
        self.calibration = self.get_parameter(
            'calibration').get_parameter_value().bool_value
        
        if self.calibration:
            self.add_on_set_parameters_callback(self.GetParam)

    def GetParam(self, parameters):
        for param in parameters:
            self.get_logger().info(f'Parameter name: {param.name}')
            self.get_logger().info(f'Parameter value: {param.value}')
            self.get_logger().info(f'Parameter type: {param.type_}')
            if param.name == 'camera.top_x':
                self.top_x = param.value
            elif param.name == 'camera.top_y':
                self.top_y = param.value
            elif param.name == 'camera.bottom_x':
                self.bottom_x = param.value
            elif param.name == 'camera.bottom_y':
                self.bottom_y = param.value
            elif param.name == 'detect.lane.white.hue_l':
                self.hue_white_l = param.value
            elif param.name == 'detect.lane.white.hue_h':
                self.hue_white_h = param.value
            elif param.name == 'detect.lane.white.saturation_l':
                self.saturation_white_l = param.value
            elif param.name == 'detect.lane.white.saturation_h':
                self.saturation_white_h = param.value
            elif param.name == 'detect.lane.white.lightness_l':
                self.lightness_white_l = param.value
            elif param.name == 'detect.lane.white.lightness_h':
                self.lightness_white_h = param.value
            elif param.name == 'detect.lane.yellow.hue_l':
                self.hue_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.hue_h':
                self.hue_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.saturation_l':
                self.saturation_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.saturation_h':
                self.saturation_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.lightness_l':
                self.lightness_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.lightness_h':
                self.lightness_yellow_h = param.value
        self.get_logger().info(f'change: {self.top_x}')
        self.get_logger().info(f'change: {self.top_y}')
        self.get_logger().info(f'change: {self.bottom_x}')
        self.get_logger().info(f'change: {self.bottom_y}')
        self.get_logger().info(f'change: {self.hue_white_l}')
        self.get_logger().info(f'change: {self.hue_white_h}')
        self.get_logger().info(f'change: {self.saturation_white_l}')
        self.get_logger().info(f'change: {self.saturation_white_h}')
        self.get_logger().info(f'change: {self.lightness_white_l}')
        self.get_logger().info(f'change: {self.lightness_white_h}')
        self.get_logger().info(f'change: {self.hue_yellow_l}')
        self.get_logger().info(f'change: {self.hue_yellow_h}')
        self.get_logger().info(f'change: {self.saturation_yellow_l}')
        self.get_logger().info(f'change: {self.saturation_yellow_h}')
        self.get_logger().info(f'change: {self.lightness_yellow_l}')
        self.get_logger().info(f'change: {self.lightness_yellow_h}')
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()