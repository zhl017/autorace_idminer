#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    calibration_arg = DeclareLaunchArgument(
        'calibration',
        default_value='False',
        description='type [Ture, False]'
    )

    calibration = LaunchConfiguration('calibration')

    detect_param = os.path.join(
        get_package_share_directory('autorace_idminer'),
        'param',
        'traffic_light.yaml'
    )

    node = Node(
        package='autorace_idminer',
        executable='detect_traffic_light',
        name='detect_traffic_light',
        output='screen',
        parameters=[
            {'calibration': calibration},
            detect_param
        ],
        remappings=[
            ('/camera/input/compressed', '/image_raw/compressed'),
            ('/camera/output/compressed', '/detect/traffic_light/compressed'),
        ],
    )

    return LaunchDescription([
        calibration_arg,
        node,
    ])
