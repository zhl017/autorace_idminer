#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    nav2_pkg_share = get_package_share_directory('turtlebot3_navigation2')

    param_file = os.path.join(
        get_package_share_directory('autorace_idminer'),
        'param',
        'nav.yaml'
    )

    detect_tunnel_node = Node(
        package='autorace_idminer',
        executable='detect_tunnel',
        name='detect_tunnel',
        output='screen',
        parameters=[param_file],
        remappings=[
            ('/camera/input/compressed', '/image_raw/compressed'),
            ('/camera/output/compressed', '/detect/sign/compressed'),
        ],
    )

    return LaunchDescription([
        detect_tunnel_node,
    ])
