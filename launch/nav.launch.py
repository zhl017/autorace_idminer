#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    nav2_pkg_share = get_package_share_directory('turtlebot3_navigation2')

    map_file = os.path.join(
        get_package_share_directory('autorace_idminer'),
        'map',
        'map.yaml'
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_pkg_share, 'launch', 'navigation2.launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'True',
            'map': map_file
        }.items()
    )

    return LaunchDescription([
        nav2_launch,
    ])
