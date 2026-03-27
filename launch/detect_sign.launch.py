#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    mission_arg = DeclareLaunchArgument(
        'mission',
        default_value='intersection',
        description='type [intersection, ' \
                            'construction,' \
                            ' parking,' \
                            ' level_crossing,' \
                            ' tunnel]'
    )

    mission = LaunchConfiguration('mission')

    node = Node(
        package='autorace_idminer',
        executable=['detect_', mission],
        name=['detect_', mission],
        output='screen',
        remappings=[
            ('/camera/input/compressed', '/image_raw/compressed'),
            ('/camera/output/compressed', '/detect/sign/compressed'),
        ],
    )

    return LaunchDescription([mission_arg, node])