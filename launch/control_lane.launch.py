#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    node = Node(
        package='autorace_idminer',
        executable='control_lane',
        name='control_lane',
        output='screen',
        remappings=[
            ('/control/lane', '/detect/lane'),
            ('/control/cmd_vel', '/cmd_vel')
        ]
    )
    return LaunchDescription([
        node,
    ])