#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Bool

class ControlLane(Node):

    def __init__(self):
        super().__init__('control_lane')

        # Publisher
        self.pub_cmd = self.create_publisher(
            Twist,
            '/control/cmd_vel',
            1
        )

        # Subscription
        self.sub_lane = self.create_subscription(
            Float64,
            '/control/lane',
            self.cbFollowLane,
            1
        )
        self.sub_max_vel = self.create_subscription(
            Float64,
            '/control/max_vel',
            self.cbMaxVel,
            1
        )
        self.sub_aviod_cmd = self.create_subscription(
            Twist,
            '/avoid_control',
            self.cbAvoidCmd,
            1
        )
        self.sub_avoid_active = self.create_subscription(
            Bool,
            '/avoid_active',
            self.cbAvoidActive,
            1
        )

        self.last_error = 0
        self.MAX_VEL = 0.1

        self.avoid_active = True
        self.avoid_cmd = Twist()

        self.lane_received = False
        self.lane_cmd = Twist()
        self.timer = self.create_timer(0.5, self.cbTimer)

    def cbMaxVel(self, msg):
        self.MAX_VEL = msg.data

    def cbFollowLane(self, msg):
        if self.avoid_active:
            return
        
        self.lane_received = True

        center = msg.data
        error = center - 500

        Kp = 0.004
        Kd = 0.003

        angular_z = Kp * error + Kd * (error - self.last_error)
        self.last_error = error

        twist = Twist()
        twist.linear.x = min(self.MAX_VEL * (max(1 - abs(error) / 1200, 0) ** 2.2), 0.1)
        twist.angular.z = -max(angular_z, -1.8) if angular_z < 0 else -min(angular_z, 1.8)

        self.lane_cmd = twist

    def cbAvoidCmd(self, msg):
        self.avoid_cmd = msg

    def cbAvoidActive(self, msg):
        self.avoid_active = msg.data
        if self.avoid_active:
            self.get_logger().info('Avoidance mode activated.')
        else:
            self.get_logger().info('Avoidance mode deactivated. Returning to lane following.')

    def cbTimer(self):
        if self.avoid_active:
            self.pub_cmd.publish(self.avoid_cmd)
            return

        if not self.lane_received:
            self.pub_cmd.publish(Twist())
            return
        
        self.pub_cmd.publish(self.lane_cmd)

        self.lane_received = False

def main(args=None):
    rclpy.init(args=args)
    node = ControlLane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()