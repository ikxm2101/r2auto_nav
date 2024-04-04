'''
Brief: Navigates to waypoints using Twist()
Subscriptions: 
Publishes: 
'''
import rclpy
from rclpy.time import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist

import math
import cmath
import numpy as np

# CONSTANTS
WAYPOINT = [2.5, 2.5]
WAYPOINT_X = WAYPOINT[0]
WAYPOINT_Y = WAYPOINT[1]
SPEED = 0.05
ROTATECHANGE = 0.1


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypointnavigator')

        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.rotated = False


        self.pose_subscription = self.create_subscription(
            PoseStamped,
            'currentpose',
            self.pose_callback,
            qos_profile_sensor_data
        )
        self.pose_subscription

        self.cmdvel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

    def odom_callback(self, msg):
        self.get_logger().info('In odom_callback')
        orientation_quat = msg.pose.pose.orientation
        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_quat.x,
                                                                orientation_quat.y,
                                                                orientation_quat.z,
                                                                orientation_quat.w)

    def pose_callback(self, msg):
        self.get_logger().info('In pose_callback')

        def waypoint_reached(currentposition, threshold=0.25):
            x_dist = abs(currentposition.x - WAYPOINT_X)
            y_dist = abs(currentposition.y - WAYPOINT_Y)
            return (x_dist < threshold), (y_dist < threshold)

        cmd_vel = Twist()
        reached = waypoint_reached(msg.pose.position)
        x_reached, y_reached = reached[0], reached[1]

        if all(reached):
            self.get_logger().info(
                f'x_reached: {x_reached}, y_reached: {y_reached}')
            cmd_vel.linear.x = 0.0
        elif not x_reached:
            self.get_logger().info(
                f'x_reached: {x_reached}, y_reached:{y_reached}')
            cmd_vel.linear.x = SPEED
        elif not y_reached:
            if not self.rotated:
                self.rotatebot(90)
                self.rotated = True
            self.get_logger().info(
                f'x_reached: {x_reached}, y_reached: {y_reached}')
            cmd_vel.linear.x = SPEED

        self.cmdvel_publisher.publish(cmd_vel)

    def rotatebot(self, rot_angle):
        # self.get_logger().info('In rotatebot')
        # create Twist object
        cmd_vel = Twist()

        # get current yaw angle
        current_yaw = self.yaw
        # log the info
        self.get_logger().info('Current: %f' % math.degrees(current_yaw))
        # we are going to use complex numbers to avoid problems when the angles go from
        # 360 to 0, or from -180 to 180
        c_yaw = complex(math.cos(current_yaw), math.sin(current_yaw))
        # calculate desired yaw
        target_yaw = current_yaw + math.radians(rot_angle)
        # convert to complex notation
        c_target_yaw = complex(math.cos(target_yaw), math.sin(target_yaw))
        self.get_logger().info('Desired: %f' % math.degrees(cmath.phase(c_target_yaw)))
        # divide the two complex numbers to get the change in direction
        c_change = c_target_yaw / c_yaw
        # get the sign of the imaginary component to figure out which way we have to turn
        c_change_dir = np.sign(c_change.imag)
        # set linear speed to zero so the TurtleBot rotates on the spot
        cmd_vel.linear.x = 0.0
        # set the direction to rotate
        cmd_vel.angular.z = c_change_dir * ROTATECHANGE
        # start rotation
        self.cmdvel_publisher.publish(cmd_vel)

        # we will use the c_dir_diff variable to see if we can stop rotating
        c_dir_diff = c_change_dir
        # self.get_logger().info('c_change_dir: %f c_dir_diff: %f' % (c_change_dir, c_dir_diff))
        # if the rotation direction was 1.0, then we will want to stop when the c_dir_diff
        # becomes -1.0, and vice versa
        while (c_change_dir * c_dir_diff > 0):
            # allow the callback functions to run
            self.get_logger().info('hello')
            rclpy.spin_once(self, timeout_sec=1)
            self.get_logger().info('hello1')
            current_yaw = self.yaw
            # convert the current yaw to complex form
            c_yaw = complex(math.cos(current_yaw), math.sin(current_yaw))
            # self.get_logger().info('Current Yaw: %f' % math.degrees(current_yaw))
            # get difference in angle between current and target
            c_change = c_target_yaw / c_yaw
            # get the sign to see if we can stop
            c_dir_diff = np.sign(c_change.imag)
            self.get_logger().info('hello2')
            # self.get_logger().info('c_change_dir: %f c_dir_diff: %f' % (c_change_dir, c_dir_diff))

        self.get_logger().info('End Yaw: %f' % math.degrees(current_yaw))
        # set the rotation speed to 0
        cmd_vel.angular.z = 0.0
        # stop the rotation
        self.cmdvel_publisher.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    waypointnavigator = WaypointNavigator()
    rclpy.spin(waypointnavigator)
    waypointnavigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
