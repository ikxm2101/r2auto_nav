'''
Brief: Provides current pose to navigation nodes
Subscriptions: 
Publishes: robot's pose transformed onto map frame
'''

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class BotCurrentPose(Node):
    def __init__(self, name='currentpose'):
        super().__init__(name)
        self.init_topics()
        
        # Listener to transform robot's pose from odom and map frame
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)
        
        # A timer that periodically publishes the pose of the robot every 1 second
        self.timer = self.create_timer(1.0, self.publish_pose)

    def publish_pose(self):
        # Get the robot's position in the map frame
        
        ## 'base_link' is the coordinate frame attached to the center of the robot
        ## 'map' is the  coordinate frame attached to the map
        ## 'odom' is the coordinate frame attached to the robot's odometry
        
        try:
            now = rclpy.time.Time()
            tfPos = self.tfBuffer.lookup_transform(
                'map',
                'base_link',
                now
            )

            tfOrient = self.tfBuffer.lookup_transform(
                'odom',
                'base_link',
                now
            )
            
            # Create 'currentpose' using PoseStamped message
            currentpose = PoseStamped()
            currentpose.header.stamp = now.to_msg() # set the timestamp of the current pose
            currentpose.header.frame_id = 'map' # current pose is defined in the map frame
            # associate 'base_link' and 'map' frames for position
            currentpose.pose.position.x = tfPos.transform.translation.x
            currentpose.pose.position.y = tfPos.transform.translation.y
            # associate 'base_link' and 'odom' frames for oritentation
            currentpose.pose.orientation = tfOrient.transform.rotation

            self.pose_publisher.publish(currentpose)
            self.get_logger().info('Pose published successfully!')

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f'Failed to get robot pose: {e}')
            
    def init_topics(self):
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            qos_profile_sensor_data
        )
        self.map_subscription

        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.pose_publisher = self.create_publisher(
            PoseStamped,
            'currentpose',
            qos_profile_sensor_data
        )

    '''
    Callback functions
    '''
    def map_callback(self, msg):
        # self.get_logger().info('In map callback!')
        pass

    def odom_callback(self, msg):
        # self.get_logger().info('In odom callback!')
        pass

def main(args=None):
    rclpy.init(args=args)
    currentpose = BotCurrentPose()
    rclpy.spin(currentpose)
    currentpose.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
