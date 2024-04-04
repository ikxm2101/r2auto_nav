'''
'''

import scipy.ndimage
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

import rclpy.time
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

import scipy
import numpy as np


class BotMapperParameters():
    def __init__(self):

        # Map parameters to be obtained from 'map' topic
        self.resolution = 0.05
        self.origin_x = None
        self.origin_y = None
        self.prev_height = None
        self.prev_width = None
        self.height = None
        self.width = None

        # Maze parameters
        # self.min_x = # in meters
        # self.max_x = # in meters
        # self.min_y = # in meters
        # self.max_y = # in meters

        # Occupancy parameters
        self.occ_bins = [-1, 0, 50, 100]
        self.unknown = 1
        self.unoccupied = 2
        self.occupied = 3

        # Costmap parameters
        ## Parameters for inflation layer
        self.inflation_inc = 5  # cost increment on the inflation layer for every nearby occupied cell within the inflation radius
        self.inflation_min = 0  # minimum cost on the inflation layer
        self.inflation_max = 100  # maximum cost on the inflation layer
        self.inflation_radius = 0.10  # inflation radius in meters

        ## Parameters for obstacle layer
        self.obstacle_initial = 50  # initial cost on the obstacle layer
        self.obstacle_inc = 5 # cost increment if cell observed to be an obstacle or decrement if observed to be free
        self.obstacle_min = 0  # minimum cost on the obstacle layer
        self.obstacle_unknown = 50  # cost for unknown cells on the obstacle layer
        self.obstacle_max = 100  # maximum cost on the obstacle layer

        self.unoccupied_thres = 20  # threshold for considering a cell as unoccupied
        self.obstacle_thres = 80  # threshold for considering a cell as an obstacle

        # Parameters for costmap
        self.costmap_min = 0
        self.costmap_max = 127
    
        # Layers
        self.inflation_layer = np.array([])
        self.obstacle_layer = np.array([])
        self.costmap = np.array([])

class BotMapper(Node):
    def __init__(self, name='mapper'):
        super().__init__(name)
        self.params = BotMapperParameters()
        self.init_topics()

    def flatten_layers(self):
        self.params.inflation_layer = self.params.inflation_layer.flatten()
        self.params.obstacle_layer = self.params.obstacle_layer.flatten()

    def reshape_layers(self):
        if self.params.inflation_layer.any():
            zoom_y = self.params.height / self.params.prev_height
            zoom_x = self.params.width / self.params.prev_width

            # Use scipy's zoom function to resize the array
            self.params.inflation_layer = scipy.ndimage.zoom(
                self.params.inflation_layer, (zoom_y, zoom_x))

            # Ensure the values are still in the correct range after interpolation
            self.params.inflation_layer = np.clip(
                self.params.inflation_layer, self.params.inflation_min, self.params.inflation_max)
            self.params.inflation_layer = np.uint8(self.params.inflation_layer)
            # self.params.inflation_layer = np.uint8(self.params.inflation_layer.reshape(self.params.height, self.params.width)) # row-major

        if self.params.obstacle_layer.any():
            zoom_y = self.params.height / self.params.prev_height
            zoom_x = self.params.width / self.params.prev_width

            # Use scipy's zoom function to resize the array
            self.params.obstacle_layer = scipy.ndimage.zoom(
                self.params.obstacle_layer, (zoom_y, zoom_x))

            # Ensure the values are still in the correct range after interpolation
            self.params.obstacle_layer = np.clip(
                self.params.obstacle_layer, self.params.inflation_min, self.params.inflation_max)
            self.params.obstacle_layer = np.uint8(self.params.obstacle_layer)
            # self.params.obstacle_layer = np.uint8(self.params.obstacle_layer.reshape(self.params.height, self.params.width)) # row-major

    def update_inflationlayer(self):
        # initialise inflation layer with cost of zeroes
        self.params.inflation_layer = np.zeros(
            (self.params.height, self.params.width))

        radius = np.ceil((self.params.inflation_radius /
                         self.params.resolution))  # in grid cells

        # Get the indices of the occupied cells
        occupied_indices = np.where(self.occ_2d == self.params.occupied)
        self.get_logger().info(f'{occupied_indices}')

        for y, x in zip(*occupied_indices):
            # Range of cells to inflate around the occupied cell
            x_min = int(max(0, x - radius))
            x_max = int(min(self.params.width, x + radius + 1))
            y_min = int(max(0, y - radius))
            y_max = int(min(self.params.height, y + radius + 1))

            # Inflate the cells
            for i in range(y_min, y_max):
                for j in range(x_min, x_max):
                    # Calculate the distance from the occupied cell
                    dist = np.sqrt((i - y)**2 + (j - x)**2)
                    # Calculate the cost to be added to the cell
                    if dist <= radius:
                        # if distance is within the inflation radius
                        cost = self.params.inflation_inc
                        # Update the cost of the cell
                        self.params.inflation_layer[i, j] = min(
                            self.params.inflation_max, self.params.inflation_layer[i, j] + cost)

        # Publishing inflation layer
        layer_msg = OccupancyGrid()
        now = rclpy.time.Time()
        # Set the header
        layer_msg.header.frame_id = 'map'
        layer_msg.header.stamp = now.to_msg()
        # Set the info
        layer_msg.info.resolution = self.params.resolution
        layer_msg.info.width = self.params.width
        layer_msg.info.height = self.params.height
        layer_msg.info.origin.position.x = self.params.origin_x
        layer_msg.info.origin.position.y = self.params.origin_y
        # Flatten the layer and convert it to a list of integers
        self.get_logger().info(
            f'{min(self.params.inflation_layer.flatten())}, {max(self.params.inflation_layer.flatten())}')
        layer_msg.data = list(map(int, self.params.inflation_layer.flatten()))

        # Publish the inflation layer message
        self.inflation_publisher.publish(layer_msg)

    def update_obstaclelayer(self):
        # initialiise obstacle layer by creating a copy of occ_2d
        self.params.obstacle_layer = self.occ_2d.copy()
        # assign costs to cells accordingly
        self.params.obstacle_layer[self.params.obstacle_layer ==
                                   self.params.unknown] = self.params.obstacle_unknown
        self.params.obstacle_layer[self.params.obstacle_layer ==
                                   self.params.unoccupied] = self.params.obstacle_min
        self.params.obstacle_layer[self.params.obstacle_layer ==
                                   self.params.occupied] = self.params.obstacle_max

        # Publishing obstacle layer
        layer_msg = OccupancyGrid()
        now = rclpy.time.Time()
        # Set the header
        layer_msg.header.frame_id = 'map'
        layer_msg.header.stamp = now.to_msg()
        # Set the info
        layer_msg.info.resolution = self.params.resolution
        layer_msg.info.width = self.params.width
        layer_msg.info.height = self.params.height
        layer_msg.info.origin.position.x = self.params.origin_x
        layer_msg.info.origin.position.y = self.params.origin_y
        # Flatten the layer and convert it to a list of integers
        self.get_logger().info(
            f'{min(self.params.obstacle_layer.flatten())}, {max(self.params.obstacle_layer.flatten())}')
        layer_msg.data = list(map(int, self.params.obstacle_layer.flatten()))

        # Publish the obstacle layer message
        self.obstacle_publisher.publish(layer_msg)
    
    def update_costmap(self):
        # add the costs on the inflation layer
        self.params.costmap = self.params.inflation_layer + self.params.obstacle_layer
        # Clip the costs to the range 0 to 127
        self.params.costmap = np.clip(self.params.costmap, 0, 127)
        # Publishing costmap
        costmap_msg = OccupancyGrid()
        now = rclpy.time.Time()
        # Set the header
        costmap_msg.header.frame_id = 'map'
        costmap_msg.header.stamp = now.to_msg()
        # Set the info
        costmap_msg.info.resolution = self.params.resolution
        costmap_msg.info.width = self.params.width
        costmap_msg.info.height = self.params.height
        costmap_msg.info.origin.position.x = self.params.origin_x
        costmap_msg.info.origin.position.y = self.params.origin_y
        # Flatten the costmap and convert it to a list of integers
        self.get_logger().info(f'{min(self.params.costmap.flatten())}, {max(self.params.costmap.flatten())}')
        costmap_msg.data = list(map(int, self.params.costmap.flatten()))
        
        # Publish the costmap message
        self.costmap_publisher.publish(costmap_msg)
        
    def init_topics(self):
        '''
        Subscriptions
        '''
        self.occ_subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.occ_callback,
            qos_profile_sensor_data
        )
        self.occ_subscription  # prevent unused variable warning
        
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            'currentpose',
            self.pose_callback,
            qos_profile_sensor_data
        )

        '''
        Publishers
        '''
        self.inflation_publisher = self.create_publisher(
            OccupancyGrid,
            'map/inflation_layer',
            qos_profile_sensor_data
        )

        self.obstacle_publisher = self.create_publisher(
            OccupancyGrid,
            'map/obstacle_layer',
            qos_profile_sensor_data
        )

        self.costmap_publisher = self.create_publisher(
            OccupancyGrid,
            'map/costmap',
            qos_profile_sensor_data
        )
        
        # self.botInMap_publisher = self.create_publisher(
        # )

        # self.goalInMap_publisher = self.create_publisher(
        # )

    '''
    Callback Functions and Updates
    '''

    def occ_callback(self, msg):
        # self.get_logger().info('In occ callback')
        # self.flatten_layers()
        self.update_params(msg)
        # self.reshape_layers()

        occ_data = np.array(msg.data)  # create numpy array

        # compute histogram to get
        occ_counts, edges, self.occ_flat = scipy.stats.binned_statistic(
            occ_data, np.nan, statistic='count', bins=self.params.occ_bins)
        # self.get_logger().info('Unmapped: %i Unoccupied: %i Occupied: %i Total: %i' % (self.occ_counts[0][0], self.occ_counts[0][1], self.occ_counts[0][2], total_bins))

        # occ_counts go from 1 to 3 so we can use uint8 (1 - Unknown, 2 - Unoccupied,, 3 - Occupied)
        # reshape into 2D
        self.occ_2d = np.uint8(self.occ_flat.reshape(
            self.params.height, self.params.width))  # row-major
        # self.occdata = np.uint8(oc2.reshape(msg.info.height,msg.info.width,order='F')) # column-major

        self.update_inflationlayer()
        self.update_obstaclelayer()
        self.update_costmap()

    def pose_callback(self, msg):
        pass
    
    def update_params(self, msg):
        self.params.prev_width = self.params.width
        self.params.prev_height = self.params.height
        self.params.width = msg.info.width
        self.params.height = msg.info.height
        self.params.origin_x = msg.info.origin.position.x
        self.params.origin_y = msg.info.origin.position.y


def main(args=None):
    rclpy.init(args=args)
    mapper = BotMapper()
    rclpy.spin(mapper)
    mapper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
