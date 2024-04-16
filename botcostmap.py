import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from nav_msgs.msg import OccupancyGrid
import numpy as np

import cv2 as cv
from array import array as Array

def inflate(src: np.ndarray, dilate=2, threshold=40, inflation_radius=4, inflation_step=2, erode=2):
    array_eroded: np.ndarray = src.copy()
    array_eroded[src == -1] = 1
    array_eroded[src != -1] = 0
    array_eroded = cv.morphologyEx(
        np.uint8(array_eroded),
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_RECT, (2 * erode + 1, 2 * erode + 1))
    ) # Dilate then erode the unknown points, to get rid of stray wall and stray unoccupied points

    # create a copy of the occ_2d consisting of values, -1 to 100 indicating probability of occupancy
    array_dilated: np.ndarray = src.copy()
    # any cell with probability of possibly being occupied is assigned an inflation step    
    array_dilated[src >= threshold] = inflation_step
    # any cell with probability of not being occupied is assigned 0
    array_dilated[src < threshold] = 0
    array_dilated = cv.dilate(
        np.uint8(array_dilated),
        cv.getStructuringElement(
            cv.MORPH_RECT, (2 * dilate + 1, 2 * dilate + 1)),
    )

    wall_indexes = np.nonzero(array_dilated)

    array_inflated: np.ndarray = array_dilated.copy()

    for _ in range(inflation_radius):
        array_inflated = array_inflated + \
            cv.dilate(np.uint8(array_dilated),
                        cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
        array_dilated = array_inflated.copy()
        array_dilated[array_dilated != 0] = inflation_step

    array_inflated = np.int8(array_inflated)
    array_inflated[array_eroded == 1] = -1
    array_inflated[wall_indexes] = 100
    return array_inflated

class BotCostmap(Node):
    def __init__(self):
        super().__init__("costmap_pub")
        self.subscription = self.create_subscription(
            OccupancyGrid, "/map", self.listener_callback, qos_profile_sensor_data
        )
        self.publisher = self.create_publisher(
            OccupancyGrid, "/global_costmap/costmap", qos_profile_system_default
        )
        self.get_logger().info("Costmap Publishing...")

    def listener_callback(self, msg):
        occdata = np.array(msg.data)
        # np.savetxt("map.txt",occdata.reshape(msg.info.height, msg.info.width),"%d")
        odata = inflate(
            occdata.reshape(msg.info.height, msg.info.width),
            dilate=int((0.300 / 2) // msg.info.resolution + 1),
            inflation_radius=8,
            inflation_step=5,
            threshold=52,
            erode=6,
        )
        omap = OccupancyGrid()
        omap.data = Array("b", odata.ravel().astype(np.int8))
        omap.info = msg.info
        omap.header = msg.header
        self.publisher.publish(omap)


def main(args=None):
    rclpy.init(args=args)

    costmap = BotCostmap()
    rclpy.spin(costmap)

    costmap.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()