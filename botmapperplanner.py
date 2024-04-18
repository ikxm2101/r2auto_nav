'''
Mapper node for turtlebot
'''

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point

from array import array as Array
from queue import PriorityQueue

import math
import scipy
import cv2 as cv
import numpy as np


class BotMapperPlannerParameters():
    def __init__(self):
        '''
        Mapper Parameters
        '''
        # Map parameters to be obtained from 'map' topic
        self.resolution = 0.05
        self.origin_x = None
        self.origin_y = None
        self.prev_height = None  # in meters
        self.prev_width = None  # in meters
        self.height = None  # in meters
        self.width = None  # in cells

        # Maze parameters
        # Respect to map (in meters)
        self.maze_min_x = 0
        self.maze_max_x = 0
        self.maze_min_y = 0
        self.maze_max_y = 2.1

        # Respect to occupancy grid (in cells)
        self.occ_min_x = None
        self.occ_max_x = None
        self.occ_min_y = None
        self.occ_max_y = None

        # Occupancy parameters
        self.unknown = 1
        self.unoccupied = 2
        self.occupied = 3
        self.occ_bins = [-1, 0, 50, 100]
        self.occ_2d = np.array([])

        # Costmap construction parameters
        # Parameters for inflation layer
        self.inflation_inc = 5  # cost increment on the inflation layer
        self.inflation_min = 0  # minimum cost on the inflation layer
        self.inflation_max = 100  # maximum cost on the inflation layer
        self.inflation_radius = 0.10  # inflation radius in meters

        # Parameters for obstacle layer
        self.obstacle_initial = 50  # initial cost on the obstacle layer
        self.obstacle_inc = 5  # cost increment if cell observed to be an obstacle or decrement
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

        # Robot Position and Targets
        self.pos_in_occ = tuple()  # in occupancy grid coordinates
        self.goal_in_occ = tuple()  # in occupancy grid coordinates
        self.pos_in_map = tuple()  # in meters with respect to map origin
        self.goal_in_map = tuple()  # in meters with respect to map origin

        self.path_occ = list()
        self.path_map = list()
        self.path_wps = list()

        '''
        Planner Parameters
        '''
        self.wall_dilate_size = int((0.200 / 2) // self.resolution + 1)
        self.turning_cost = 7


class BotMapperPlanner(Node):
    '''
    Constructor
    '''

    def __init__(self, name: str = 'mapperplanner'):
        super().__init__(name)
        self.params = BotMapperPlannerParameters()
        self.init_topics()

    '''
    Costmap
    '''

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
            # self.params.inflation_layer = np.uint8(self.params.inflation_layer.reshape(self.params.height, self.params.width)) # occ_y-major

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
            # self.params.obstacle_layer = np.uint8(self.params.obstacle_layer.reshape(self.params.height, self.params.width)) # occ_y-major

    def update_inflationlayer(self):
        # initialise inflation layer with cost of zeroes
        self.params.inflation_layer = np.zeros(
            (self.params.height, self.params.width))

        radius = np.ceil((self.params.inflation_radius /
                         self.params.resolution))  # in grid cells

        # Get the indices of the occupied cells
        occupied_indices = np.where(self.occ_2d == self.params.occupied)
        # self.get_logger().info(f'occupied indices: {occupied_indices}')

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
        # self.get_logger().info(f'{min(self.params.inflation_layer.flatten())}, {max(self.params.inflation_layer.flatten())}')
        layer_msg.data = list(map(int, self.params.inflation_layer.flatten()))

        # Publish the inflation layer message
        self.inflation_publisher.publish(layer_msg)

    def update_obstaclelayer(self):
        # initialiise obstacle layer by creating a copy of map_2d
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
        # layer_msg.header.stamp = now.to_msg()
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

    def create_costmap(self, src: np.ndarray, dilate=2, threshold=40, inflation_radius=4, inflation_step=2, erode=2):
        array_eroded: np.ndarray = src.copy()
        array_eroded[src == -1] = 1
        array_eroded[src != -1] = 0
        array_eroded = cv.morphologyEx(
            np.uint8(array_eroded),
            cv.MORPH_CLOSE,
            cv.getStructuringElement(
                cv.MORPH_RECT, (2 * erode + 1, 2 * erode + 1))
        )  # Dilate then erode the unknown points, to get rid of stray wall and stray unoccupied points

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

    def update_costmap(self):
        raw_occ_2d = self.occ_data.reshape(
            self.params.height, self.params.width)
        self.params.costmap = self.create_costmap(
            raw_occ_2d,
            # the higher this is, the further i avoid straight walls
            dilate=int((0.273 / 2) // self.params.resolution + 1),
            inflation_radius=4,
            inflation_step=24,
            threshold=52,
            erode=6)

        # create costmap message
        # now = rclpy.time.Time()
        # costmap_msg = OccupancyGrid()
        # # header
        # costmap_msg.header.frame_id = 'map'
        # costmap_msg.header.stamp = now.to_msg()
        # # info
        # costmap_msg.info.resolution = self.params.resolution
        # costmap_msg.info.height = self.params.height
        # costmap_msg.info.width = self.params.width
        # costmap_msg.info.origin.position.x = self.params.origin_x
        # costmap_msg.info.origin.position.y = self.params.origin_y
        # # data
        # costmap_msg.data = Array(
        #     'b', self.params.costmap.ravel().astype(np.int8))

        # # add the costs on the inflation layer
        # self.params.costmap = self.params.inflation_layer + self.params.obstacle_layer
        # # Clip the costs to the range 0 to 127
        # self.params.costmap = np.clip(self.params.costmap, 0, 127)

        # ## data
        # # self.get_logger().info(f'{min(self.params.costmap.flatten())}, {max(self.params.costmap.flatten())}')
        # costmap_msg.data = Array(map(int, self.params.costmap.flatten())) # Flatten the costmap and convert it to a list of integers

        # Publish the costmap message
        # self.costmap_publisher.publish(costmap_msg)

    '''
    Coordinate Conversions:
    a) Origin is the bottom left corner of the map given in meters
    b) All coordinates (both occupancy grid & map) are given in tuples (x, y)
    
    Sequence to convert from occupancy grid coordinates to map coordinates
    1. occ_to_map (units: cell * meter/cell = meter)
    2. map_coord_to_map_origin (units: meter + meter = meter)
    
    - occ_to_map returns map coordinates with respect to occupancy grid origin (0, 0)
    - map_coord_to_map_origin returns map coordinates with respect to map origin (origin_x, origin_y)
    
    Sequence to convert from map coordinates to occupancy grid coordinates
    1. map_coord_to_occ_origin (units: meter + meter = meters)
    2. map_to_occ (units: meter / meter/cell = cell)
    
    - map_coord_to_occ_origin returns map coordinates with respect to occupancy grid origin (0, 0)
    - map_to_occ returns occupancy grid coordinates with respect to occupancy grid origin (0, 0)
    
    '''

    def map_coord_to_occ_origin(self, map_coord: tuple):
        return (
            map_coord[0] - self.params.origin_x,
            map_coord[1] - self.params.origin_y
        )

    def map_coord_to_map_origin(self, map_coord: tuple):
        return (
            map_coord[0] + self.params.origin_x,
            map_coord[1] + self.params.origin_y
        )

    # Function to convert from occupancy grid coordinates (cells) to map coordinates (meters)
    def occ_to_map(self, occgrid_coord: tuple):
        return (
            occgrid_coord[0] * self.params.resolution,
            occgrid_coord[1] * self.params.resolution
        )

    # Function to convert from map coordinates (meters) to occupancy grid coordinates (cells)
    def map_to_occ(self, map_coord: tuple):
        return (
            round(map_coord[0] /
                  self.params.resolution),
            round(map_coord[1] /
                  self.params.resolution)
        )

    '''
    Navigation Targets
    '''

    def get_directions(self, occ_x, occ_y):
        directions = (
            (occ_x + 1,  occ_y + 0),
            (occ_x + 0,  occ_y + 1),
            (occ_x + -1, occ_y + 0),
            (occ_x + 0,  occ_y + -1),
            (occ_x + 1,  occ_y + 1),
            (occ_x + -1, occ_y + 1),
            (occ_x + -1, occ_y + -1),
            (occ_x + 1,  occ_y + -1)
        )
        return directions

    def check_for_target(self, occ_x, occ_y, target):
        # maze boundaries in occupancy grid coordinates
        # dereference maze boundaries from origin
        deref_maze_min_x, deref_maze_min_y = self.map_coord_to_map_origin(
            (self.params.maze_min_x, self.params.maze_min_y))
        deref_maze_max_x, deref_maze_max_y = self.map_coord_to_map_origin(
            (self.params.maze_max_x, self.params.maze_max_y))
        # convert map coordinates to occupancy grid coordinates
        self.params.occ_min_x, self.params.occ_min_y = self.map_to_occ(
            (deref_maze_min_x, deref_maze_min_y))
        self.params.occ_max_x, self.params.occ_max_y = self.map_to_occ(
            (deref_maze_max_x, deref_maze_max_y))
        # check if within maze

        def is_in_bounds(neigh_occ_x, neigh_occ_y):
            return (neigh_occ_x < self.params.width and neigh_occ_y < self.params.height
                    and neigh_occ_x > 0 and neigh_occ_y > 0)
        # check if is target

        def is_target(neigh_occ_x, neigh_occ_y):
            return self.occ_2d[neigh_occ_y][neigh_occ_x] == target
        # check for all directions
        for direction in self.get_directions(occ_x, occ_y):
            neigh_occ_x, neigh_occ_y = direction
            if is_in_bounds(neigh_occ_x, neigh_occ_y) and is_target(neigh_occ_x, neigh_occ_y):
                return True
        return False

    def get_goal(self, checked_goals: set, lobby_coord: tuple = None):
        # self.params.goal_in_occ = tuple()
        # furthest_coord = 0

        # def is_in_bounds(occ_x, occ_y):
        #     return (occ_x < self.params.width and occ_y < self.params.height
        #             and occ_x > 0 and occ_y > 0)

        def is_unoccupied(occ_x, occ_y):
            return (self.occ_2d[occ_y][occ_x] == self.params.unoccupied)

        def is_next_to_unknown(occ_x, occ_y):
            return self.check_for_target(occ_x, occ_y, self.params.unknown)

        # # find furthest y coordinate first
        # for occ_y in range(self.params.height):
        #     for occ_x in range(self.params.width):
        #         if occ_x + occ_y > furthest_coord and is_unoccupied(occ_x, occ_y) and is_next_to_unknown(occ_x, occ_y) and is_in_bounds(occ_x, occ_y):
        #             furthest_coord = occ_x + occ_y
        #             self.params.goal_in_occ = (occ_x, occ_y)

        # goal_coord_to_occ_origin = self.occ_to_map(self.params.goal_in_occ)
        # self.params.goal_in_map = self.map_coord_to_map_origin(
        #     goal_coord_to_occ_origin)

        # goal_in_occ = (0, 0)
        # occ_x, occ_y = occ_x_start, occ_y_start

        # while goal_in_occ == (0, 0) and occ_y >= 0:

        #     # start searching for goal starting from last found goal
        #     if occ_y == occ_y_start:

        #         # search from the right first (furthest x)
        #         # make sure occ_x remains in bounds
        #         while goal_in_occ == (0, 0) and occ_x_start >= 0:
        #             if is_unoccupied(occ_x_start, occ_y) \
        #                     and is_next_to_unknown(occ_x_start, occ_y):
        #                 goal_in_occ = (occ_x_start, occ_y)
        #             occ_x_start -= 1  # decrement occ_x_start to check all the columns

        #     # if the entire row for the same occ_y is explored already,
        #     # start looking through the rows starting from furthest x
        #     else:
        #         occ_x = self.params.width - 1
        #         while goal_in_occ == (0, 0) and occ_x >= 0:
        #             if is_unoccupied(occ_x, occ_y) \
        #                     and is_next_to_unknown(occ_x, occ_y):
        #                 goal_in_occ = (occ_x, occ_y)
        #             occ_x -= 1
        #     occ_y -= 1

        # return goal_in_occ

        # # check if exploration is complete
        # if not self.params.goal_in_occ:
        #     self.get_logger().info('No goal found')
        #     return False
        # return True

        possible_goals = set()
        furthest_goal = (0, 0)

        # get possible goals
        for occ_y in range(self.params.height):
            for occ_x in range(self.params.width):
                if is_unoccupied(occ_x, occ_y) and is_next_to_unknown(occ_x, occ_y):
                    possible_goals.add((occ_x, occ_y))

        # filter possible goals with checked goals
        possible_goals = possible_goals - checked_goals

        if lobby_coord:
            lobby_coord_in_occ = self.map_to_occ(
                self.map_coord_to_occ_origin(lobby_coord))
            # filter possible goals with lobby_coord_in_occ (any less than or equal y position)

            def in_maze(goal):
                return goal[1] <= lobby_coord_in_occ[1]

            possible_goals_copy = possible_goals.copy()
            possible_goals = set(in_maze(goal)
                                 for goal in possible_goals_copy if in_maze)

        if possible_goals == set():
            return furthest_goal, checked_goals

        # find largest y for any x
        for goal in possible_goals:
            if not lobby_coord: # if still finding lobby
                if goal[1] > furthest_goal[1]:
                    furthest_goal = goal
            
            else:
                furthest_goal = goal
                
        checked_goals.add(furthest_goal)
        
        return furthest_goal, checked_goals

    def get_robot(self):
        # return self.params.pos_in_map
        while True:
            try:
                tf = self.tfBuffer.lookup_transform(
                    'map',
                    'base_link',
                    rclpy.time.Time()
                )
                break
            except LookupException as e1:
                self.get_logger().info('lookup')
            except ConnectivityException as e2:
                self.get_logger().info('connectivity')
            except ExtrapolationException as e3:
                self.get_logger().info('extrapolation')

        # get pos in map and pos in occ
        self.params.pos_in_map = (
            tf.transform.translation.x, tf.transform.translation.y)
        self.params.pos_in_occ = self.map_to_occ(
            self.map_coord_to_occ_origin(self.params.pos_in_map))

        temp_curr_pos = self.params.pos_in_occ
        # check if robot is in occupied
        is_in_occupied = self.occ_2d[int(temp_curr_pos[1])][int(
            temp_curr_pos[0])] == self.params.occupied

        if is_in_occupied:
            unoccupied_indexes = np.transpose(np.nonzero(self.occ_2d == 2))
            closest_dist = 999999
            closest_pos = (0, 0)
            for pos in unoccupied_indexes:
                pos = (pos[1], pos[0])
                calc_dist = self.heuristic(temp_curr_pos, pos)
                if calc_dist < closest_dist:
                    closest_dist = calc_dist
                    closest_pos = pos

            self.pos_in_occ = closest_pos
    '''
    Path Planning
    '''

    def heuristic(self, curr_pos, next_pos):
        (x1, y1) = curr_pos
        (x2, y2) = next_pos
        return round(math.sqrt(abs(x1 - x2)**2 + abs(y1 - y2)**2), 2)

    # a-star algorithm to find optimal path from start to goal
    def a_star_search(self, graph, start, goal):

        def cost_to_goal(curr_pos, goal):
            dist_x = abs(goal[0] - curr_pos[0])
            dist_y = abs(goal[1] - curr_pos[1])
            return math.sqrt(dist_x**2 + dist_y**2)

        # needed so robot don't bang wall
        def is_within_goal(curr_pos, goal):
            is_within_goal_x = (
                curr_pos[0] < (goal[0] + self.params.wall_dilate_size) and curr_pos[0] > (goal[0] - self.params.wall_dilate_size))
            is_within_goal_y = (
                curr_pos[1] < (goal[1] + self.params.wall_dilate_size) and curr_pos[1] > (goal[1] - self.params.wall_dilate_size))
            return is_within_goal_x and is_within_goal_y

        def neighbors(curr_pos, graph):

            def is_in_bounds(curr_pos):
                # if curr_pos given is with reference to occ
                temp_pos_in_occ = curr_pos
                # # if curr_pos given is with reference to map
                # temp_pos_to_occ_origin = self.map_coord_to_occ_origin(curr_pos)
                # temp_pos_in_occ = self.map_to_occ(temp_pos_to_occ_origin)
                if temp_pos_in_occ[1] > self.params.height \
                        or temp_pos_in_occ[0] > self.params.width \
                        or temp_pos_in_occ[1] < 0 \
                        or temp_pos_in_occ[0] < 0:
                    return False
                return True

            def is_not_blocked(curr_pos):
                # if curr_pos given is with reference to occ
                temp_pos_in_occ = tuple(map(round, curr_pos))
                # # if curr_pos given is with reference to map
                # temp_pos_to_occ_origin = self.map_coord_to_occ_origin(curr_pos)
                # temp_pos_in_occ = self.map_to_occ(temp_pos_to_occ_origin)
                if graph[temp_pos_in_occ[1]][temp_pos_in_occ[0]] == self.params.unoccupied:
                    return True
                return False

            (x, y) = curr_pos
            directions = [
                (x + 1, y),  # right
                (x, y - 1),  # down
                (x - 1, y),  # left
                (x, y + 1),  # up
                (x + 1, y + 1),
                (x + 1, y - 1),
                (x - 1, y + 1),
                (x - 1, y - 1),
            ]
            directions = filter(is_in_bounds, directions)
            directions = filter(is_not_blocked, directions)
            return directions

        self.get_logger().info(f'Start: {start}')
        self.get_logger().info(f'Goal: {goal}')

        # priority queue to store waypoints
        waypoints = PriorityQueue()
        waypoints.put((0, start))

        turning_cost = 2
        # dictionary to store the path (next_pos (key) -> curr_pos (value))
        came_from = {start: None}
        cost_so_far = {start: 0}  # dictionary to store the path cost
        final_pos = tuple()

        while not waypoints.empty():
            (_, curr_pos) = waypoints.get()

            if is_within_goal(curr_pos, goal):
                final_pos = curr_pos  # update final position of path to current position
                self.get_robot()  # update robot positionn
                self.get_logger().info(
                    f'Start/Robot position in occ: {self.params.pos_in_occ}')
                self.get_logger().info(f'Goal Found! : {goal}')
                self.get_logger().info(f'End of waypoint: {final_pos}')
                break

            for next_pos in neighbors(curr_pos, graph):
                new_cost = cost_so_far[curr_pos] + \
                    cost_to_goal(curr_pos, next_pos)
                prev_pos = came_from[curr_pos]
                if prev_pos != None:
                    # next_direction = (int(next[0] - current[0]), int(next[1] - current[1]))
                    # current_direction = (int(current[0] - prev[0]), int(current[1] - prev[1]))
                    next_direction = (
                        next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])
                    current_direction = (
                        curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                    if current_direction != next_direction:

                        new_cost += turning_cost

                new_cost = round(new_cost, 2)

                if next_pos not in cost_so_far or new_cost < round(cost_so_far[next_pos], 2):

                    priority = new_cost + self.heuristic(goal, next_pos)
                    priority += self.params.costmap[next_pos[1]][next_pos[0]]
                    cost_so_far[next_pos] = new_cost
                    waypoints.put((priority, next_pos))
                    came_from[next_pos] = curr_pos

        return came_from, cost_so_far, final_pos

    def get_waypoints(self, path_array: list):  # path_array in rviz coord
        """Generate waypoints (vertices) from shortest path

        Args:
            path_array (list): List of each individual point (in (x,y) rviz coordinates) of the shortest path

        Returns:
            waypoints_array (list): List of waypoints (in rviz coordinates format)
        """
        waypoints_array = []
        prev_diff = (round((path_array[1][0] - path_array[0][0]), 2),
                     round((path_array[1][1] - path_array[0][1]), 2))
        for i in range(1, len(path_array)):
            current_diff = (round((path_array[i][0] - path_array[i-1][0]), 2), round(
                (path_array[i][1] - path_array[i-1][1]), 2))
            if prev_diff != current_diff:
                prev_diff = current_diff
                print(prev_diff)
                print(current_diff)
                waypoints_array.append(path_array[i-1])
        waypoints_array.append(path_array[-1])
        return waypoints_array

    # get path using a_star algorithm
    def get_path(self, start, goal):
        # get path starting from goal position with reference to occ
        # graph to be used for a star (can be costmap or occ_2d)
        graph = self.occ_2d
        start_pos = start  # start position for a star
        goal_pos = goal  # goal position for a star
        came_from, cost_so_far, final_pos = self.a_star_search(
            graph, start_pos, goal_pos)

        if final_pos == tuple():
            self.get_logger().info('No path to goal found!')
            return None

        last_pos = final_pos
        path_occ = [last_pos]
        prev_pos = tuple()

        # access came_from dictionary to get path_occ starting from last_pos
        while prev_pos != start_pos:  # while previous position is not the start position of the robot
            # update previous position based on current position
            prev_pos = came_from[last_pos]
            last_pos = prev_pos
            # insert previous position to path_occ as first index
            path_occ.insert(0, last_pos)
        # get path with reference to map using path_occ
        path_map_to_map_origin = [(self.map_coord_to_map_origin(
            self.occ_to_map(occ_pos))) for occ_pos in path_occ]

        path_map = []
        for point_in_path_map in path_map_to_map_origin:
            path_map.append(
                (round(point_in_path_map[0], 4), (round(point_in_path_map[1], 4))))
        return path_map

    '''
    Publishes
    '''

    def publish_goal_in_map(self):
        self.get_logger().info(
            f'Publishing goal_in_map: {self.params.goal_in_map}')
        now = rclpy.time.Time()
        goal_in_map_msg = PoseStamped()
        goal_in_map_msg.header.frame_id = 'map'
        goal_in_map_msg.header.stamp = now.to_msg()
        goal_in_map_msg.pose.position.x = float(self.params.goal_in_map[0])
        goal_in_map_msg.pose.position.y = float(self.params.goal_in_map[1])
        goal_in_map_msg.pose.orientation.x = 0.0
        goal_in_map_msg.pose.orientation.y = 0.0
        goal_in_map_msg.pose.orientation.y = 0.0
        goal_in_map_msg.pose.orientation.w = 1.0
        self.goal_in_map_publisher.publish(goal_in_map_msg)

    def publish_goal_in_occ(self):
        self.get_logger().info(
            f'Publishing goal_in_occ: {self.params.goal_in_occ}')
        goal_in_occ_msg = PoseStamped()
        goal_in_occ_msg.pose.position.x = float(self.params.goal_in_occ[0])
        goal_in_occ_msg.pose.position.y = float(self.params.goal_in_occ[1])
        self.goal_in_occ_publisher.publish(goal_in_occ_msg)

    '''
    BotMapper Topics
    '''

    def init_topics(self):
        '''
        Subscriptions
        '''
        self.costmap_subscription = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.costmap_callback,
            qos_profile_sensor_data
        )
        self.costmap_subscription  # prevent unused variable warning

        # self.pose_subscription = self.create_subscription(
        #     PoseStamped,
        #     '/currentpose',
        #     self.pose_callback,
        #     qos_profile_sensor_data
        # )
        # self.pose_subscription  # prevent unused variable warining

        # self.goal_reached_subscription = self.create_subscription(
        #     String,
        #     '/goal_reached',
        #     self.goal_reached_callback,
        #     10
        # )
        # self.goal_reached_subscription  # prevent unused variable warning

        # self.costmap_subscription = self.create_subscription(
        #     OccupancyGrid,
        #     '/global_costmap/costmap',
        #     self.costmap_callback,
        #     qos_profile_system_default
        # )
        # self.costmap_subscription  # prevent unused variable warning

        '''
        Publishers
        '''
        # self.inflation_publisher = self.create_publisher(
        #     OccupancyGrid,
        #     '/map/inflation_layer',
        #     qos_profile_sensor_data
        # )

        # self.obstacle_publisher = self.create_publisher(
        #     OccupancyGrid,
        #     '/map/obstacle_layer',
        #     qos_profile_sensor_data
        # )

        # self.costmap_publisher = self.create_publisher(
        #     OccupancyGrid,
        #     '/global_costmap/costmap',
        #     qos_profile_system_default
        # )

        # self.goal_in_map_publisher = self.create_publisher(
        #     PoseStamped,
        #     '/goal_in_map',
        #     10
        # )

        # self.goal_in_occ_publisher = self.create_publisher(
        #     PoseStamped,
        #     '/goal_in_occ',
        #     10
        # )

        '''
        Transforms
        '''
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(
            self.tfBuffer, self)

    '''
    Callbacks and Updates
    '''

    def costmap_callback(self, msg):
        # self.get_logger().info('In occ callback')

        # update map parameters
        self.update_params(msg)

        # process map data
        # self.flatten_layers()
        # self.reshape_layers()

        self.costmap_data = np.array(msg.data)

        # costmap data into 2d array
        self.params.costmap = self.costmap_data.copy()
        self.params.costmap = self.params.costmap.reshape(
            self.params.height, self.params.width)

        # occupancy data based on costmap
        self.occ_data = self.costmap_data.copy()
        self.occ_data[self.costmap_data == -
                      1] = self.params.unknown  # assign unknown
        self.occ_data[self.costmap_data >=
                      0] = self.params.unoccupied  # assign unoccupied
        # assign lethal obstacle
        self.occ_data[self.costmap_data == 100] = self.params.occupied

        # occ_counts go from 1 to 3 so we can use uint8 (1 - Unknown, 2 - Unoccupied,, 3 - Occupied)
        # reshape into 2D
        self.occ_2d = np.uint8(self.occ_data.reshape(
            self.params.height, self.params.width))  # occ_y-major
        # self.occ_2d = np.uint8(oc2.reshape(msg.info.height,msg.info.width,order='F')) # column-major

        # update costmap
        # self.update_inflationlayer()
        # self.update_obstaclelayer()], self.occ_counts[0][2], total_bins))
        # self.update_costmap()

        # get robot position
        self.get_robot()  # get robot position

        # publish to topics
        # self.publish_goal_in_map()
        # self.publish_goal_in_occ()

        raise SystemExit  # exit node

    # def goal_reached_callback(self, msg):
    #     if msg.data == 'goal_in_map reached':
    #         self.get_goal()
    #         self.publish_goal_in_map()

    # def pose_callback(self, msg):
    #     self.params.pos_in_map = (msg.pose.position.x, msg.pose.position.y)

    def update_params(self, msg):
        self.params.prev_width = self.params.width
        self.params.prev_height = self.params.height
        self.params.width = msg.info.width
        self.params.height = msg.info.height
        self.params.origin_x = msg.info.origin.position.x
        self.params.origin_y = msg.info.origin.position.y


class LobbyCheck(BotMapperPlanner):
    def __init__(self, lobby_map_coord=(1.8, 2.7)):
        super().__init__("lobbycheck")

        self.quit = 0  # flag to quit search
        self.lobby_map_coord = lobby_map_coord
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.costmap_callback,
            10)
        self.subscription  # prevent unused variable warning

    def costmap_callback(self, msg):
        self.update_params(msg)
        occdata = np.array(msg.data)
        cdata = occdata.reshape(msg.info.height, msg.info.width)
        # cdata[cdata == 100] = -1
        cdata[cdata >= 0] = 1
        cdata[cdata == -1] = 0
        explored_indexes = np.nonzero(cdata)  # gives row, col -> y,x
        odata_lobby_coord = self.map_to_occ(
            self.map_coord_to_occ_origin(self.lobby_map_coord))

        if (odata_lobby_coord[0]-3 in explored_indexes[1]
            and odata_lobby_coord[0]+3 in explored_indexes[1]
                and odata_lobby_coord[1]+5 in explored_indexes[0]):  # lobby is found when x is withing 3 cells a
            self.quit = 1
            print('found lobby, quit search')


def main(args=None):
    rclpy.init(args=args)
    mapperplanner = BotMapperPlanner()
    rclpy.spin(mapperplanner)
    mapperplanner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
