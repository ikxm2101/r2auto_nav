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


class MapperPlannerParameters():
    def __init__(self):
        '''
        Mapper Parameters
        '''
        # Map parameters to be obtained from 'map' topic
        self.resolution = 0.05
        self.origin_x = None
        self.origin_y = None
        self.height = None  # in meters
        self.width = None  # in cells

        # Occupancy parameters
        self.unknown = 1
        self.unoccupied = 2
        self.occupied = 3
        self.occ_bins = [-1, 0, 50, 100]
        self.occ_2d = np.array([])

        # Costmap construction parameters
        # Parameters for inflation layer
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


class MapperPlanner(Node):
    '''
    Constructor
    '''

    def __init__(self, name: str = 'mapperplanner'):
        super().__init__(name)
        self.params = MapperPlannerParameters()
        self.init_topics()

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

        # check if in bounds
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

        def is_unoccupied(occ_x, occ_y):
            return (self.occ_2d[occ_y][occ_x] == self.params.unoccupied)

        def is_next_to_unknown(occ_x, occ_y):
            return self.check_for_target(occ_x, occ_y, self.params.unknown)

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

        '''
        Publishers
        '''

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

        # occ_counts go from 1 to 3 so we can use uint8 (1 - Unknown, 2 - Unoccupied, 3 - Occupied)
        # reshape into 2D
        self.occ_2d = np.uint8(self.occ_data.reshape(
            self.params.height, self.params.width))  # occ_y-major
        # self.occ_2d = np.uint8(oc2.reshape(msg.info.height,msg.info.width,order='F')) # column-major

        # get robot position
        self.get_robot()  # get robot position
        
        raise SystemExit  # exit node

    def update_params(self, msg):
        self.params.width = msg.info.width
        self.params.height = msg.info.height
        self.params.origin_x = msg.info.origin.position.x
        self.params.origin_y = msg.info.origin.position.y


class LobbyCheck(MapperPlanner):
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
    mapperplanner = MapperPlanner()
    rclpy.spin(mapperplanner)
    mapperplanner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
