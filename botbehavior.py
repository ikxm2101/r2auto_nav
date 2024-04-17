'''
State machine for turtlebot behavior to complete mission
'''
import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import time
import numpy as np

# nodes for navigation
from .botmapperplanner import BotMapperPlanner, LobbyCheck
from .botcontroller import move_straight, move_turn, get_curr_pos

# libs for mission tasks
from .lib.open_door import open_door
from .lib.door_mover import door_mover
from .lib.just_move import time_straight
from .lib.bucket_tools import move_to_bucket
from .lib.servo_client import launch_servo

# constants
lobby_map_coord = (1.8,2.85) # supposed to be 2.8
ipaddr = '192.168.xxx.xxx'

class BotBehaviorParameters():
    def __init__(self):
        self.curr_state = None
        self.states = {
            'init': 'init',
            'maze': 'maze',
            'exit': 'exit',
            'bucket': 'bucket'
        }

        self.costmap = np.array([])
        
class BotBehavior(Node):
    def __init__(self, name='behavior'):
        super().__init__(name)
        self.params = BotBehaviorParameters()
        self.init_topics()
    
    '''
    Getters and Setters
    '''
    def get_costmap(self):
        return self.params.costmap
    
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
    
    # '''
    # Callbacks and Updates
    # '''
    # def costmap_callback(self, msg):
    #     self.params.costmap = np.array(msg.data).reshape(
    #     msg.info.height, msg.info.width)
        


def search_occ():
    '''
    1. searches occupancy grid
    2. generates a path in map coordinates
    3. returns wps for vertices
    '''
    mapperplanner = BotMapperPlanner()
    
    try:
        rclpy.spin(mapperplanner)
    except SystemExit:
        rclpy.logging.get_logger(
            "Exiting mapper planner").info('Done callback')

    mapperplanner.destroy_node()
     # based on the current occupancy grid
    checked_goals = set() # to keep track of goals that have been checked for a path
    curr_pos = mapperplanner.params.pos_in_occ
    goal_pos, checked_goals = mapperplanner.get_goal(checked_goals)
    rclpy.logging.get_logger('Current Position').info(f'{curr_pos}')
    rclpy.logging.get_logger('Goal Position').info(f'{goal_pos}')

    path_map = None
    while path_map is None:
        path_map = mapperplanner.get_path(curr_pos, goal_pos) # get a new path here, None if no path can be found
        goal_pos, checked_goals = mapperplanner.get_goal(checked_goals) # get a new goal here
        if goal_pos == (0, 0): # map fully explored
            rclpy.logging_get_logger('Occupancy Grid').info(f'Fully explored')
            return
    path_wps = mapperplanner.get_waypoints(path_map)
    return path_map, path_wps

def path_to_door(goal_in_map=(1.8, 2.8)):
    
    mapperplanner = BotMapperPlanner()
    
    try:
        rclpy.spin(mapperplanner)
    except SystemExit:
        rclpy.logging.get_logger(
            "Exiting mapper planner").info('Done callback')
    
    mapperplanner.destroy_node()
    checked_goals = set()
    curr_pos = mapperplanner.params.pos_in_occ
    goal_pos = mapperplanner.map_to_occ(mapperplanner.map_coord_to_occ_origin(goal_in_map))

    path_map = None
    while path_map is None:
        path_map = mapperplanner.get_path(curr_pos, goal_pos) # get a new path here, None if no path can be found
        goal_pos, checked_goals = mapperplanner.get_goal(checked_goals) # get a new goal here
        if goal_pos == (0, 0): # map fully explored
            rclpy.logging_get_logger('Occupancy Grid').info(f'Fully explored')
            return
    path_wps = mapperplanner.get_waypoints(path_map)
    return path_map, path_wps

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher')
        self.publisher_ = self.create_publisher(Path, '/global_plan', 10)
        self.path = Path()

        # Initialize the path message
        self.path.header.frame_id = 'map'

    def publish_path(self, wps):
        # Clear any old path
        self.path.poses = []

        # Convert waypoints to poses and add them to the path
        for wp in wps:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            self.path.poses.append(pose)

        # Publish the path
        self.publisher_.publish(self.path)
    
def main(args=None):

    rclpy.init(args=args)
    # lobby check node
    lobbycheck = LobbyCheck()
    pathpub = PathPublisher()
    
    # time_straight(-0.1, 14)
    # search for lobby by priortising furthest y coordinates
    for num_searches in range(20): # just to timeout the search eventually
        rclpy.logging.get_logger('Search Number').info(f'{num_searches}')
        path_map, path_wps = search_occ()
        start_time = time.time() # start time for search
        pathpub.publish_path(path_map)
        for wps in path_wps:
            if lobbycheck.quit:
                break
            if (time.time() - start_time) > 5: # 20s to do new search based on new occupancy grid
                break
            print(f'current wp: {wps}')
            move_turn(wps)
            move_straight(wps)

            rclpy.spin_once(lobbycheck) # spin lobbycheck node once 
        if lobbycheck.quit:
            print('can see lobby')
            break
    
    print('-----------------------to lobby!-----------------------')  
    for _ in range(2): # run twice to confirm its at the lobby coordinates, as path_to_door() allows for some margin of error
        path_map, path_wps = path_to_door(goal_in_map=lobby_map_coord)
        for wps in path_wps:
            # print(x)
            print(f'current wp: {wps}')
            move_turn(wps)
            move_straight(wps)
    
    # print('-----------------------http call!-----------------------')
    # door = 0
    # try:
    #     door = open_door(ipaddr)
    # except Exception as e:
    #     print(e)
    # while door == 0:
    #     print('-----------------------request failed!-----------------------')
    #     time.sleep(1)
    #     try:
    #         door = open_door(ipaddr)
    #     except Exception as e:
    #         print(e)
    # door = 1
    
    # print(f'-----------------------going to door {door}!-----------------------')
    # # face front
    # move_turn((get_curr_pos().x, get_curr_pos().y+5))
    # door_mover()
    # if door == 1:
    #     turndeg = -5
    # elif door == 2:
    #     turndeg = 5
    # move_turn((get_curr_pos().x, get_curr_pos().y+5)) # to refresh curr_pos after door_mover
    # move_turn((get_curr_pos().x+turndeg, get_curr_pos().y))
    # time_straight(0.15, 4)
    

    # while(move_to_bucket(threshold=0.03, dist=0.17) is None):
    #     time_straight(0.15, 2)
        
    # print(f'-----------------------launching balls!-----------------------')
    # launch_servo()
    
    print('-----------------------finishing search!!!-----------------------')
    for num_searches in range(20): # just to timeout the search eventually
        rclpy.logging.get_logger('Search Number').info(f'{num_searches}')
        path_wps = search_occ()
        start_time = time.time() # start time for search
        for wps in path_wps:
            if (time.time() - start_time) > 20: # 20s to do new search based on new occupancy grid
                break
            print(f'current wp: {wps}')
            move_turn(wps, angular_speed_limit=2)
            move_straight(wps, linear_speed_limit=0.2, angular_speed_limit=2)
    print(f'-----------------------done!-----------------------')
    
    
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
