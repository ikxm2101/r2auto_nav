'''
State machine for turtlebot behavior to complete mission
'''
import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data

from nav_msgs.msg import OccupancyGrid


import time
import numpy as np

# nodes for navigation
from .botmapperplanner import BotMapperPlanner, LobbyCheck
from .botcontroller import move_straight, move_turn

# libs for mission tasks
from .lib.open_door import open_door
from .lib.door_mover import door_mover
from .lib.just_move import time_straight, time_turn
from .lib.bucket_tools import move_to_bucket
from .lib.servo_client import launch_servo

# constants
lobby_map_coord = (1.8,2.85) # supposed to be 2.8
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
        
    # r

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
    return path_wps

def main(args=None):

    rclpy.init(args=args)
    # lobby check node
    lobbycheck = LobbyCheck()
    # search for lobby by priortising furthest y coordinates
    for num_searches in range(20): # just to timeout the search eventually
        rclpy.logging.get_logger('Search Number').info(f'{num_searches}')
        path_wps = search_occ()
        start_time = time.time() # start time for search
        for wps in path_wps:
            if lobbycheck.quit:
                break
            if (time.time() - start_time) > 20: # 20s to do new search based on new occupancy grid
                break
            print(f'current wp: {wps}')
            move_turn(wps, angular_speed_limit=2)
            move_straight(wps, linear_speed_limit=0.2, angular_speed_limit=2)

            rclpy.spin_once(lobbycheck) # spin lobbycheck node once 
        if lobbycheck.quit:
            print('can see lobby')
            break
    
    print('-----------------------to lobby!-----------------------')  
    
    rclpy.shutdown()

    
if __name__ == "__main__":
    main()
