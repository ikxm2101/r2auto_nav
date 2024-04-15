'''
State machine for turtlebot behavior to complete mission
'''
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point

import time

# nodes for navigation
from .botmapperplanner import BotMapperPlanner
from .botcontroller import move_straight, move_turn

# libs for mission tasks

class BotBehaviorParameters():
    def __init__(self):
        self.curr_state = None
        self.states = {
            'init': 'init',
            'maze': 'maze',
            'exit': 'exit',
            'bucket': 'bucket'
        }

class BotBehavior(Node):
    def __init__(self, name='behavior'):
        super().__init__(name)
        self.params = BotBehaviorParameters()

def main(args=None):
    
    rclpy.init(args=args)
    while True:
        
        try: 
            mapperplanner = BotMapperPlanner()
            rclpy.spin(mapperplanner)
        except SystemExit:
            pass
        
        if not mapperplanner.params.goal_in_map:
            break
        
        path_wps = mapperplanner.params.path_map
        for wps in path_wps:
            print(f'current wp: {wps}')
            move_turn(wps)
            move_straight(wps)
        mapperplanner.destroy_node()
    rclpy.shutdown()    

if __name__ == "__main__":
    main()