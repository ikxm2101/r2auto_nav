from example_interfaces.srv import SetBool
import rclpy
from rclpy.node import Node
from time import sleep, time

from .servo_device import ServoDevice

class ServoService(Node):

    def __init__(self):
        super().__init__('servo_service')
        self.srv = self.create_service(SetBool, 'set_bool', self.set_bool_callback)
        self.get_logger().info(f'Up! at {time()}')

    def set_bool_callback(self, request, response):
        if request.data:
            self.throw_routine()
            response.success = True
            response.message = "Done at " + str(time())
        else:
            response.success = False
            response.message = "input is False"
        return response

    def throw_routine(self):
        
        # servo gpio pin numbers turtlebot's perspective
        left_servo_pin = 12
        right_servo_pin = 13 
        
        # left and right servo devices
        self.left_servo = ServoDevice(left_servo_pin, min_angle=0, max_angle=180, initial_angle=None)
        self.right_servo = ServoDevice(right_servo_pin, min_angle=180, max_angle=0, initial_angle=None)
        
        #  set initial angles of servo
        self.left_servo.set_angle(25)
        self.right_servo.set_angle(25)
        
        sleep(1)
        
        # launch!
        self.left_servo.set_angle(70)
        self.right_servo.set_angle(70)
        
        sleep(1)
        
        # reset servo positions
        for turntime in range(0,50):
            self.left_servo.set_angle(70-turntime)
            self.right_servo.set_angle(70-turntime)
            sleep(0.1)
        
        self.left_servo.close()
        self.right_servo.close()

def main():
    rclpy.init()

    servo_service = ServoService()
    # servo_service.throw_routine() # test throw routine
    rclpy.spin(servo_service)

    
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
