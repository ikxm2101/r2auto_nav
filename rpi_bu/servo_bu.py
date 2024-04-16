from example_interfaces.srv import SetBool
import rclpy
from rclpy.node import Node
from time import sleep, time

from .servo_device import ServoDevice

class ServoService(Node):

    def __init__(self):
        super().__init__('servo_service')
        self.srv = self.create_service(SetBool, 'set_bool', self.set_bool_callback)
        self.get_logger('Servo service').info('f{Up!}')

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
        self.left_servo = ServoDevice(left_servo_pin)
        self.right_servo = ServoDevice(right_servo_pin)
        
        # set initial angles of servo
        self.left_servo.set_angle(0)
        self.right_servo.set_angle(180)
        
        sleep(1)
        
        # launch!
        self.left_servo.set_angle(110)
        self.right_servo.set_angle(70)
        
        sleep(1)
        
        # reset servo positions
        for turntime in range(0,110):
            self.left_servo.set_angle(110-turntime)
            self.right_servo.set_angle(70+turntime)
        time.sleep(0.01)
        self.left_servo.close()
        self.right_servo.close()
        
        
        # TODO: change throw routine
        
        # self.pi = pigpio.pi()
        
        
        # try:
        #     self.pi.set_PWM_frequency(left_servo, 50) # set frequency to 50Hz for servo
        #     self.pi.set_PWM_frequency(right_servo, 50) # set frequency to 50Hz for servo
        #     self.pi.set_PWM_dutycycle(left_servo, 8) # 2.5% duty cycle, 2.5 / 100 * 255~7, 0 degrees
        #     self.pi.set_PWM_dutycycle(right_servo, 33) # starting position

        #     sleep(1)

        #     # go more than halfway
        #     self.pi.set_PWM_dutycycle(left_servo, 28)
        #     self.pi.set_PWM_dutycycle(right_servo, right_servo)

        #     sleep(1)

        #     # go all the way
        #     self.pi.set_PWM_dutycycle(left_servo, 33)
        #     self.pi.set_PWM_dutycycle(right_servo, 8)

        #     sleep(2)

        #     for i in range(1, 33-8+1):
        #         self.pi.set_PWM_dutycycle(left_servo, 33-i)
        #         self.pi.set_PWM_dutycycle(right_servo, 8+i)
        #         sleep(0.1)

        # except KeyboardInterrupt:
        #     pass

        # self.pi.set_mode(left_servo, pigpio.INPUT)
        # self.pi.set_mode(right_servo, pigpio.INPUT)
        # self.pi.stop()

def main():
    rclpy.init()

    servo_service = ServoService()
    servo_service.throw_routine() # test throw routine
    rclpy.spin(servo_service)

    
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()