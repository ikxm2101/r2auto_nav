from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

factory = PiGPIOFactory() # setup to use PiGPIO for hardware PWM

class ServoDevice(AngularServo):
  def __init__(self, servo_pin: int, initial_angle: int = 0, min_angle: int = 0, max_angle: int = 180,
               min_pulse_width: float = 0.5/1000, max_pulse_width: float = 2.5/1000,
               frame_width: float = 20/1000, pin_factory = factory) -> None:
    super().__init__(servo_pin, initial_angle, min_angle, max_angle,
                     min_pulse_width, max_pulse_width, 
                     frame_width, pin_factory)
  
  def set_angle(self, angle: int) -> None:
    self.angle = int(angle)

  def reset(self) -> None:
    self.set_angle(0)

  