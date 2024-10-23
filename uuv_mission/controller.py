import numpy as np

class Controller:
    def __init__(self, Kp: float = 0.05, Kd: float = 0.8):
        self.Kp = Kp
        self.Kd = Kd
        self.previous_error = 0

    def compute(self, reference: float, measurement: float) -> float:
        error = reference - measurement
        delta_error = error - self.previous_error
        
        control_action = self.Kp * error + self.Kd * delta_error
        
        self.previous_error = error
        
        return control_action
