"""
It should provide actions in a same manner that RL SAC agent would do. 

Action space:

x force
y force
z force

"""

import numpy as np

import os
import sys

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.math import floatMatrix


class PIDAction:
    def updateSetpoint(self, setpoint: floatMatrix):
        self.setpoint = setpoint

    def __init__(self, kp: float, ki: float, kd: float, setpoint: floatMatrix):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = np.zeros_like(setpoint)
        self.integral = np.zeros_like(setpoint)

    def getAction(self, current_value: floatMatrix, dt: float) -> floatMatrix:
        error = self.setpoint - current_value
        derivative = (error - self.previous_error) / dt
        self.integral += error * dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.previous_error = error

        return output
        

if __name__ == "__main__":
    # Simple test of the PIDAction class
    pid = PIDAction(kp=1.0, ki=0.1, kd=0.05, setpoint=np.array([[10.0, 10.0]]))

    current_value = np.array([[0.0, 0.0]])
    dt = 0.1

    for _ in range(100):
        action = pid.getAction(current_value, dt)
        print(f"Action: {action}")
        current_value += action * dt  # Simulate system response