import numpy as np
from numpy.typing import NDArray
import sys
import os

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

floatMatrix = NDArray[np.float64]

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        """
        Initialize the PID controller with gains.
        @param kp: Proportional gain.
        @param ki: Integral gain.
        @param kd: Derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """
        Compute the PID control output.
        @param setpoint: Desired value.
        @param measurement: Current value.
        @param dt: Time step.
        @return: Control output.
        """
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        return output
    
    def reset(self) -> None:
        """
        Reset the PID controller state.
        """
        self.previous_error = 0.0
        self.integral = 0.0
    
    def giveForces(self) -> floatMatrix:
        return np.array([[0.0], [0.0], [0.0]], dtype=np.float64)  # Placeholder for forces, to be implemented
    
    def giveMoments(self) -> floatMatrix:
        return np.array([[0.0], [0.0], [0.0]], dtype=np.float64)  # Placeholder for moments, to be implemented