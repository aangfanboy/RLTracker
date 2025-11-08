import numpy as np

import sys
import os

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.math import floatMatrix

class Enviroment:
    def __init__(self, xDim: int, yDim: int, maxHeight: float, maxVelocity: float = 1000.0):
        self.xDim = xDim
        self.yDim = yDim
        self.maxHeight = maxHeight
        self.maxVelocity = maxVelocity
        self.point_coordinate: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        self.point_velocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        self.target_coordinate: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        self.target_velocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

    def iterateRK4(self, acceleration: floatMatrix, dt: float) -> None:
        """
        Update the point's position and velocity using the RK4 method.
        @param acceleration: Acceleration of the point as a float.
        @param dt: Time step for the update as a float.
        """
        def derivatives(position: floatMatrix, velocity: floatMatrix) -> tuple[floatMatrix, floatMatrix]:
            return velocity, acceleration

        k1_v, k1_a = derivatives(self.point_coordinate, self.point_velocity)
        k2_v, k2_a = derivatives(self.point_coordinate + 0.5 * dt * k1_v, self.point_velocity + 0.5 * dt * k1_a)
        k3_v, k3_a = derivatives(self.point_coordinate + 0.5 * dt * k2_v, self.point_velocity + 0.5 * dt * k2_a)
        k4_v, k4_a = derivatives(self.point_coordinate + dt * k3_v, self.point_velocity + dt * k3_a)

        self.point_coordinate += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        self.point_velocity += (dt / 6.0) * (k1_a + 2.0 * k2_a + 2.0 * k3_a + k4_a)

        self.point_velocity = np.clip(self.point_velocity, -self.maxVelocity, self.maxVelocity)

    def put_inside_bounds(self) -> bool:
        isInside: bool = self.check_in_bounds()
        if isInside:
            return False

        self.point_coordinate[0] = np.clip(self.point_coordinate[0], 0, self.xDim)
        self.point_coordinate[1] = np.clip(self.point_coordinate[1], 0, self.yDim)
        self.point_coordinate[2] = np.clip(self.point_coordinate[2], 0, self.maxHeight)

        self.point_velocity[0] = 0.0
        self.point_velocity[1] = 0.0
        self.point_velocity[2] = 0.0

        return True
        
    def check_in_bounds(self) -> bool:
        if self.point_coordinate[0] < 0 or self.point_coordinate[0] > self.xDim:
            return False
        if self.point_coordinate[1] < 0 or self.point_coordinate[1] > self.yDim:
            return False
        if self.point_coordinate[2] < 0 or self.point_coordinate[2] > self.maxHeight:
            return False
        
        return True

    def check_collision_with_target(self, distance_threshold: float = 2.0, positionBefore: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64), steps: int = 20) -> bool:
        """interpolate between previous and current position to check for collision with target"""
        for step in range(steps + 1):
            interp_position = positionBefore + (self.point_coordinate - positionBefore) * (step / steps)
            if np.linalg.norm(interp_position - self.target_coordinate) <= distance_threshold:
                return True
            
        return False


if __name__ == "__main__":
    # Example usage
    env = Enviroment(xDim=1000, yDim=1000, maxVelocity=500.0)
    env.point_coordinate = np.array([[100.0], [100.0]], dtype=np.float64)
    env.point_velocity = np.array([[0.0], [0.0]], dtype=np.float64)

    while True:
        acceleration = np.array([[10.0], [5.0]], dtype=np.float64) # assume mass = 1 for simplicity
        print(acceleration.shape)
        env.iterateRK4(acceleration=acceleration, dt=1.0)
        print(f"Position: {env.point_coordinate.flatten()}, Velocity: {env.point_velocity.flatten()}")
        input("Press Enter to continue...")  # Pause for user input to step through the simulation
