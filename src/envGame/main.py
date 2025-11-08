import numpy as np

import os
import sys

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.math import floatMatrix
from core.missile import Missile
from core.target import Target

class Enviroment:
    def resetMap(self):
        self.map = self.create_3D_map(self.map.shape[0], self.map.shape[1], numMountains=self.numMountains, maxHeight=self.maxHeight)

    def __init__(self, xDim: int, yDim: int, numMountains: int = 10, maxHeight: int = 300, missile: Missile = None, target: Target = None, maxAllowableHeight: int = 1000):
        self.missile = missile
        self.target = target
        self.numMountains = numMountains
        self.maxHeight = maxHeight
        self.maxAllowableHeight = maxAllowableHeight
        self.quaternion: floatMatrix = self.missile.quaternions

        self.map: floatMatrix = self.create_3D_map(xDim, yDim, numMountains=numMountains, maxHeight=maxHeight)
        self.point_coordinate: floatMatrix = self.missile.position
        self.target_coordinate: floatMatrix = self.target.position

    def put_inside_bounds(self) -> bool:
        coordinate: floatMatrix = self.missile.position
        isInside: bool = self.check_in_bounds(coordinate)
        if isInside:
            return False

        outsideX = 1
        outsideY = 1
        outsideZ = 1
        
        x, y, z = coordinate.flatten()
        xp = np.clip(x, 1, self.map.shape[0] - 1)
        yp = np.clip(y, 1, self.map.shape[1] - 1)
        zp = np.clip(z, 1, self.maxAllowableHeight - 1)
        
        if x != xp: outsideX = -.0
        if y != yp: outsideY = -.0
        if z != zp: outsideZ = -.0

        self.missile.position = np.array([[xp], [yp], [zp]])
        self.missile.velocity = np.array([[self.missile.velocity[0, 0] * outsideX],
                                          [self.missile.velocity[1, 0] * outsideY],
                                          [self.missile.velocity[2, 0] * outsideZ]])
        
        return True
        

    def check_in_bounds(self, coordinate: floatMatrix) -> bool:
        x, y, z = coordinate.flatten()
        if 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1] and 0 <= z < self.maxAllowableHeight:
            return True
        return False
    
    def check_collision_with_terrain(self, coordinate: floatMatrix) -> bool:
        """Check if the given coordinate collides with the terrain"""
        x, y, z = coordinate.flatten()
        return z <= self.map[int(x), int(y)]

    def check_collision_with_target(self, distance_threshold: float = 2.0, positionBefore: floatMatrix = None) -> bool:
        """interpolate between previous and current position to check for collision with target"""
        steps = 20
        for i in range(steps + 1):
            interp_position = positionBefore + (self.missile.position - positionBefore) * (i / steps)
            distance = np.linalg.norm(interp_position - self.target.position)
            if distance <= distance_threshold:
                return True
        return False

    def create_3D_map(self, xDim: int, yDim: int, numMountains: int = 5, maxHeight: int = 100) -> floatMatrix:
        """Create a simple 3D map with random mountains"""
        map: floatMatrix = np.zeros((xDim, yDim))
        x = np.arange(xDim)
        y = np.arange(yDim)
        x, y = np.meshgrid(x, y)

        for _ in range(numMountains):
            # Random center
            cx = np.random.randint(0, xDim)
            cy = np.random.randint(0, yDim)
            height = np.random.randint(maxHeight // 2, maxHeight)
            width = np.random.randint(maxHeight // 10, maxHeight // 5)
            
            # Create a Gaussian mountain with O(n)
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            map += height * np.exp(-(dist ** 2) / (2 * (width ** 2)))
            # --- IGNORE ---

        return map

if __name__ == "__main__":
    env: Enviroment = Enviroment(500, 500)