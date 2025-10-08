import numpy as np
from numpy.typing import NDArray
import sys
import os

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from config import TargetConfig, ACConfig
import core.eom as eom
import core.missile as missile

floatMatrix = NDArray[np.float64]

"""
@brief Target class for simulating target behavior in the simulation.
@file core/target.py

@details This class handles the initialization of a target's position and velocity.

@param initPosition: Initial position of the target as a floatMatrix of shape (3,1) wrt origin in ECI frame.
@param initQuaternions: Initial quaternions of the target as a floatMatrix of shape (4,1), representing the orientation.
@param initVelocity: Initial velocity of the target as a floatMatrix of shape (3,1).
@param initAngularVelocity: Initial angular velocity of the target as a floatMatrix of shape (3,1).
@param mass: Mass of the target as a float.
@param inertiaMatrix: Inertia matrix of the target as a floatMatrix of shape (3,3).

"""

class Target(missile.Missile):
    def __init__(self, initPosition: floatMatrix, initQuaternions: floatMatrix, initVelocity: floatMatrix, initAngularVelocity: floatMatrix, constraintsConfig: TargetConfig.Constraints | ACConfig.Constraints, eomClass: eom.EOM):
        """
        Initialize the target with its position, orientation, velocity, angular velocity, mass, and inertia matrix.
        """
        super().__init__(initPosition, initQuaternions, initVelocity, initAngularVelocity, constraintsConfig, eomClass)
        