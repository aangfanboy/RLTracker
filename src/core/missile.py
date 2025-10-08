import numpy as np
from numpy.typing import NDArray
import sys
import os

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from config import MissileConfig, ACConfig
import core.eom as eom

floatMatrix = NDArray[np.float64]

"""
@brief Missile class for simulating missile behavior in the simulation.
@file core/missile.py

@details This class handles the initialization of a missile's position and velocity.

@param initPosition: Initial position of the missile as a floatMatrix of shape (3,1) wrt origin in ECI frame.
@param initQuaternions: Initial quaternions of the missile as a floatMatrix of shape (4,1), representing the orientation.
@param initVelocity: Initial velocity of the missile as a floatMatrix of shape (3,1).
@param initAngularVelocity: Initial angular velocity of the missile as a floatMatrix of shape (3,1).
@param mass: Mass of the missile as a float.
@param inertiaMatrix: Inertia matrix of the missile as a floatMatrix of shape (3,3).

"""

class Missile:
    def __init__(self, initPosition: floatMatrix, initQuaternions: floatMatrix, initVelocity: floatMatrix, initAngularVelocity: floatMatrix, constraintsConfig: MissileConfig.Constraints | ACConfig.Constraints, eomClass: eom.EOM):
        self.position: floatMatrix = initPosition # Initial position of the missile
        self.quaternions: floatMatrix = initQuaternions # Initial quaternions of the missile (orientation)
        self.velocity: floatMatrix = initVelocity # Initial velocity of the missile
        self.angularVelocity: floatMatrix = initAngularVelocity # Initial angular velocity of the missile

        self.mass: float = eomClass.mass
        self.inertiaMatrix: floatMatrix = eomClass.inertiaMatrix

        self.constraints = constraintsConfig
        self.eom = eomClass  # Initialize the equations of motion class with mass and inertia matrix

    def reset(self, initPosition: floatMatrix, initQuaternions: floatMatrix, initVelocity: floatMatrix, initAngularVelocity: floatMatrix) -> None:
        """
        Reset the missile's position, orientation, velocity, and angular velocity.
        @param initPosition: Initial position of the missile as a floatMatrix of shape (3,1).
        @param initQuaternions: Initial quaternions of the missile as a floatMatrix of shape (4,1).
        @param initVelocity: Initial velocity of the missile as a floatMatrix of shape (3,1).
        @param initAngularVelocity: Initial angular velocity of the missile as a floatMatrix of shape (3,1).
        """
        self.position = initPosition
        self.quaternions = initQuaternions
        self.velocity = initVelocity
        self.angularVelocity = initAngularVelocity

    def update(self, translationalForceCommand: floatMatrix, angularMomentCommand: floatMatrix, dt: float) -> None:
        """
        Update the missile's position, velocity, and orientation based on the translational and angular accelerations.
        @param translationalForceCommand: Translational force command as a floatMatrix of shape (3,1).
        @param angularMomentCommand: Angular moment command as a floatMatrix of shape (3,
        @param dt: Time step for the update.
        """
        
        X_command: float = translationalForceCommand[0, 0]
        Y_command: float = translationalForceCommand[1, 0]
        Z_command: float = translationalForceCommand[2, 0]
        L_command: float = angularMomentCommand[0, 0]
        M_command: float = angularMomentCommand[1, 0]
        N_command: float = angularMomentCommand[2, 0]

        X_command = np.clip(X_command, -self.constraints.MAX_X, self.constraints.MAX_X)
        Y_command = np.clip(Y_command, -self.constraints.MAX_Y, self.constraints.MAX_Y)
        Z_command = np.clip(Z_command, -self.constraints.MAX_Z, self.constraints.MAX_Z)
        L_command = np.clip(L_command, -self.constraints.MAX_P, self.constraints.MAX_P)
        M_command = np.clip(M_command, -self.constraints.MAX_Q, self.constraints.MAX_Q)
        N_command = np.clip(N_command, -self.constraints.MAX_R, self.constraints.MAX_R)

        # Perform a single Runge-Kutta 4th order step for the equations of motion
        [self.position, self.quaternions, self.velocity, self.angularVelocity] = self.eom.rk4_step(
            X_command, Y_command, Z_command,
            L_command, M_command, N_command,
            self.position, self.quaternions,
            self.velocity, self.angularVelocity, dt
        )

if __name__ == "__main__":
    # Example usage
    missile = Missile(
        initPosition=np.array([[0.0], [0.0], [0.0]], dtype=np.float64),
        initQuaternions=np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64),
        initVelocity=np.array([[100.0], [0.0], [0.0]], dtype=np.float64),
        initAngularVelocity=np.array([[0.1], [0.1], [0.1]], dtype=np.float64),
        constraintsConfig=MissileConfig.Constraints(),
        eomClass=eom.EOM(MissileConfig.MASS, MissileConfig.INERTIA_MATRIX)
    )

    print(f"Missile initialized with position: {missile.position.flatten()}, velocity: {missile.velocity.flatten()}, quaternions: {missile.quaternions.flatten()}")