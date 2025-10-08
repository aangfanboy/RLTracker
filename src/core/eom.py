import numpy as np
from numpy.typing import NDArray
import sys
import os

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import utils.math as math_utils

floatMatrix = NDArray[np.float64]

class EOM:
    def __init__(self, mass: float, inertiaMatrix: floatMatrix, g: float = 9.81):
        """
        Initialize the equations of motion with mass and inertia matrix.
        """
        self.mass = mass
        self.inertiaMatrix = inertiaMatrix
        self.g = g

    def compute_translational_acceleration(self, X: float, Y: float, Z: float, A_BI: floatMatrix, u: float, v: float, w: float, p: float, q: float, r: float) -> floatMatrix:
        """
        Compute the translational acceleration based on the position.
        @param X: aero-propulsive force in the body X direction.
        @param Y: aero-propulsive force in the body Y direction.
        @param Z: aero-propulsive force in the body Z direction.
        @return: Translational acceleration as a floatMatrix of shape (3,1).
        """
        uDot: float = (1 / self.mass) * (X + self.mass* self.g * A_BI[0, 2]) - (q*w - r*v)
        vDot: float = (1 / self.mass) * (Y + self.mass* self.g * A_BI[1, 2]) - (r*u - p*w)
        wDot: float = (1 / self.mass) * (Z + self.mass* self.g * A_BI[2, 2]) - (p*v - q*u)

        return np.array([[uDot], [vDot], [wDot]], dtype=np.float64)
    
    def compute_translational_velocity(self, u: float, v: float, w: float) -> floatMatrix:
        """
        Compute the translational velocity based on the forces.
        @param u: velocity in the body X direction.
        @param v: velocity in the body Y direction.
        @param w: velocity in the body Z direction.
        """
        x_dot = u
        y_dot = v
        z_dot = w

        return np.array([[x_dot], [y_dot], [z_dot]], dtype=np.float64)
    
    def compute_angular_acceleration(self, L: float, M: float, N: float, p: float, q: float, r: float) -> floatMatrix:
        """
        Compute the angular acceleration based on the angular velocities.
        @param p: Angular velocity in the body X direction.
        @param q: Angular velocity in the body Y direction.
        @param r: Angular velocity in the body Z direction.
        @return: Angular acceleration as a floatMatrix of shape (3,1).
        """
        pDot = (1 / (self.inertiaMatrix[0, 0]*self.inertiaMatrix[2, 2] - self.inertiaMatrix[0, 2]*self.inertiaMatrix[0, 2])) * (self.inertiaMatrix[2, 2]*L - self.inertiaMatrix[0, 2]*N)
        qDot = (1 / (self.inertiaMatrix[1, 1]))*M
        rDot = (1 / (self.inertiaMatrix[0, 0]*self.inertiaMatrix[2, 2] - self.inertiaMatrix[0, 2]*self.inertiaMatrix[0, 2])) * (self.inertiaMatrix[0, 0]*N - self.inertiaMatrix[0, 2]*L)
 
        return np.array([[pDot], [qDot], [rDot]], dtype=np.float64)
    
    def compute_quaternion_rates(self, quaternion: floatMatrix, p: float, q: float, r: float) -> floatMatrix:
        """
        Compute the quaternion rates based on the angular velocities.
        @param quaternion: Current quaternion as a floatMatrix of shape (4,1).
        @param p: Angular velocity in the body X direction.
        @param q: Angular velocity in the body Y direction.
        @param r: Angular velocity in the body Z direction.
        @return: Quaternion rates as a floatMatrix of shape (4,1).
        """
        q1 = quaternion[0, 0]
        q2 = quaternion[1, 0]
        q3 = quaternion[2, 0]
        q4 = quaternion[3, 0]

        q1Dot = 0.5 * (p*q4 + r*q2 - q*q3)
        q2Dot = 0.5 * (q*q4 + p*q3 - r*q1)
        q3Dot = 0.5 * (r*q4 - p*q2 + q*q1)
        q4Dot = 0.5 * (-q1*p - q2*q - q3*r)

        return np.array([[q1Dot], [q2Dot], [q3Dot], [q4Dot]], dtype=np.float64)
    
    def rk4_step(self, X: float, Y: float, Z: float, L: float, M: float, N: float, position: floatMatrix, quaternion: floatMatrix, translationalVelocity: floatMatrix, angularVelocity: floatMatrix, dt: float) -> tuple[floatMatrix, floatMatrix, floatMatrix, floatMatrix]:
        """
        Perform a single Runge-Kutta 4th order step for the equations of motion.
        @param X: Aero-propulsive force in the body X direction.
        @param Y: Aero-propulsive force in the body Y direction.
        @param Z: Aero-propulsive force in the body Z direction.
        @param L: Aero-propulsive angular momentum in the body X direction.
        @param M: Aero-propulsive angular momentum in the body Y direction.
        @param N: Aero-propulsive angular momentum in the body Z direction.
        @param position: Current position as a floatMatrix of shape (3,1).
        @param quaternion: Current quaternion as a floatMatrix of shape (4,1).
        @param translationalVelocity: Current translational velocity as a floatMatrix of shape (3,1).
        @param angularVelocity: Current angular velocity as a floatMatrix of shape (3,1).
        @param dt: Time step for the RK4 integration.
        @return: Updated position, quaternion, velocity, and angular velocity as floatMatrices.
        """

        A_bi: floatMatrix = math_utils.calculateAttitudeMatrixFromQuaternion(quaternion)

        k1Translational = self.compute_translational_acceleration(X, Y, Z, A_bi, translationalVelocity[0, 0], translationalVelocity[1, 0], translationalVelocity[2, 0], angularVelocity[0, 0], angularVelocity[1, 0], angularVelocity[2, 0])
        k1Velocity = self.compute_translational_velocity(translationalVelocity[0, 0], translationalVelocity[1, 0], translationalVelocity[2, 0])
        k1Angular = self.compute_angular_acceleration(L, M, N, angularVelocity[0, 0], angularVelocity[1, 0], angularVelocity[2, 0])
        k1Quaternion = self.compute_quaternion_rates(quaternion, angularVelocity[0, 0], angularVelocity[1, 0], angularVelocity[2, 0])
        
        translationalVelocity_p = translationalVelocity + dt * k1Translational * 0.5
        # position_p = position + dt * k1Velocity * 0.5
        angularVelocity_p = angularVelocity + dt * k1Angular * 0.5
        quaternion_p = quaternion + dt * k1Quaternion * 0.5
        A_bi_p = math_utils.calculateAttitudeMatrixFromQuaternion(quaternion_p)

        k2Translational = self.compute_translational_acceleration(X, Y, Z, A_bi_p, translationalVelocity_p[0, 0], translationalVelocity_p[1, 0], translationalVelocity_p[2, 0], angularVelocity_p[0, 0], angularVelocity_p[1, 0], angularVelocity_p[2, 0])
        k2Velocity = self.compute_translational_velocity(translationalVelocity_p[0, 0], translationalVelocity_p[1, 0], translationalVelocity_p[2, 0])
        k2Angular = self.compute_angular_acceleration(L, M, N, angularVelocity_p[0, 0], angularVelocity_p[1, 0], angularVelocity_p[2, 0])
        k2Quaternion = self.compute_quaternion_rates(quaternion_p, angularVelocity_p[0, 0], angularVelocity_p[1, 0], angularVelocity_p[2, 0])

        translationalVelocity_pp = translationalVelocity + dt * k2Translational
        # position_pp = position + dt * k2Velocity * 0.5
        angularVelocity_pp = angularVelocity + dt * k2Angular
        quaternion_pp = quaternion + dt * k2Quaternion * 0.5
        A_bi_pp = math_utils.calculateAttitudeMatrixFromQuaternion(quaternion_pp)

        k3Translational = self.compute_translational_acceleration(X, Y, Z, A_bi_pp, translationalVelocity_pp[0, 0], translationalVelocity_pp[1, 0], translationalVelocity_pp[2, 0], angularVelocity_pp[0, 0], angularVelocity_pp[1, 0], angularVelocity_pp[2, 0])
        k3Velocity = self.compute_translational_velocity(translationalVelocity_pp[0, 0], translationalVelocity_pp[1, 0], translationalVelocity_pp[2, 0])
        k3Angular = self.compute_angular_acceleration(L, M, N, angularVelocity_pp[0, 0], angularVelocity_pp[1, 0], angularVelocity_pp[2, 0])
        k3Quaternion = self.compute_quaternion_rates(quaternion_pp, angularVelocity_pp[0, 0], angularVelocity_pp[1, 0], angularVelocity_pp[2, 0])

        translationalVelocity_ppp = translationalVelocity + dt * k3Translational
        # position_ppp = position + dt * k3Velocity * 0.5
        angularVelocity_ppp = angularVelocity + dt * k3Angular
        quaternion_ppp = quaternion + dt * k3Quaternion
        A_bi_ppp = math_utils.calculateAttitudeMatrixFromQuaternion(quaternion_ppp)

        k4Translational = self.compute_translational_acceleration(X, Y, Z, A_bi_ppp, translationalVelocity_ppp[0, 0], translationalVelocity_ppp[1, 0], translationalVelocity_ppp[2, 0], angularVelocity_ppp[0, 0], angularVelocity_ppp[1, 0], angularVelocity_ppp[2, 0])
        k4Velocity = self.compute_translational_velocity(translationalVelocity_ppp[0, 0], translationalVelocity_ppp[1, 0], translationalVelocity_ppp[2, 0])
        k4Angular = self.compute_angular_acceleration(L, M, N, angularVelocity_ppp[0, 0], angularVelocity_ppp[1, 0], angularVelocity_ppp[2, 0])
        k4Quaternion = self.compute_quaternion_rates(quaternion_ppp, angularVelocity_ppp[0, 0], angularVelocity_ppp[1, 0], angularVelocity_ppp[2, 0])

        position = position + dt * (k1Velocity + 2*k2Velocity + 2*k3Velocity + k4Velocity) / 6
        quaternion = quaternion + dt * (k1Quaternion + 2*k2Quaternion + 2*k3Quaternion + k4Quaternion) / 6
        translationalVelocity = translationalVelocity + dt * (k1Translational + 2*k2Translational + 2*k3Translational + k4Translational) / 6
        angularVelocity = angularVelocity + dt * (k1Angular + 2*k2Angular + 2*k3Angular + k4Angular) / 6
        
        return (position, quaternion, translationalVelocity, angularVelocity)

if __name__ == "__main__":
    # Example usage
    eom = EOM(mass=1000.0, inertiaMatrix=np.eye(3)*1000.0)
    quaternion = np.array([[0.0], [-0.707106781186547], [0.0], [0.707106781186547]], dtype=np.float64)
    u, v, w, p, q, r = 100.0, 0.0, 0.0, 0, 0, 0
    dt = 1
    result = eom.rk4_step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                          np.array([[0.0], [0.0], [0.0]], dtype=np.float64), 
                          quaternion, 
                          np.array([[u], [v], [w]], dtype=np.float64), 
                          np.array([[p], [q], [r]], dtype=np.float64), dt)
    print("RK4 step result:", result)