import math
import config

from core.missile import Missile
from core.eom import EOM
from core.target import Target

from guidance.pid import PIDController

from visualize.orientationVis import OrientationVisualizer
from visualize.pointMapVis import PointMapVisualizer

import numpy as np
import math
from numpy.typing import NDArray

floatMatrix = NDArray[np.float64]

def resetClasses():
    missile.reset(missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity)
    target.reset(targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity)

    pidController.reset()

    orientationVisualizer.update_quaternion(missile.quaternions)

def mainLoop():
    dt = 0.02  # Time step for simulation

    while True:
        translationalForceCommand = pidController.giveForces()
        angularMomentCommand = pidController.giveMoments()

        missile.update(translationalForceCommand, angularMomentCommand, dt)

        # Update the target's position and orientation if needed
        # For now, we will keep the target static

        # Update the visualizers with the current quaternion
        orientationVisualizer.update_quaternion(missile.quaternions)
        pointMapVisualizer.update_points(np.array([missile.position.flatten(), target.position.flatten()]))

        # Sleep to simulate real-time updates
if __name__ == "__main__":
    missileInitPosition: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    missileInitQuaternions: floatMatrix = np.array([[0.0], [math.cos(math.pi/4)], [0.0], [-math.cos(math.pi/4)]], dtype=np.float64)  # Identity quaternion
    missileInitVelocity: floatMatrix = np.array([[1000.0], [0.0], [0.0]], dtype=np.float64)
    missileInitAngularVelocity: floatMatrix = np.array([[0.0], [0.05], [0.0]], dtype=np.float64)

    targetInitPosition: floatMatrix = np.array([[100.0], [100.0], [100.0]], dtype=np.float64)
    targetInitQuaternions: floatMatrix = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64)  # Identity quaternion
    targetInitVelocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    targetInitAngularVelocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    
    eomMissile = EOM(config.MissileConfig.MASS, config.MissileConfig.INERTIA_MATRIX)
    eomTarget = EOM(config.TargetConfig.MASS, config.TargetConfig.INERTIA_MATRIX, g=0.0)

    missile = Missile(missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity, config.MissileConfig.Constraints(), eomMissile)
    target = Target(targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity, config.TargetConfig.Constraints(), eomTarget)

    pidController = PIDController(kp=1.0, ki=0.1, kd=0.01)
    orientationVisualizer = OrientationVisualizer("visualize\\plane.obj", missile.quaternions)
    pointMapVisualizer = PointMapVisualizer(np.array([missile.position.flatten(), target.position.flatten()]))

    orientationVisualizer.start_thread()
    pointMapVisualizer.start()

    try:
        mainLoop()
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        orientationVisualizer.stop()
        pointMapVisualizer.close()
