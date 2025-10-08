import numpy as np
from numpy.typing import NDArray

floatMatrix = NDArray[np.float64]

class ACConfig:
    MASS: float = 5000.0  # @unit kg, @type float
    INERTIA_MATRIX: floatMatrix = np.array([[10000.0, 0.0, 0.0],
                                             [0.0, 10000.0, 0.0],
                                             [0.0, 0.0, 10000.0]])  # @unit kg*m^2, @type floatMatrix
    class Constraints:
        MAX_X: float = 20000.0  # @unit N, @type float
        MAX_Y: float = 500.0  # @unit N, @type float
        MAX_Z: float = 500.0  # @unit N, @type float
        MAX_P: float = 1.5  # @unit rad/s, @type float
        MAX_Q: float = 1.5  # @unit rad/s, @type float
        MAX_R: float = 1.5  # @unit rad/s, @type float

class MissileConfig(ACConfig):
    MASS: float = 1000.0  # @unit kg, @type float
    INERTIA_MATRIX: floatMatrix = np.array([[500.0, 0.0, 0.0],
                                             [0.0, 500.0, 0.0],
                                             [0.0, 0.0, 500.0]])  # @unit kg*m^2, @type floatMatrix
    class Constraints(ACConfig.Constraints):
        MAX_X: float = 10000.0  # @unit N, @type float
        MAX_Y: float = 200.0  # @unit N, @type float
        MAX_Z: float = 200.0  # @unit N, @type float
        MAX_P: float = 0.5 # @unit rad/s, @type float
        MAX_Q: float = 0.5 # @unit rad/s, @type float
        MAX_R: float = 0.5 # @unit rad/s, @type float

class TargetConfig(ACConfig):
    MASS: float = 2000.0  # @unit kg, @type float
    INERTIA_MATRIX: floatMatrix = np.array([[300.0, 0.0, 0.0],
                                             [0.0, 300.0, 0.0],
                                             [0.0, 0.0, 300.0]])  # @unit kg*m^2, @type floatMatrix
    class Constraints(ACConfig.Constraints):
        MAX_X: float = 5000.0  # @unit N, @type float
        MAX_Y: float = 100.0  # @unit N, @type float
        MAX_Z: float = 100.0  # @unit N, @type float
        MAX_P: float = 1.0 # @unit rad/s, @type float
        MAX_Q: float = 1.0 # @unit rad/s, @type float
        MAX_R: float = 1.0 # @unit rad/s, @type float