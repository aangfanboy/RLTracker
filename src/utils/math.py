import numpy as np
from numpy.typing import NDArray

floatMatrix = NDArray[np.float64]

def quaternionMultiply(q1: floatMatrix, q2: floatMatrix) -> floatMatrix:
    """
    Multiply two quaternions.
    @param q1: First quaternion as a floatMatrix of shape (4,1).
    @param q2: Second quaternion as a floatMatrix of shape (4,1).
    @return: Resulting quaternion as a floatMatrix of shape (4,1).

    """
    
    x1, y1, z1, w1 = q1[0, 0], q1[1, 0], q1[2, 0], q1[3, 0]
    x2, y2, z2, w2 = q2[0, 0], q2[1, 0], q2[2, 0], q2[3, 0]

    x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
    y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
    z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
    w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1

    return np.array([[x], [y], [z], [w]], dtype=np.float64)

def calculateAttitudeMatrixFromQuaternion(quaternion: floatMatrix) -> floatMatrix:
    """
    Calculate the attitude matrix from quaternions.
    @param quaternion: Quaternion as a floatMatrix of shape (4,1).
    @return: Attitude matrix as a floatMatrix of shape (3,3).

    """
    
    q1, q2, q3, q4 = quaternion[0, 0], quaternion[1, 0], quaternion[2, 0], quaternion[3, 0]
    
    attitudeMatrix = np.zeros((3, 3), dtype=np.float64)
    attitudeMatrix[0, 0] = q4*q4 + q1*q1 - q2*q2 - q3*q3
    attitudeMatrix[0, 1] = 2 * (q1*q2 - q4*q3)
    attitudeMatrix[0, 2] = 2 * (q4*q2 + q1*q3)
    attitudeMatrix[1, 0] = 2 * (q4*q3 + q1*q2)
    attitudeMatrix[1, 1] = q4*q4 - q1*q1 + q2*q2 - q3*q3
    attitudeMatrix[1, 2] = 2 * (q2*q3 - q4*q1)
    attitudeMatrix[2, 0] = 2 * (q1*q3 - q4*q2)
    attitudeMatrix[2, 1] = 2 * (q4*q1 + q2*q3)
    attitudeMatrix[2, 2] = q4*q4 - q1*q1 - q2*q2 + q3*q3
    
    return attitudeMatrix

def calculateQuaternionFromAttitudeMatrix(attitudeMatrix: floatMatrix) -> floatMatrix:
    """
    Calculate the quaternion from an attitude matrix.
    @param attitudeMatrix: Attitude matrix as a floatMatrix of shape (3,3).
    @return: Quaternion as a floatMatrix of shape (4,1).

    """

    tr = attitudeMatrix[0, 0] + attitudeMatrix[1, 1] + attitudeMatrix[2, 2]
    epsilon = 1e-12  # Minimum value to treat a number as zero

    if tr > epsilon:  # If tr > 0
        q1 = 0.5 * np.sqrt(tr + 1)
        s_inv = 1 / (q1 * 4)
        q2 = (attitudeMatrix[2, 1] - attitudeMatrix[1, 2]) * s_inv
        q3 = (attitudeMatrix[0, 2] - attitudeMatrix[2, 0]) * s_inv
        q4 = (attitudeMatrix[1, 0] - attitudeMatrix[0, 1]) * s_inv
    else:  # If tr <= 0
        if (attitudeMatrix[0, 0] > attitudeMatrix[1, 1]) and (attitudeMatrix[0, 0] > attitudeMatrix[2, 2]):
            q2 = 0.5 * np.sqrt(attitudeMatrix[0, 0] - attitudeMatrix[1, 1] - attitudeMatrix[2, 2] + 1)
            s_inv = 1 / (q2 * 4)
            q1 = (attitudeMatrix[2, 1] + attitudeMatrix[1, 2]) * s_inv
            q3 = (attitudeMatrix[0, 1] + attitudeMatrix[1, 0]) * s_inv
            q4 = (attitudeMatrix[2, 0] + attitudeMatrix[0, 2]) * s_inv
        elif attitudeMatrix[1, 1] > attitudeMatrix[2, 2]:
            q3 = 0.5 * np.sqrt(attitudeMatrix[1, 1] - attitudeMatrix[2, 2] - attitudeMatrix[0, 0] + 1)
            s_inv = 1 / (q3 * 4)
            q1 = (attitudeMatrix[0, 2] - attitudeMatrix[2, 0]) * s_inv
            q2 = (attitudeMatrix[0, 1] + attitudeMatrix[1, 0]) * s_inv
            q4 = (attitudeMatrix[2, 1] + attitudeMatrix[1, 2]) * s_inv
        else:
            q4 = 0.5 * np.sqrt(attitudeMatrix[2, 2] - attitudeMatrix[0, 0] - attitudeMatrix[1, 1] + 1)
            s_inv = 1 / (q4 * 4)
            q1 = (attitudeMatrix[1, 0] - attitudeMatrix[0, 1]) * s_inv
            q2 = (attitudeMatrix[2, 0] + attitudeMatrix[0, 2]) * s_inv
            q3 = (attitudeMatrix[2, 1] + attitudeMatrix[1, 2]) * s_inv

    quaternion = np.array([[q2], [q3], [q4], [q1]], dtype=np.float64)  # Quaternion in the order (x, y, z, w)

    return quaternion / np.linalg.norm(quaternion)  # Normalize the quaternion
