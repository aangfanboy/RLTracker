import math
import config

from core.missile import Missile
from core.eom import EOM
from core.target import Target

from envGame.main import Enviroment
from flightData.flightLogger import FlightLogger

import numpy as np
import time
from numpy.typing import NDArray

from guidance.QNetwork import Agent

floatMatrix = NDArray[np.float64]

"""
state-space definition:

x-position
y-position
z-position
x-velocity
y-velocity
z-velocity

action-space definition:

0:  X
1:  Y
2:  Z


"""

def getRandomInitCoordinates(minValue: float, maxValue: float) -> floatMatrix:
    x = np.random.uniform(minValue, maxValue)
    y = 200
    z = 200
    return np.array([[x], [y], [z]], dtype=np.float64)

def getRandomInitVelocity(minValue: float, maxValue: float, zVel: float = 30.0) -> floatMatrix:
    vx = np.random.uniform(minValue, maxValue)
    vy = 0
    vz = 0  # Fixed initial upward velocity
    return np.array([[vx], [vy], [vz]], dtype=np.float64)

def createInitValues():
    missileInitPosition: floatMatrix = getRandomInitCoordinates(100.0, 400.0)
    missileInitQuaternions: floatMatrix = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64)  # Identity quaternion
    missileInitVelocity: floatMatrix = getRandomInitVelocity(0.01, 0.01)
    missileInitAngularVelocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

    targetInitPosition: floatMatrix = getRandomInitCoordinates(100.0, 400.0)
    targetInitQuaternions: floatMatrix = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64)  # Identity quaternion
    targetInitVelocity: floatMatrix = getRandomInitVelocity(0.0, 0.0, zVel=0.0)
    targetInitAngularVelocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

    return (missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity,
            targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity)

def resetTargetAndMissile():
    (missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity,
     targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity) = createInitValues()

    missile.reset(missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity)
    target.reset(targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity)

    enviroment.quaternion_queue.put(missile.quaternions)
    enviroment.point_coordinate = missile.position.flatten()

def mainLoop(stepCounter: int = 0):
    dt = 0.5  # Time step for simulation

    initTime: float = time.time()

    translationalForceCommand: floatMatrix = np.zeros((3,1), dtype=np.float64)
    angularMomentCommand: floatMatrix = np.zeros((3,1), dtype=np.float64)

    done = False

    totalReward: float = 0.0
    internalCounter: int = 0

    while not done:
        state: floatMatrix = np.concatenate((
            missile.position.flatten()[0].reshape(-1) / enviroment.map.shape[0],  # x position
            missile.velocity.flatten()[0].reshape(-1) / 100,  # x velocity
            (missile.position.flatten()[0].reshape(-1) - target.position.flatten()[0].reshape(-1)) / enviroment.map.shape[0],  # target x position
        )).reshape(-1)

        action = dqnAgent.choose_action(state) * 10000.0  # Scale action to force range
        translationalForceCommand = np.array([[action[0]], [0.0], [0.0]], dtype=np.float64)

        # Log flight data
        logTime: float = time.time() - initTime
        flightLogger.flightLog(
            timeFloat=logTime,
            position=missile.position.flatten(),    
            velocity=missile.velocity.flatten(),
            orientation=missile.quaternions.flatten(),
            angular_velocity=missile.angularVelocity.flatten(),
            target_position=target.position.flatten(),
            target_velocity=target.velocity.flatten()
        )

        flightLogger.commandLog(
            timeFloat=logTime,
            X=translationalForceCommand[0, 0],
            Y=translationalForceCommand[1, 0],
            Z=translationalForceCommand[2, 0],
            L=angularMomentCommand[0, 0],
            M=angularMomentCommand[1, 0],
            N=angularMomentCommand[2, 0],
        )

        missile.update(translationalForceCommand, angularMomentCommand, dt)

        nextState: floatMatrix = np.concatenate((
            missile.position.flatten()[0].reshape(-1) / enviroment.map.shape[0],  # x position
            missile.velocity.flatten()[0].reshape(-1) / 100,  # x velocity
            (missile.position.flatten()[0].reshape(-1) - target.position.flatten()[0].reshape(-1)) / enviroment.map.shape[0],  # target x position
        )).reshape(-1)

        print("State:", state)

        reward: float = -abs(missile.position[0,0] - target.position[0,0]) / enviroment.map.shape[0]  # Negative distance reward

        if enviroment.check_collision_with_target(10.0):
            flightLogger.writeInfo("Target hit, exiting round")
            reward += 10
            done = True

        if not enviroment.check_in_bounds(missile.position):
            flightLogger.writeInfo("Missile out of bounds, exiting round")
            reward -= 10
            done = True

        if internalCounter > 150:
            flightLogger.writeInfo("Max steps reached, exiting round")
            done = True

        """
        if enviroment.check_collision_with_terrain(missile.position):
            flightLogger.writeInfo("Missile collided, exiting round")
            reward -= 1000.0
            done = True
        """ # Terrain collision disabled for simplicity

        dqnAgent.remember(state, action, reward, nextState, done)
        dqnAgent.learn()
        stepCounter += 1
        internalCounter += 1

        totalReward += reward

    print(f"Round finished with total reward: {totalReward:.2f}")

if __name__ == "__main__":
    (missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity,
     targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity) = createInitValues()
    
    eomMissile = EOM(config.MissileConfig.MASS, config.MissileConfig.INERTIA_MATRIX, g=0.0) # For simplicity, set g=0.0
    eomTarget = EOM(config.TargetConfig.MASS, config.TargetConfig.INERTIA_MATRIX, g=0.0) # Static target, g=0.0

    missile = Missile(missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity, config.MissileConfig.Constraints(), eomMissile)
    target = Target(targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity, config.TargetConfig.Constraints(), eomTarget)

    enviroment = Enviroment(600, 600, numMountains=0, maxHeight=190, missile=missile, target=target, visualize=False)

    enviroment.run()

    runCounter: int = 0
    stepCounter: int = 0
    numEpisodes: int = 1000

    flightLogger = FlightLogger(f"{'dqnTraining1000'}_{runCounter}", enviroment.map)

    dqnAgent = Agent(input_dims=[3], n_actions=1, max_action=1)

    while runCounter < numEpisodes:
        try:
            mainLoop(stepCounter)
        except KeyboardInterrupt:
            print(f"Simulation interrupted by user after {runCounter} runs.")
            break

        runCounter += 1

        print(f"Completed runs: {runCounter}, resetting simulation.")

        resetTargetAndMissile()
        flightLogger.reset(f"flight_data_{runCounter}", enviroment.map)
        enviroment.resetMap()

    flightLogger.deleteLastFlightLog()
    flightLogger.exit()
    enviroment.stop()

    print("Simulation finished.")
