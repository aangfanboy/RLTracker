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
x-velocity
y-velocity
target-x-position
target-y-position

action-space definition:

0:  X
1:  Y

"""

def getRandomInitCoordinates(minValue: float, maxValue: float) -> floatMatrix:
    x = np.random.uniform(minValue, maxValue)
    y = np.random.uniform(minValue, maxValue)
    z = 200
    return np.array([[x], [y], [z]], dtype=np.float64)

def getRandomInitVelocity(minValue: float, maxValue: float, zVel: float = 30.0) -> floatMatrix:
    vx = np.random.uniform(minValue, maxValue)
    vy = np.random.uniform(minValue, maxValue)
    vz = 0  # Fixed initial upward velocity
    return np.array([[vx], [vy], [vz]], dtype=np.float64)

def createInitValues():
    missileInitPosition: floatMatrix = getRandomInitCoordinates(100.0, 400.0)
    missileInitQuaternions: floatMatrix = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64)  # Identity quaternion
    missileInitVelocity: floatMatrix = getRandomInitVelocity(0.0001, 0.0001)
    missileInitAngularVelocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

    targetInitPosition: floatMatrix = getRandomInitCoordinates(250.0, 250.0)
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

    enviroment.point_coordinate = missile.position.flatten()

def giveReward(missilePosition: floatMatrix, targetPosition: floatMatrix, missileVelocity: floatMatrix, action: floatMatrix, collision: bool, outOfArea: bool, targetHit: bool, counterExceed: bool) -> float:
    distance: float = np.linalg.norm(missilePosition - targetPosition) / enviroment.map.shape[0]
    accelerationMagnitude: float = np.linalg.norm(action) / 10000.0  # Normalize action to [0, 1]

    reward: float = -abs(distance) * 10  # Reward closer distance and higher speed

    if np.sign(missileVelocity[0,0]) != np.sign(targetPosition[0,0] - missilePosition[0,0]):
        reward -= abs(missileVelocity[0,0]) * 5  # Penalty for moving away from target

    if np.sign(action[0,0]) != np.sign(targetPosition[0,0] - missilePosition[0,0]):
        reward -= abs(action[0,0]) / 1000.0  # Penalty for wrong acceleration direction

    if np.sign(missileVelocity[1,0]) != np.sign(targetPosition[1,0] - missilePosition[1,0]):
        reward -= abs(missileVelocity[1,0]) * 5 # Penalty for moving away from target

    if np.sign(action[1,0]) != np.sign(targetPosition[1,0] - missilePosition[1,0]):
        reward -= abs(action[1,0]) / 1000.0  # Penalty for wrong acceleration direction

    if collision:
        reward -= 200.0  # Penalty for avoiding collision
    if outOfArea:
        reward -= 200.0  # Penalty for going out of bounds
    if targetHit:
        reward += 600.0  # Reward for hitting the target
    if counterExceed:
        reward -= 200.0  # Penalty for exceeding max steps

    reward -= accelerationMagnitude * 1000.0  # Penalty for high acceleration

    return reward / 100

def mainLoop(stepCounter: int = 0):
    dt = 0.4  # Time step for simulation

    initTime: float = time.time()

    translationalForceCommand: floatMatrix = np.zeros((3,1), dtype=np.float64)
    angularMomentCommand: floatMatrix = np.zeros((3,1), dtype=np.float64)

    done = False

    totalReward: float = 0.0
    internalCounter: int = 0
    positionBefore: floatMatrix = missile.position.copy()
    critic_loss: float = 0.0
    actor_loss: float = 0.0
    temperature_loss: float = 0.0

    total_critic_loss: float = 0.0
    total_actor_loss: float = 0.0
    total_temperature_loss: float = 0.0

    while not done:
        state: floatMatrix = np.concatenate((
            missile.position.flatten()[0:2].reshape(-1) / enviroment.map.shape[0],  # x,y position
            missile.velocity.flatten()[0:2].reshape(-1) / 100,  # x,y velocity
        )).reshape(-1)

        action = rlAgent.choose_action(state).numpy() * 10000.0  # Scale action to force range
        translationalForceCommand = np.array([[action[0,0]], [action[0,1]], [0.0]], dtype=np.float64)

        # Log flight data
        """
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
        """
        missile.update(translationalForceCommand, angularMomentCommand, dt)

        nextState: floatMatrix = np.concatenate((
            missile.position.flatten()[0:2].reshape(-1) / enviroment.map.shape[0],  # x,y position
            missile.velocity.flatten()[0:2].reshape(-1) / 100,  # x,y velocity
        )).reshape(-1)

        #collision: bool = enviroment.check_collision_with_terrain(missile.position)
        outOfArea: bool = not enviroment.check_in_bounds(missile.position)
        targetHit: bool = enviroment.check_collision_with_target(30.0, positionBefore)
        counterExceed: bool = internalCounter > 250

        if targetHit:
            flightLogger.writeInfo("Target hit, exiting round")
            done = True

        if outOfArea:
            flightLogger.writeInfo("Missile out of bounds, exiting round")
            done = True

        if counterExceed:
            flightLogger.writeInfo("Max steps reached, exiting round")
            done = True

        reward: float = giveReward(missile.position, target.position, missile.velocity, translationalForceCommand, False, outOfArea, targetHit, counterExceed)
        totalReward += reward

        #if collision:
        #    flightLogger.writeInfo("Collision with terrain, exiting round")
        #    done = True

        rlAgent.addToReplayBuffer(state, action, reward, nextState, done)
        
        # Train the agent and get losses, note that it may give None if not enough samples
        trainResults = rlAgent.train()
        if trainResults is not None:
            critic_loss1, critic_loss2, actor_loss, temperature_loss = trainResults
            critic_loss1 = critic_loss1.numpy()
            critic_loss2 = critic_loss2.numpy()
            critic_loss = critic_loss1 + critic_loss2
            actor_loss = actor_loss.numpy()
            temperature_loss = temperature_loss.numpy()

            total_critic_loss += critic_loss1 + critic_loss2
            total_actor_loss += actor_loss
            total_temperature_loss += temperature_loss
            """
            flightLogger.lossLog(
                timeFloat=logTime,
                criticLoss1=critic_loss1,
                criticLoss2=critic_loss2,
                actorLoss=actor_loss,
                temperatureLoss=temperature_loss
            )
            """

        if stepCounter % 10 == 0 or done:
            print(f"[{stepCounter}][{internalCounter}]State: {state}, Action: {action}, Reward: {reward:.2f}, Critic Loss: {critic_loss:.2f}, Actor Loss: {actor_loss:.2f}, Temp Loss: {temperature_loss:.2f}")

        stepCounter += 1
        internalCounter += 1
        positionBefore = missile.position.copy()

    averageReward: float = totalReward / internalCounter if internalCounter > 0 else 0.0
    averageCriticLoss: float = total_critic_loss / internalCounter if internalCounter > 0 else 0.0
    averageActorLoss: float = total_actor_loss / internalCounter if internalCounter > 0 else 0.0
    averageTemperatureLoss: float = total_temperature_loss / internalCounter if internalCounter > 0 else 0.0

    print(f"Round finished with total reward: {totalReward:.2f}, average reward: {averageReward:.2f}, average critic loss: {averageCriticLoss:.4f}, average actor loss: {averageActorLoss:.4f}, average temperature loss: {averageTemperatureLoss:.4f}")
    flightLogger.addReward(runCounter, reward=totalReward, averageReward=averageReward, averageCriticLoss=averageCriticLoss, averageActorLoss=averageActorLoss, averageTemperatureLoss=averageTemperatureLoss)
    return stepCounter

if __name__ == "__main__":
    runCounter: int = 0
    stepCounter: int = 0
    numEpisodes: int = 500000
    trainingName: str = "SACTraining500000RHA"

    (missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity,
     targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity) = createInitValues()
    
    eomMissile = EOM(config.MissileConfig.MASS, config.MissileConfig.INERTIA_MATRIX, g=0.0) # For simplicity, set g=0.0
    eomTarget = EOM(config.TargetConfig.MASS, config.TargetConfig.INERTIA_MATRIX, g=0.0) # Static target, g=0.0

    missile = Missile(missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity, config.MissileConfig.Constraints(), eomMissile)
    target = Target(targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity, config.TargetConfig.Constraints(), eomTarget)

    enviroment = Enviroment(750, 750, numMountains=0, maxHeight=190, missile=missile, target=target)

    flightLogger = FlightLogger(unique_name=trainingName, run_id=runCounter, map=enviroment.map)

    rlAgent = Agent(trainingName=trainingName, stateDims=4, nActions=2, gamma=0.98, learningRate=0.0003, tau=0.005, batchSize=256, minBufferSize=1000)

    while runCounter < numEpisodes:
        try:
            stepCounter = mainLoop(stepCounter)
        except KeyboardInterrupt:
            print(f"Simulation interrupted by user after {runCounter} runs.")
            break

        print(f"Completed runs: {runCounter}, resetting simulation.")
        runCounter += 1

        if runCounter % 100 == 0:
            rlAgent.saveModels(f"flightData/{trainingName}/models")
            print("Models saved.")

        resetTargetAndMissile()
        # flightLogger.reset(unique_name=trainingName, run_id=runCounter, map=enviroment.map)
        # enviroment.resetMap()

    flightLogger.deleteLastFlightLog()
    flightLogger.exit()

    print("Simulation finished.")
