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
target-x-position
target-y-position
target-z-position

action-space definition:

0:  X
1:  Y
2:  Z

"""

def getRandomInitCoordinates(minValue: float, maxValue: float, minHeight: float = 200.0, maxHeight: float = 400.0) -> floatMatrix:
    x = np.random.uniform(minValue, maxValue)
    y = np.random.uniform(minValue, maxValue)
    z = np.random.uniform(minHeight, maxHeight)
    return np.array([[x], [y], [z]], dtype=np.float64)

def getRandomInitVelocity(minValue: float, maxValue: float, zVel: float = 30.0) -> floatMatrix:
    vx = np.random.uniform(minValue, maxValue)
    vy = np.random.uniform(minValue, maxValue)
    vz = zVel  # Fixed initial upward velocity
    return np.array([[vx], [vy], [vz]], dtype=np.float64)

def createInitValues():
    missileInitPosition: floatMatrix = getRandomInitCoordinates(100.0, 900.0, minHeight=100.0, maxHeight=600.0)
    missileInitQuaternions: floatMatrix = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64)  # Identity quaternion
    missileInitVelocity: floatMatrix = getRandomInitVelocity(0.0, 0.0, zVel=0.0)
    missileInitAngularVelocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

    targetInitPosition: floatMatrix = getRandomInitCoordinates(300.0, 600.0, minHeight=200.0, maxHeight=400.0)
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

def giveReward(
    missilePosition: floatMatrix,
    targetPosition: floatMatrix,
    missileVelocity: floatMatrix,
    action: floatMatrix,
    collision: bool,
    outOfArea: bool,
    targetHit: bool,
    counterExceed: bool,
    isBounced: bool,
    outsideZ: int
) -> float:
    """
    Improved reward for 2D missile tracking (static target).

    - Uses change in distance to target (preferred) or velocity projection if prev position not available.
    - Penalizes squared action magnitude (energy).
    - Adds terminal bonuses/penalties for hit/collision/outOf-area.
    - Returns a scalar reward (tunable via constants below).
    """
    
    # --- Distance and direction ---
    direction_to_target = targetPosition - missilePosition
    distance = np.linalg.norm(direction_to_target) + 1e-6
    direction_unit = direction_to_target / distance
    distanceScaled = distance / 1000.0  # scale distance for reward calculation

    # --- Velocity alignment (cosine similarity) ---
    vel_norm = np.linalg.norm(missileVelocity) + 1e-6
    velocity_alignment = float(np.dot(missileVelocity.T, direction_unit) / vel_norm)  # [-1, +1]

    # --- Acceleration alignment (cosine similarity) ---
    acc_norm = np.linalg.norm(action) / 100.0  # normalize using your action scale
    if np.linalg.norm(action) > 1e-6:
        accel_alignment = float(np.dot(action.T, direction_unit) / np.linalg.norm(action))  # [-1, +1]
    else:
        accel_alignment = 0.0

    # --- Reward terms ---
    reward: float = 0.0
    # reward += velocity_alignment * 5    # reward for flying toward target
    # reward += accel_alignment * 2       # smaller reward for thrust direction
    # reward -= acc_norm            # penalty for large acceleration use
    
    # squared distance penalty and reward
    reward += (1.0 - distanceScaled) * 10.0  # closer is better
    reward -= distanceScaled ** 2 * 5.0      # far is worse

    if collision or outOfArea or counterExceed:
        reward -= 50.0  # large penalty for failure
    if isBounced:
        reward -= 20.0
    if targetHit:
        reward += 100.0  # large bonus for success

    return reward

def mainLoop(stepCounter: int = 0, batching: bool = True) -> tuple[int, bool]:
    dt = 0.2  # Time step for simulation

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
    temperature: float = 0.0

    total_critic_loss: float = 0.0
    total_actor_loss: float = 0.0
    total_temperature_loss: float = 0.0
    total_temperature: float = 0.0
    n_bounced: int = 0

    nextState: floatMatrix = np.concatenate((
        missile.position.flatten().reshape(-1) / enviroment.map.shape[0],  # x,y position
        missile.velocity.flatten().reshape(-1) / 100,  # x,y velocity
        target.position.flatten().reshape(-1) / enviroment.map.shape[0],  # target x,y position
    )).reshape(-1)

    discountedSum: float = 0.0
    
    while not done:
        state = nextState

        action = rlAgent.choose_action(state).numpy()
        translationalForceCommand = np.array([[action[0,0]], [action[0,1]], [action[0,2]]], dtype=np.float64)

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

        # Using the consecutive steps technique
        if batching and np.random.uniform() < 0.5:
            for _ in range(4):
                missile.update(translationalForceCommand, angularMomentCommand, dt)

        missile.update(translationalForceCommand, angularMomentCommand, dt)

        nextState: floatMatrix = np.concatenate((
            missile.position.flatten().reshape(-1) / enviroment.map.shape[0],  # x,y position
            missile.velocity.flatten().reshape(-1) / 100,  # x,y velocity
            target.position.flatten().reshape(-1) / enviroment.map.shape[0],  # target x,y position
        )).reshape(-1)

        #collision: bool = enviroment.check_collision_with_terrain(missile.position)
        isBounced, outsideX, outsideY, outsideZ = enviroment.put_inside_bounds()
        outOfArea: bool = not enviroment.check_in_bounds(missile.position)
        targetHit: bool = enviroment.check_collision_with_target(50.0, positionBefore)
        counterExceed: bool = internalCounter > 2000

        if targetHit:
            flightLogger.writeInfo("Target hit, exiting round")
            done = True

        if outOfArea:
            flightLogger.writeInfo("Missile out of bounds, exiting round")
            done = True

        if counterExceed:
            flightLogger.writeInfo("Max steps reached, exiting round")
            done = True

        if isBounced:
            n_bounced += 1

        reward: float = giveReward(
            missilePosition=missile.position,
            targetPosition=target.position,
            missileVelocity=missile.velocity,
            action=translationalForceCommand,
            collision=False,
            outOfArea=outOfArea,
            targetHit=targetHit,
            counterExceed=counterExceed,
            isBounced=isBounced,
            outsideZ=outsideZ
        )
        totalReward += reward
        discountedSum += reward * (rlAgent.gamma ** internalCounter)

        #if collision:
        #    flightLogger.writeInfo("Collision with terrain, exiting round")
        #    done = True

        rlAgent.addToReplayBuffer(state, action.reshape(-1), reward, nextState, done)
        
        # Train the agent and get losses, note that it may give None if not enough samples
        critic_loss1, critic_loss2, actor_loss, temperature_loss, temperature = rlAgent.train()
        critic_loss1 = critic_loss1
        critic_loss2 = critic_loss2
        critic_loss = critic_loss1 + critic_loss2

        if batching and critic_loss != 0.0:
            print("Disabling batching for this episode due to successful training step.")
            batching = False

        total_critic_loss += critic_loss1 + critic_loss2
        total_actor_loss += actor_loss
        total_temperature_loss += temperature_loss
        total_temperature += temperature

        flightLogger.lossLog(
            timeFloat=logTime,
            criticLoss1=critic_loss1,
            criticLoss2=critic_loss2,
            actorLoss=actor_loss,
            temperatureLoss=temperature_loss,
        )

        if stepCounter % 100 == 0 or done:
            print(f"[{stepCounter}][{internalCounter}]State: {state}, Action: {action}, Reward: {reward:.2f}, Critic Loss: {critic_loss:.2f}, Actor Loss: {actor_loss:.2f}, Temp Loss: {temperature_loss:.2f}, Temperature: {temperature:.2f}, Bounced: {n_bounced}")

        stepCounter += 1
        internalCounter += 1
        positionBefore = missile.position.copy()

    averageReward: float = totalReward / internalCounter if internalCounter > 0 else 0.0
    averageCriticLoss: float = total_critic_loss / internalCounter if internalCounter > 0 else 0.0
    averageActorLoss: float = total_actor_loss / internalCounter if internalCounter > 0 else 0.0
    averageTemperatureLoss: float = total_temperature_loss / internalCounter if internalCounter > 0 else 0.0
    averageTemperature: float = total_temperature / internalCounter if internalCounter > 0 else 0.0

    print(f"Round finished with total reward: {totalReward:.2f}, average reward: {averageReward:.2f}, average critic loss: {averageCriticLoss:.4f}, average actor loss: {averageActorLoss:.4f}, average temperature loss: {averageTemperatureLoss:.4f}, discounted sum: {discountedSum:.2f}, n_bounced: {n_bounced}")
    flightLogger.addReward(runCounter, reward=totalReward, averageReward=averageReward, averageCriticLoss=averageCriticLoss, averageActorLoss=averageActorLoss, averageTemperatureLoss=averageTemperatureLoss, averageTemperature=averageTemperature, n_bounced=n_bounced, discountedSum=discountedSum)
    return stepCounter, batching

if __name__ == "__main__":
    runCounter: int = 0
    stepCounter: int = 0
    numEpisodes: int = 500000
    trainingName: str = "SACTraining3D"

    (missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity,
     targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity) = createInitValues()
    
    eomMissile = EOM(config.MissileConfig.MASS, config.MissileConfig.INERTIA_MATRIX, g=0.0)
    eomTarget = EOM(config.TargetConfig.MASS, config.TargetConfig.INERTIA_MATRIX, g=0.0) # Static target, g=0.0

    missile = Missile(missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity, config.MissileConfig.Constraints(), eomMissile)
    target = Target(targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity, config.TargetConfig.Constraints(), eomTarget)

    enviroment = Enviroment(1000, 1000, numMountains=0, maxHeight=190, missile=missile, target=target, maxAllowableHeight=1000)

    flightLogger = FlightLogger(unique_name=trainingName, run_id=runCounter, map=enviroment.map)

    rlAgent = Agent(trainingName=trainingName, stateDims=9, nActions=3, gamma=0.99, learningRate=0.0003, tau=0.005, batchSize=512, minBufferSize=5000, actionScale=100.0, temperature=1.0)

    batching: bool = True
    while runCounter < numEpisodes:
        try:
            stepCounter, batching = mainLoop(stepCounter, batching)
        except KeyboardInterrupt:
            print(f"Simulation interrupted by user after {runCounter} runs.")
            break

        print(f"Completed runs: {runCounter}, resetting simulation.")
        runCounter += 1

        if runCounter % 100 == 0:
            rlAgent.saveModels(f"flightData/{trainingName}/models")
            print("Models saved.")

        resetTargetAndMissile()
        flightLogger.reset(unique_name=trainingName, run_id=runCounter, map=enviroment.map)
        # enviroment.resetMap()

    flightLogger.deleteLastFlightLog()
    flightLogger.exit()

    print("Simulation finished.")
