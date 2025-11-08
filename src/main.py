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
from guidance.PIDAction import PIDAction

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

PENALTY_STEP: float = 0.1
PENALTY_DISTANCE_FACTOR: float = 1.0
PENALTY_VELOCITY_FACTOR: float = 0.0
PENALTY_ACTION_FACTOR: float = 0.0
PENALTY_COLLISION: float = 0.0
PENALTY_OUT_OF_AREA: float = 0.0
REWARD_TARGET_HIT: float = 1000.0
REWARD_DISTANCE_BONUS_FACTOR: float = 0.0
PENALTY_COUNTER_EXCEED: float = 0.0
PENALTY_BOUNCE: float = 0.0
REWARD_PROGRESS_FACTOR: float = 0.0

def getRandomInitCoordinates(minValue: float, maxValue: float, minHeight: float = 200.0, maxHeight: float = 400.0) -> floatMatrix:
    x = np.random.uniform(minValue, maxValue)
    y = 10
    z = 10
    return np.array([[x], [y], [z]], dtype=np.float64)

def getRandomInitVelocity(minValue: float, maxValue: float, zVel: float = 30.0) -> floatMatrix:
    vx = np.random.uniform(minValue, maxValue)
    vy = 0
    vz = zVel  # Fixed initial upward velocity
    return np.array([[vx], [vy], [vz]], dtype=np.float64)

def createInitValues():
    missileInitPosition: floatMatrix = getRandomInitCoordinates(100.0, 900.0, minHeight=100.0, maxHeight=600.0)
    missileInitQuaternions: floatMatrix = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64)  # Identity quaternion
    missileInitVelocity: floatMatrix = getRandomInitVelocity(0.0, 0.0, zVel=0.0)
    missileInitAngularVelocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

    targetInitPosition: floatMatrix = getRandomInitCoordinates(400.0, 400.0, minHeight=200.0, maxHeight=400.0)
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
    distanceBefore: float,
) -> tuple[float, float]:
    """
    Improved reward for 2D missile tracking (static target).

    - Uses change in distance to target (preferred) or velocity projection if prev position not available.
    - Penalizes squared action magnitude (energy).
    - Adds terminal bonuses/penalties for hit/collision/outOf-area.
    - Returns a scalar reward (tunable via constants below).
    """
    
    # --- Distance and direction ---
    direction_to_target = targetPosition - missilePosition
    distance = np.linalg.norm(direction_to_target) + 1e-9
    direction_unit = direction_to_target / distance
    distanceNorm = distance / 1414.21356  # normalize using max distance in the environment (~1414.21 for 1000x1000x1000), [0, 1]

    # --- Change in distance based reward component ---
    distance_change = distanceBefore - distance  # Positive if getting closer
    distance_change_norm = distance_change / 1414.21356  # normalize

    # --- Velocity alignment (cosine similarity) ---
    vel_norm = np.linalg.norm(missileVelocity) + 1e-9
    velocity_alignment = float(np.dot(missileVelocity.T, direction_unit) / vel_norm)  # [-1, +1]
    velocity_alignment = min(velocity_alignment, 0.0)

    # --- Normalize action to get acceleration direction ---    
    action_norm = np.linalg.norm(action) / 500.0 + 1e-9

    # --- Acceleration alignment (cosine similarity) ---
    if np.linalg.norm(action) > 1e-9:
        accel_alignment = float(np.dot(action.T, direction_unit) / np.linalg.norm(action))  # [-1, +1]
        accel_alignment = min(accel_alignment, 0.0)
    else:
        accel_alignment = 0.0

    # --- Reward calculation ---
    reward = -PENALTY_STEP * ((action_norm + 1)** 2)  # Small step penalty scaled by action magnitude squared

    reward += REWARD_PROGRESS_FACTOR * distance_change_norm  # Reward for getting closer
    reward -= PENALTY_DISTANCE_FACTOR * distanceNorm  # Penalize distance to target
    reward -= PENALTY_VELOCITY_FACTOR * velocity_alignment  # Penalize velocity away from target
    reward -= PENALTY_ACTION_FACTOR * accel_alignment  # Penalize acceleration away from target

    if collision:
        reward -= PENALTY_COLLISION  # Large penalty for collision

    if outOfArea:
        reward -= PENALTY_OUT_OF_AREA  # Penalty for going out of area

    if counterExceed:
        reward -= PENALTY_COUNTER_EXCEED  # Penalty for exceeding step counter

    if isBounced:
        reward -= PENALTY_BOUNCE  # Penalty for bouncing

    if targetHit:
        reward += REWARD_TARGET_HIT  # Reward for hitting the target

    if outOfArea or collision or counterExceed or targetHit:
        # additional reward depending on the distance at termination, non-linear scaling
        reward += REWARD_DISTANCE_BONUS_FACTOR * np.exp(-5.0 * distanceNorm)

    return reward, distance

def mainLoop(stepCounter: int = 0, batching: bool = True, logEachStep: bool = True, logGeneralInfo: bool = True, useConsecutiveSteps: bool = True) -> tuple[int, bool]:
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

    reward: float = 0.0
    distanceBefore: float = np.linalg.norm(target.position - missile.position) + 1e-9
    isBounced: bool = False

    nextState: floatMatrix = np.concatenate((
        missile.position.flatten()[0].reshape(-1) / enviroment.map.shape[0],  # x position
        missile.velocity.flatten()[0].reshape(-1) / config.MissileConfig.Constraints.MAX_VELOCITY,  # x velocity
        # target.position.flatten()[0].reshape(-1) / enviroment.map.shape[0],  # target x position
    )).reshape(-1)

    discountedSum: float = 0.0
    actionSmallerOddForPID: float = 0.0 if rlAgent.shouldTrain() else 0.0

    actionDecision: str = "PID" if np.random.uniform() < actionSmallerOddForPID else "RL"
    print(f"Action decision for this episode: {actionDecision}")

    if logEachStep:
        flightLogger.writeInfo(f"Action decision for this episode: {actionDecision} (PID odds: {actionSmallerOddForPID:.2f})")

    while not done:
        state = nextState

        if actionDecision == "PID":
            pidAction.updateSetpoint(target.position)
            action = pidAction.getAction(missile.position, dt).reshape(1, -1)
        elif actionDecision == "RL":
            action = rlAgent.choose_action(state).numpy()
        translationalForceCommand = np.array([[action[0,0]], [0.0], [0.0]], dtype=np.float64)

        # Log flight data
        logTime: float = time.time() - initTime

        if logEachStep:
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
        if useConsecutiveSteps and batching and np.random.uniform() < 0.5:
            for _ in range(4):
                missile.update(translationalForceCommand, angularMomentCommand, dt)

        missile.update(translationalForceCommand, angularMomentCommand, dt)

        nextState: floatMatrix = np.concatenate((
            missile.position.flatten()[0].reshape(-1) / enviroment.map.shape[0],  # x position
            missile.velocity.flatten()[0].reshape(-1) / config.MissileConfig.Constraints.MAX_VELOCITY,  # x velocity
            # target.position.flatten()[0].reshape(-1) / enviroment.map.shape[0],  # target x position
        )).reshape(-1)

        #collision: bool = enviroment.check_collision_with_terrain(missile.position)
        isBounced = enviroment.put_inside_bounds()
        outOfArea: bool = not enviroment.check_in_bounds(missile.position)
        targetHit: bool = enviroment.check_collision_with_target(10.0, positionBefore)
        counterExceed: bool = internalCounter > 2000

        if targetHit:
            flightLogger.writeInfo(f"Target hit by {actionDecision} agent!, exiting round")
            done = True

        if outOfArea:
            flightLogger.writeInfo(f"Missile out of bounds by {actionDecision} agent!, exiting round")
            done = True

        if counterExceed:
            flightLogger.writeInfo(f"Max steps reached by {actionDecision} agent!, exiting round")
            done = True

        if isBounced:
            n_bounced += 1

        reward, distanceBefore = giveReward(
            missilePosition=missile.position,
            targetPosition=target.position,
            missileVelocity=missile.velocity,
            action=translationalForceCommand,
            collision=False,
            outOfArea=outOfArea,
            targetHit=targetHit,
            counterExceed=counterExceed,
            isBounced=isBounced,
            distanceBefore=distanceBefore,
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

        if batching and rlAgent.shouldTrain():
            print("Disabling batching for this episode due to successful training step.")
            batching = False

        total_critic_loss += critic_loss1 + critic_loss2
        total_actor_loss += actor_loss
        total_temperature_loss += temperature_loss
        total_temperature += temperature

        if logEachStep:
            flightLogger.lossLog(
                timeFloat=logTime,
                criticLoss1=critic_loss1,
                criticLoss2=critic_loss2,
                actorLoss=actor_loss,
                temperatureLoss=temperature_loss,
                temperature=temperature,
            )

        if stepCounter % 100 == 0 or done:
            print(f"[{stepCounter}][{internalCounter}]nextState: {nextState}, Action: {action}, Reward: {reward:.2f}, Critic Loss: {critic_loss:.2f}, Actor Loss: {actor_loss:.2f}, Temp Loss: {temperature_loss:.2f}, Temperature: {temperature:.2f}, Bounced: {n_bounced}")

        stepCounter += 1
        internalCounter += 1
        positionBefore = missile.position.copy()

    averageReward: float = totalReward / internalCounter if internalCounter > 0 else 0.0
    averageCriticLoss: float = total_critic_loss / internalCounter if internalCounter > 0 else 0.0
    averageActorLoss: float = total_actor_loss / internalCounter if internalCounter > 0 else 0.0
    averageTemperatureLoss: float = total_temperature_loss / internalCounter if internalCounter > 0 else 0.0
    averageTemperature: float = total_temperature / internalCounter if internalCounter > 0 else 0.0

    print(f"{actionDecision} with {actionSmallerOddForPID}|| Round finished with total reward: {totalReward:.2f}, average reward: {averageReward:.2f}, average critic loss: {averageCriticLoss:.4f}, average actor loss: {averageActorLoss:.4f}, average temperature loss: {averageTemperatureLoss:.4f}, discounted sum: {discountedSum:.2f}, n_bounced: {n_bounced}")
    
    if logGeneralInfo:
        flightLogger.addReward(runCounter, reward=totalReward, averageReward=averageReward, averageCriticLoss=averageCriticLoss, averageActorLoss=averageActorLoss,
                               averageTemperatureLoss=averageTemperatureLoss, averageTemperature=averageTemperature, n_bounced=n_bounced, discountedSum=discountedSum,
                               actionSmallerOddForPID=actionSmallerOddForPID)

    return stepCounter, batching

if __name__ == "__main__":
    runCounter: int = 0
    stepCounter: int = 0
    numEpisodes: int = 500000
    trainingName: str = "SACTraining_1D_ConsTargetWOState_Consecitive_5000Buffer_MaxAct500_FullDamp_OnlyDistanceReward_NoBouncePenalty_BasePenalty_NoPID"
    logEachStep: bool = False
    logGeneralInfo: bool = True
    loadModels: bool = False
    useConsecutiveSteps: bool = True

    (missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity,
     targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity) = createInitValues()
    
    eomMissile = EOM(config.MissileConfig.MASS, config.MissileConfig.INERTIA_MATRIX, g=0.0)
    eomTarget = EOM(config.TargetConfig.MASS, config.TargetConfig.INERTIA_MATRIX, g=0.0) # Static target, g=0.0

    missile = Missile(missileInitPosition, missileInitQuaternions, missileInitVelocity, missileInitAngularVelocity, config.MissileConfig.Constraints(), eomMissile)
    target = Target(targetInitPosition, targetInitQuaternions, targetInitVelocity, targetInitAngularVelocity, config.TargetConfig.Constraints(), eomTarget)

    enviroment = Enviroment(1000, 1000, numMountains=0, maxHeight=190, missile=missile, target=target, maxAllowableHeight=1000)

    if logEachStep or logGeneralInfo:
        flightLogger = FlightLogger(unique_name=trainingName, run_id=runCounter, map=enviroment.map)

    rlAgent = Agent(trainingName=trainingName, stateDims=2, nActions=1, gamma=0.99, learningRate=0.0003, tau=0.01, batchSize=512, minBufferSize=5000, actionScale=500.0, temperature=1.0)
    pidAction = PIDAction(kp=50.1, ki=2.5, kd=100.0, setpoint=target.position)

    if loadModels:
        runCounter = rlAgent.loadModels(f"flightData/{trainingName}/models")
        print("Models loaded.")

    batching: bool = True

    while runCounter < numEpisodes:

        try:
            stepCounter, batching = mainLoop(stepCounter, batching, logEachStep=logEachStep, logGeneralInfo=logGeneralInfo, useConsecutiveSteps=useConsecutiveSteps)
        except KeyboardInterrupt:
            print(f"Simulation interrupted by user after {runCounter} runs.")
            break

        print(f"Completed runs: {runCounter}, resetting simulation.")
        runCounter += 1

        if runCounter % 100 == 0:
            rlAgent.saveModels(f"flightData/{trainingName}/models", runCounter)
            print("Models saved.")

        resetTargetAndMissile()
        if logEachStep:
            flightLogger.reset(unique_name=trainingName, run_id=runCounter, map=enviroment.map)
        # enviroment.resetMap()

    if logEachStep:
        flightLogger.deleteLastFlightLog()

    if logEachStep or logGeneralInfo:
        flightLogger.exit()

    print("Simulation finished.")
