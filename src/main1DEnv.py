from core import missile
from envGame.oneDEnv import Enviroment
from flightData.flightLogger import FlightLogger

import numpy as np
from numpy.typing import NDArray

from guidance.QNetwork import Agent
from guidance.PIDAction import PIDAction

floatMatrix = NDArray[np.float64]

PENALTY_PER_STEP: float = 0.0
REWARD_ACTION_PENALTY_COEFF: float = 0.01
REWARD_DISTANCE_DIFF_COEFF: float = 10.0
PENALTY_BOUNCE_COEFF: float = 5.0
REWARD_VELOCITY_PROJ_COEFF: float = 0.5
REWARD_HIT_BONUS: float = 100.0

def createInitValues():
    missileInitPosition: floatMatrix = np.array([[np.random.uniform(100.0, 900.0)], [np.random.uniform(100.0, 900.0)], [np.random.uniform(100.0, 900.0)]], dtype=np.float64)
    missileInitVelocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

    targetInitPosition: floatMatrix = np.array([[np.random.uniform(100.0, 900.0)], [np.random.uniform(100.0, 900.0)], [np.random.uniform(100.0, 900.0)]], dtype=np.float64)
    targetInitVelocity: floatMatrix = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

    return (missileInitPosition, missileInitVelocity,
            targetInitPosition, targetInitVelocity)

def resetTargetAndMissile():
    (missileInitPosition, missileInitVelocity,
     targetInitPosition, targetInitVelocity) = createInitValues()

    enviroment.point_coordinate = missileInitPosition
    enviroment.point_velocity = missileInitVelocity
    enviroment.target_coordinate = targetInitPosition
    enviroment.target_velocity = targetInitVelocity

def giveReward(
    action: floatMatrix,
    targetHit: bool,
    missilePosition: floatMatrix,
    missileVelocity: floatMatrix,
    targetPosition: floatMatrix,
    distanceBefore: float,
    isBounced: bool,
) -> float:

    # --- Distance and direction ---
    direction_to_target = targetPosition - missilePosition
    distance = np.linalg.norm(direction_to_target) + 1e-9
    direction_unit = direction_to_target / distance

    # --- Normalize action to get acceleration direction ---
    action_norm = np.linalg.norm(action) / (np.sqrt(3) * 1.0)  # Normalize by max possible action magnitude

    # --- Distance-based reward component ---
    distance_delta: float = (distanceBefore - distance) / np.sqrt( (enviroment.xDim **2) + (enviroment.yDim **2) + (enviroment.maxHeight **2) )  # Normalize by max possible distance in the environment

    # --- Velocity alignment (cosine similarity) ---
    vel_norm = np.linalg.norm(missileVelocity) + 1e-9
    velocity_alignment = float(np.dot(missileVelocity.T, direction_unit) / vel_norm)  # [-1, +1]

    # --- Reward calculation ---
    reward: float = -PENALTY_PER_STEP  # Base time step penalty
    reward = -REWARD_ACTION_PENALTY_COEFF * ((action_norm * 2) ** 2)  # Small step penalty scaled by action magnitude squared
    reward += REWARD_DISTANCE_DIFF_COEFF * distance_delta  # Reward for reducing distance to target
    reward += REWARD_VELOCITY_PROJ_COEFF * (velocity_alignment * (vel_norm / 10.0))  # Reward for velocity towards target

    if isBounced:
        reward -= PENALTY_BOUNCE_COEFF  # Penalty for bouncing off boundaries

    if targetHit:
        reward += REWARD_HIT_BONUS  # Reward for hitting the target

    return reward


def behaviorCloningTrain(numBatches: int = 100) -> None:
    batchCounter: int = 0
    dt = 1  # Time step for simulation
    n_bounced: int = 0
    internalCounter: int = 0
    positionBefore: floatMatrix = enviroment.point_coordinate

    bc_loss: float = 0.0
    total_bc_loss: float = 0.0

    while batchCounter < numBatches:
        done = False

        state: floatMatrix = np.array([[
            enviroment.point_coordinate[0] / enviroment.xDim,
            enviroment.point_coordinate[1] / enviroment.yDim,
            enviroment.point_coordinate[2] / enviroment.maxHeight,
            enviroment.point_velocity[0] / 10.,
            enviroment.point_velocity[1] / 10.,
            enviroment.point_velocity[2] / 10.,
            enviroment.target_coordinate[0] / enviroment.xDim,
            enviroment.target_coordinate[1] / enviroment.yDim,
            enviroment.target_coordinate[2] / enviroment.maxHeight,
        ]], dtype=np.float64).reshape(-1)

        pidAction.updateSetpoint(enviroment.target_coordinate)
        action = pidAction.getAction(enviroment.point_coordinate, dt=dt).reshape(-1)
        action = np.clip(action, -rlAgent.actionScale, rlAgent.actionScale)

        enviroment.iterateRK4(acceleration=action.reshape(-1, 1) / 1.0, dt=dt)

        nextState: floatMatrix = np.array([[
            enviroment.point_coordinate[0] / enviroment.xDim,
            enviroment.point_coordinate[1] / enviroment.yDim,
            enviroment.point_coordinate[2] / enviroment.maxHeight,
            enviroment.point_velocity[0] / 10.,
            enviroment.point_velocity[1] / 10.,
            enviroment.point_velocity[2] / 10.,
            enviroment.target_coordinate[0] / enviroment.xDim,
            enviroment.target_coordinate[1] / enviroment.yDim,
            enviroment.target_coordinate[2] / enviroment.maxHeight,
        ]], dtype=np.float64).reshape(-1)

        #collision: bool = enviroment.check_collision_with_terrain(missile.position)
        isBounced = enviroment.put_inside_bounds()
        targetHit: bool = enviroment.check_collision_with_target(40.0, positionBefore=positionBefore, steps=40)
        counterExceed: bool = internalCounter > 1500

        if targetHit:
            print(f"Target hit by agent!, exiting round")
            done = True

        rlAgent.addToReplayBuffer(state, action, 0.0, nextState, done)

        if counterExceed:
            print(f"Max steps reached by agent!, exiting round")
            done = True

        if isBounced:
            n_bounced += 1

        internalCounter += 1
        positionBefore = enviroment.point_coordinate

        if done:
            resetTargetAndMissile()
            internalCounter = 0
            n_bounced = 0
            positionBefore = enviroment.point_coordinate

        if rlAgent.shouldTrain():
            bc_loss = rlAgent.behaviorCloningTrain().numpy()
            total_bc_loss += bc_loss
            batchCounter += 1

        if batchCounter % 10 == 0 and batchCounter > 0:
            print(f"[BC Training][{batchCounter}/{numBatches}] BC Loss: {bc_loss:.4f}, Average BC Loss: {total_bc_loss / batchCounter:.4f}")

    print(f"Behavior Cloning training completed over {numBatches} batches. Average BC Loss: {total_bc_loss / numBatches:.4f}")
    rlAgent.resetReplayBuffer()


def mainLoop(stepCounter: int = 0) -> tuple[int, float, float, float, float, float, int, bool, bool]:
    dt = 1  # Time step for simulation

    done = False

    totalReward: float = 0.0
    internalCounter: int = 0
    positionBefore: floatMatrix = enviroment.point_coordinate
    distanceBefore: float = np.linalg.norm(enviroment.point_coordinate - enviroment.target_coordinate)
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
    isBounced: bool = False

    discountedSum: float = 0.0
    action: floatMatrix = None

    while not done:
        state: floatMatrix = np.array([[
            enviroment.point_coordinate[0] / enviroment.xDim,
            enviroment.point_coordinate[1] / enviroment.yDim,
            enviroment.point_coordinate[2] / enviroment.maxHeight,
            enviroment.point_velocity[0] / 10.,
            enviroment.point_velocity[1] / 10.,
            enviroment.point_velocity[2] / 10.,
            enviroment.target_coordinate[0] / enviroment.xDim,
            enviroment.target_coordinate[1] / enviroment.yDim,
            enviroment.target_coordinate[2] / enviroment.maxHeight,
        ]], dtype=np.float64).reshape(-1)

        action: floatMatrix = rlAgent.choose_action(state).numpy().reshape(-1)

        # Using the consecutive steps technique
        if not rlAgent.shouldTrain() and np.random.uniform() < 0.5:
            for _ in range(4):
                enviroment.iterateRK4(acceleration=action.reshape(-1, 1) / 1.0, dt=dt)

        enviroment.iterateRK4(acceleration=action.reshape(-1, 1) / 1.0, dt=dt)

        nextState: floatMatrix = np.array([[
            enviroment.point_coordinate[0] / enviroment.xDim,
            enviroment.point_coordinate[1] / enviroment.yDim,
            enviroment.point_coordinate[2] / enviroment.maxHeight,
            enviroment.point_velocity[0] / 10.,
            enviroment.point_velocity[1] / 10.,
            enviroment.point_velocity[2] / 10.,
            enviroment.target_coordinate[0] / enviroment.xDim,
            enviroment.target_coordinate[1] / enviroment.yDim,
            enviroment.target_coordinate[2] / enviroment.maxHeight,
        ]], dtype=np.float64).reshape(-1)

        #collision: bool = enviroment.check_collision_with_terrain(missile.position)
        isBounced = enviroment.put_inside_bounds()
        targetHit: bool = enviroment.check_collision_with_target(40.0, positionBefore=positionBefore, steps=40)
        counterExceed: bool = internalCounter > 1500

        if targetHit:
            flightLogger.writeInfo(f"Target hit by agent!, exiting round")
            done = True

        if counterExceed:
            flightLogger.writeInfo(f"Max steps reached by agent!, exiting round")
            done = True

        if isBounced:
            n_bounced += 1

        reward = giveReward(
            action=action,
            targetHit=targetHit,
            missilePosition=enviroment.point_coordinate,
            missileVelocity=enviroment.point_velocity,
            targetPosition=enviroment.target_coordinate,
            distanceBefore=distanceBefore,
            isBounced=isBounced,
        )

        totalReward += reward
        discountedSum += reward * (rlAgent.gamma ** internalCounter)

        rlAgent.addToReplayBuffer(state, action, reward, nextState, done)
        
        # Train the agent and get losses, note that it may give None if not enough samples
        critic_loss1, critic_loss2, actor_loss, temperature_loss, temperature = rlAgent.train()
        critic_loss = critic_loss1 + critic_loss2

        total_critic_loss += critic_loss
        total_actor_loss += actor_loss
        total_temperature_loss += temperature_loss
        total_temperature += temperature

        if stepCounter % 200 == 0 or done:
            print(f"[{stepCounter}][{internalCounter}]nextState: {nextState}, Action: {action}, Reward: {reward:.2f}, Critic Loss: {critic_loss:.2f}, Actor Loss: {actor_loss:.2f}, Temp Loss: {temperature_loss:.2f}, Temperature: {temperature:.2f}, Bounced: {n_bounced}, distanceBefore: {distanceBefore:.2f}")

        stepCounter += 1
        internalCounter += 1
        positionBefore = enviroment.point_coordinate
        distanceBefore = np.linalg.norm(enviroment.point_coordinate - enviroment.target_coordinate)

    averageReward: float = totalReward / internalCounter if internalCounter > 0 else 0.0
    averageCriticLoss: float = total_critic_loss / internalCounter if internalCounter > 0 else 0.0
    averageActorLoss: float = total_actor_loss / internalCounter if internalCounter > 0 else 0.0
    averageTemperatureLoss: float = total_temperature_loss / internalCounter if internalCounter > 0 else 0.0
    averageTemperature: float = total_temperature / internalCounter if internalCounter > 0 else 0.0

    print(f" Round finished with total reward: {totalReward:.2f}, average reward: {averageReward:.2f}, average critic loss: {averageCriticLoss:.4f}, average actor loss: {averageActorLoss:.4f}, average temperature loss: {averageTemperatureLoss:.4f}, discounted sum: {discountedSum:.2f}, n_bounced: {n_bounced}")
    
    if logGeneralInfo:
        flightLogger.addReward(runCounter, reward=totalReward, averageReward=averageReward, averageCriticLoss=averageCriticLoss, averageActorLoss=averageActorLoss,
                               averageTemperatureLoss=averageTemperatureLoss, averageTemperature=averageTemperature, n_bounced=n_bounced, discountedSum=discountedSum)

    return stepCounter, averageReward, averageCriticLoss, averageActorLoss, averageTemperatureLoss, averageTemperature, n_bounced, targetHit, counterExceed

if __name__ == "__main__":
    runCounter: int = 0
    stepCounter: int = 0
    numEpisodes: int = 500000
    trainingName: str = "SACTraining_3D_RandomInitMissile_RandomInitTargetWOState_Consecitive_20000Buffer_10MaxVel_1MaxAct_FullDamp_05VelocityDirReward_100DistanceReward_10HitReward_5BouncePenalty_001Fuel_0Time_HardBehaviorPID_TargetRadius10_LR0003_Tau007"
    logGeneralInfo: bool = True
    loadModels: bool = False
    useConsecutiveSteps: bool = True
    useBehaviorCloningPretrain: bool = True

    (missileInitPosition, missileInitVelocity,
     targetInitPosition, targetInitVelocity) = createInitValues()

    enviroment = Enviroment(1000, 1000, maxHeight=1000.0, maxVelocity=10.0)
    flightLogger = FlightLogger(unique_name=trainingName, run_id=runCounter)

    rlAgent = Agent(trainingName=trainingName, stateDims=9, nActions=3, gamma=0.99, learningRate=0.0003, tau=0.007, batchSize=256, minBufferSize=20000, actionScale=1.0, temperature=1.0)
    pidAction = PIDAction(kp=50.1, ki=2.5, kd=100.0, setpoint=targetInitPosition)

    if loadModels:
        runCounter = rlAgent.loadModels(f"flightData/{trainingName}/models")
        print("Models loaded.")
    elif useBehaviorCloningPretrain:
        behaviorCloningTrain(numBatches=30000)

    numTests: int = 0

    results: list[dict] = []

    while runCounter < numEpisodes:
        try:
            stepCounter, avgReward, _, _, _, _, n_bounced, hit, timeout = mainLoop(stepCounter)

            results.append({
                'stepCounter': stepCounter,
                'avgReward': avgReward,
                'n_bounced': n_bounced,
                'hit': hit,
                'timeout': timeout
            })

            if len(results) >= 100:
                results.pop(0)

        except KeyboardInterrupt:
            print(f"Simulation interrupted by user after {runCounter} runs.")
            break

        print(f"Completed runs: {runCounter}, resetting simulation.")

        # print average rewards as now, and give success rate, bounce rate, timeout rate
        
        totalAvgReward: float = 0.0
        totalHits: int = 0
        totalBounces: int = 0
        totalTimeouts: int = 0  
        totalSuccesses: int = 0
        for res in results:
            totalAvgReward += res['avgReward']
            if res['hit']:
                totalHits += 1
            if res['n_bounced'] > 0:
                totalBounces += 1
            if res['timeout']:
                totalTimeouts += 1

            if res['hit'] and res['n_bounced'] == 0 and not res['timeout']:
                totalSuccesses += 1

        overallAvgReward: float = totalAvgReward / len(results) if len(results) > 0 else 0.0
        hitRate: float = totalHits / len(results) if len(results) > 0 else 0.0
        bounceRate: float = totalBounces / len(results) if len(results) > 0 else 0.0
        timeoutRate: float = totalTimeouts / len(results) if len(results) > 0 else 0.0
        successRate: float = totalSuccesses / len(results) if len(results) > 0 else 0.0

        flightLogger.addTestResults(
            runCounter,
            successRate=successRate,
            failuresByBounce=totalBounces,
            failuresByTimeout=totalTimeouts,
            averageReward=overallAvgReward,
        )

        print(f" Overall Average Reward: {overallAvgReward:.2f}, Hit Rate: {hitRate*100:.2f}%, Bounce Rate: {bounceRate*100:.2f}%, Timeout Rate: {timeoutRate*100:.2f}%, Success Rate: {successRate*100:.2f}%")

        runCounter += 1

        if runCounter % 100 == 0:
            rlAgent.saveModels(f"flightData/{trainingName}/models", runCounter)
            print("Models saved.")

        resetTargetAndMissile()

    flightLogger.exit()

    print("Simulation finished.")
