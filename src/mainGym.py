import os
from tensorflow.python.util import deprecation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np
import gymnasium as gym

import time

from guidance.QNetwork import Agent

from flightData.flightLogger import FlightLogger

if __name__ == "__main__":
    # Creating a ReplayBuffer for the training process

    env = gym.make("BipedalWalker-v3", render_mode="human")

    state, info = env.reset()
    running = True

    rlAgent = Agent(trainingName="BipedalWalkerHardcore-v3-Agent", stateDims=24, nActions=4, gamma=0.99, learningRate=0.0005, tau=0.005, batchSize=256, minBufferSize=10000, actionScale=1.0, temperature=1.0)

    flightLogger = FlightLogger(unique_name="BipedalWalker-v3-FlightLogger", run_id=0)

    critic_loss1, critic_loss2, actor_loss, temperature_loss, temperature = 0.0, 0.0, 0.0, 0.0, 0.0
    totalReward, totalCriticLoss, totalActorLoss, totalTempLoss, totalTemp = 0.0, 0.0, 0.0, 0.0, 0.0

    counter = 0
    runCounter = 0
    discountedSum = 0.0
    check = 1
    cons_actions = 4
    next_state, reward, terminated, truncated, info = None, None, None, None, None

    initTime = time.time()

    while running:
        action = rlAgent.choose_action(state)
        action = tf.reshape(action, (rlAgent.nActions)).numpy()

        # Using the consecutive steps technique
        if check == 1 and np.random.uniform() < 0.5:
            # print(self.replay_buffer.n_entries)
            for i in range(cons_actions):
                env.step(action)

        try:
            next_state, reward, terminated, truncated, info = env.step(action)
            reward = reward * 5.0  # scale reward
        except Exception as e:
            print(f"Error occurred while stepping in environment: {e}")
            continue
        next_state = next_state.reshape(-1)
        done = terminated or truncated

        discountedSum += reward * (rlAgent.gamma ** counter)

        rlAgent.addToReplayBuffer(state, action, reward, next_state, done)

        if check and rlAgent.shouldTrain():
            check = 0

        critic_loss1, critic_loss2, actor_loss, temperature_loss, temperature = rlAgent.train() 

        logTime = time.time() - initTime

        flightLogger.lossLog(
            timeFloat=logTime,
            criticLoss1=critic_loss1,
            criticLoss2=critic_loss2,
            actorLoss=actor_loss,
            temperatureLoss=temperature_loss,
            temperature=temperature,
        )

        state = next_state
                
        counter += 1

        totalReward += reward
        totalCriticLoss += (critic_loss1 + critic_loss2) / 2
        totalActorLoss += actor_loss
        totalTempLoss += temperature_loss
        totalTemp += temperature

        if done:
            print(f"Episode finished: Total Reward: {totalReward:.2f}, Avg Critic Loss: {totalCriticLoss/counter:.4f}, Avg Actor Loss: {totalActorLoss/counter:.4f}, Avg Temp Loss: {totalTempLoss/counter:.4f}, Temperature: {temperature:.4f}, Critic Losses: {critic_loss1:.4f}, {critic_loss2:.4f}, Actor Loss: {actor_loss:.4f}, Temp Loss: {temperature_loss:.4f}")
            
            state, info = env.reset()
            done = False
        
            averageReward: float = totalReward / counter if counter > 0 else 0.0
            averageCriticLoss: float = totalCriticLoss / counter if counter > 0 else 0.0
            averageActorLoss: float = totalActorLoss / counter if counter > 0 else 0.0
            averageTemperatureLoss: float = totalTempLoss / counter if counter > 0 else 0.0
            averageTemperature: float = totalTemp / counter if counter > 0 else 0.0

            flightLogger.addReward(runCounter, reward=totalReward, averageReward=averageReward, averageCriticLoss=averageCriticLoss, averageActorLoss=averageActorLoss,
                               averageTemperatureLoss=averageTemperatureLoss, averageTemperature=averageTemperature, n_bounced=0, discountedSum=discountedSum,
                               actionSmallerOddForPID=0)
            
            runCounter += 1
            counter = 0
            totalReward = 0.0
            totalCriticLoss = 0.0
            totalActorLoss = 0.0
            totalTempLoss = 0.0
            totalTemp = 0.0
            discountedSum = 0.0
            flightLogger.reset(unique_name="BipedalWalker-v3-FlightLogger", run_id=runCounter)

    env.close()