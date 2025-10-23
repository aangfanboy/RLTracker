import gymnasium as gym

import numpy as np
import time
from numpy.typing import NDArray

from guidance.QNetwork import Agent

if __name__ == "__main__":
    # Create the Gym environment, open a window to render
    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    state, info = env.reset()
    running = True

    rlAgent = Agent(trainingName="MountainCarContinuous-v0-Agent", stateDims=2, nActions=1, gamma=0.9999, learningRate=0.0003, tau=0.01, batchSize=512, minBufferSize=1000, actionScale=1.0)

    critic_loss1, critic_loss2, actor_loss, temperature_loss, temperature = 0.0, 0.0, 0.0, 0.0, 0.0
    totalReward, totalCriticLoss, totalActorLoss, totalTempLoss = 0.0, 0.0, 0.0, 0.0

    counter = 0

    while running:
        action = rlAgent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = next_state.reshape(-1)
        done = terminated or truncated

        rlAgent.addToReplayBuffer(state, action, reward, next_state, done)

        state = next_state
                
        trainResults = rlAgent.train()
        if trainResults is not None:
            critic_loss1, critic_loss2, actor_loss, temperature_loss, temperature = trainResults

        counter += 1

        totalReward += reward
        totalCriticLoss += (critic_loss1 + critic_loss2) / 2
        totalActorLoss += actor_loss
        totalTempLoss += temperature_loss

        if done:
            print(f"Episode finished: Total Reward: {totalReward:.2f}, Avg Critic Loss: {totalCriticLoss/counter:.4f}, Avg Actor Loss: {totalActorLoss/counter:.4f}, Avg Temp Loss: {totalTempLoss/counter:.4f}, Temperature: {temperature:.4f}, Critic Losses: {critic_loss1:.4f}, {critic_loss2:.4f}, Actor Loss: {actor_loss:.4f}, Temp Loss: {temperature_loss:.4f}")
            totalReward, totalCriticLoss, totalActorLoss, totalTempLoss = 0.0, 0.0, 0.0, 0.0
            counter = 0
            state, info = env.reset()
            done = False

    env.close()