import os
from tensorflow.python.util import deprecation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import gymnasium as gym

from guidance.SACStolen import * 

if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    # Creating a ReplayBuffer for the training process

    env = gym.make("MountainCarContinuous-v0", render_mode="human")

    state, info = env.reset()
    running = True

    rlAgent = Agent(trainingName="MountainCarContinuous-v0-Agent", stateDims=2, nActions=1, gamma=0.9999, learningRate=0.0003, tau=0.01, batchSize=512, minBufferSize=400, actionScale=1.0, temperature=0.3)

    critic_loss1, critic_loss2, actor_loss, temperature_loss, temperature = 0.0, 0.0, 0.0, 0.0, 0.0
    totalReward, totalCriticLoss, totalActorLoss, totalTempLoss = 0.0, 0.0, 0.0, 0.0

    counter = 0
    check = 1
    cons_actions = 4

    while running:
        action = rlAgent.choose_action(state)

        # Using the consecutive steps technique
        if check == 1 and np.random.uniform() < 0.5:
            # print(self.replay_buffer.n_entries)
            for i in range(cons_actions):
                env.step(action)

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = next_state.reshape(-1)
        done = terminated or truncated

        rlAgent.addToReplayBuffer(state, action, reward, next_state, done)

        critic_loss1, critic_loss2, actor_loss, temperature_loss, temperature = rlAgent.train() 

        state = next_state
                
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