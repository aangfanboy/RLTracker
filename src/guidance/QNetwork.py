import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque

import os
import sys

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.math import floatMatrix

class ReplayBuffer(object):
    def __init__(self, buffer_size: int) -> None:
        self.buffer_size: int = buffer_size
        self.num_experiences: int = 0
        self.buffer: deque = deque()

    def get_batch(self, batch_size: int) -> dict:
        # Randomly sample batch_size examples
        experiences = random.sample(self.buffer, batch_size)
        states = np.asarray([exp[0] for exp in experiences], np.float32).reshape(batch_size, -1)
        actions = np.asarray([exp[1] for exp in experiences], np.float32).reshape(batch_size, -1)
        rewards = np.asarray([exp[2] for exp in experiences], np.float32).reshape(batch_size, -1)
        new_states = np.asarray([exp[3] for exp in experiences], np.float32).reshape(batch_size, -1)
        dones = np.asarray([exp[4] for exp in experiences], np.float32).reshape(batch_size, -1)

        return states, actions, rewards, new_states, dones

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    @property
    def size(self) -> int:
        return self.buffer_size

    @property
    def n_entries(self) -> int:
        # If buffer is full, return buffer size
        # Otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

class PolicyNetwork(tf.keras.Model):
    def __init__(self, trainingName: str, modelName: str, n_actions: int, log_std_min: float = -20, log_std_max: float = 2, learningRate: float = 0.0003, actionScale: float = 1.0):
        super(PolicyNetwork, self).__init__(name=modelName)
        self.trainingName = trainingName
        self.modelName = modelName
        self.n_actions = n_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.actionScale = actionScale

        self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate) # type: ignore

        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.mu = tf.keras.layers.Dense(n_actions, activation=None)
        self.log_std = tf.keras.layers.Dense(n_actions, activation=None)

    def call(self, state: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.fc1(state)
        x = self.fc2(x)
        mean = self.mu(x)
        log_std = self.log_std(x)
        log_std_clipped = tf.clip_by_value(log_std, -20, 2)
        normal_dist = tfp.distributions.Normal(mean, tf.exp(log_std_clipped))
        action = normal_dist.sample()
        squashed_actions = tf.tanh(action)
        logprob = normal_dist.log_prob(action) - tf.math.log(1.0 - tf.pow(squashed_actions, 2) + 1e-6)
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)

        evalAction = tf.tanh(mean) * self.actionScale
        return squashed_actions * self.actionScale, logprob, evalAction

class QCriticNetwork(tf.keras.Model):
    def __init__(self, trainingName: str, modelName: str, learningRate: float = 0.0003):
        super(QCriticNetwork, self).__init__(name=modelName)
        self.trainingName = trainingName
        self.modelName = modelName

        if learningRate <= 0:
            self.optimizer = None
        else:
            self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate) # type: ignore

        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)

    def call(self, state_action: tf.Tensor) -> tf.Tensor:
        x = self.fc1(state_action)
        x = self.fc2(x)
        q_value = self.q(x)

        return q_value


class Agent:
    def saveModels(self, path: str, runCounter: int):
        """Saves the models to the specified path."""
        os.makedirs(path, exist_ok=True)

        self.criticModel1.save_weights(os.path.join(path, f"{self.trainingName}_critic1.weights.h5"))
        self.criticModel2.save_weights(os.path.join(path, f"{self.trainingName}_critic2.weights.h5"))
        self.policyModel.save_weights(os.path.join(path, f"{self.trainingName}_policy.weights.h5"))

        with open(os.path.join(path, f"{self.trainingName}_temperature.txt"), 'w') as f:
            f.write(str(self.log_alpha.numpy()))

        with open(os.path.join(path, f"{self.trainingName}_runCounter.txt"), 'w') as f:
            f.write(str(runCounter))

    def loadModels(self, path: str):
        """Loads the models from the specified path."""
        self.criticModel1.load_weights(os.path.join(path, f"{self.trainingName}_critic1.weights.h5"))
        self.criticModel2.load_weights(os.path.join(path, f"{self.trainingName}_critic2.weights.h5"))
        self.policyModel.load_weights(os.path.join(path, f"{self.trainingName}_policy.weights.h5"))

        with open(os.path.join(path, f"{self.trainingName}_temperature.txt"), 'r') as f:
            temp_value = float(f.read())
            self.log_alpha.assign(tf.convert_to_tensor(temp_value))

            print(f"Loaded temperature: {temp_value}")

        self.updateTargetNetworks(tau=1.0)  # Hard update after loading

        run_counter_value: int = 0

        with open(os.path.join(path, f"{self.trainingName}_runCounter.txt"), 'r') as f:
            run_counter_value = int(f.read())
            print(f"Loaded run counter: {run_counter_value}")

        return run_counter_value
            
    def __init__(self, trainingName: str, stateDims: int, nActions: int, gamma: float, learningRate: float = 0.0003, tau: float = 0.005, batchSize: int = 256, minBufferSize: int = 1000, actionScale: float = 1.0, temperature: float = 0.2):
        self.trainingName = trainingName
        self.stateDims = stateDims
        self.nActions = nActions
        self.gamma = gamma
        self.learningRate = learningRate
        self.tau = tau
        self.batchSize = batchSize
        self.actionScale = actionScale
        self.minBufferSize = max(minBufferSize, batchSize)

        self.replay_buffer = ReplayBuffer(self.minBufferSize)

        self.log_alpha = tf.Variable(tf.convert_to_tensor(tf.math.log(temperature)), dtype=tf.float32)  # type: ignore
        self.targetEntropy = -nActions  # Target entropy for automatic temperature tuning

        self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningRate) # type: ignore

        self.criticModel1 = QCriticNetwork(trainingName, "Critic1", learningRate=self.learningRate)
        self.criticModel2 = QCriticNetwork(trainingName, "Critic2", learningRate=self.learningRate)

        self.targetCriticModel1 = QCriticNetwork(trainingName, "Target_critic1", learningRate=-1.0)
        self.targetCriticModel2 = QCriticNetwork(trainingName, "Target_critic2", learningRate=-1.0)

        input1 = tf.keras.Input(shape=(stateDims + nActions,), dtype=tf.float32)
        self.criticModel1(input1)
        self.targetCriticModel1(input1)
        self.criticModel2(input1)
        self.targetCriticModel2(input1)

        self.updateTargetNetworks(tau=1.0)  # Hard update at the beginning

        self.policyModel = PolicyNetwork(trainingName, "Policy", nActions, learningRate=self.learningRate, actionScale=self.actionScale)

        input2 = tf.keras.Input(shape=(stateDims,), dtype=tf.float32)
        self.policyModel(input2)

    def getTemperature(self) -> tf.Tensor:
        return tf.exp(self.log_alpha) # type: ignore
    
    def updateTargetNetworks(self, tau: float, clipTemp: bool = False):
        """Soft update target networks"""
        for target_param, param in zip(self.targetCriticModel1.trainable_variables, self.criticModel1.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

        for target_param, param in zip(self.targetCriticModel2.trainable_variables, self.criticModel2.trainable_variables):
            target_param.assign(tau * param + (1 - tau) * target_param)

        if clipTemp:
            self.log_alpha.assign(tf.clip_by_value(self.log_alpha, -20.0, 2.0)) # type: ignore

    @tf.function
    def train_step(self, state: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, new_state: tf.Tensor, done: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        with tf.GradientTape(persistent=True) as tape:
            state_action = tf.concat([state, action], axis=1)

            # Compute current Q-values
            current_q1 = self.criticModel1(state_action)
            current_q2 = self.criticModel2(state_action)

            # Compute action from the policy for the successor state
            next_action, log_probabilities, _ = self.policyModel(new_state)
            next_state_action = tf.concat([new_state, next_action], axis=1)

            # Compute target Q-values using target networks
            target_q1 = self.targetCriticModel1(next_state_action)
            target_q2 = self.targetCriticModel2(next_state_action)

            # Correct calculation
            target_q_values = tf.minimum(target_q1, target_q2) - self.getTemperature() * log_probabilities
            target_value = reward + self.gamma * (1 - done) * target_q_values
            target_value = tf.stop_gradient(target_value)

            critic_loss1 = tf.reduce_mean(tf.square(current_q1 - target_value))
            critic_loss2 = tf.reduce_mean(tf.square(current_q2 - target_value))

        # Compute gradients and update critic networks
        critic_grad1 = tape.gradient(critic_loss1, self.criticModel1.trainable_variables)
        critic_grad2 = tape.gradient(critic_loss2, self.criticModel2.trainable_variables)

        self.criticModel1.optimizer.apply_gradients(zip(critic_grad1, self.criticModel1.trainable_variables))
        self.criticModel2.optimizer.apply_gradients(zip(critic_grad2, self.criticModel2.trainable_variables))

        del tape

        with tf.GradientTape(persistent=True) as tape:
            new_action, log_probabilities, _ = self.policyModel(state)
            state_new_action = tf.concat([state, new_action], axis=1)

            q1_new_policy = self.criticModel1(state_new_action)
            q2_new_policy = self.criticModel2(state_new_action)
            q_new_policy = tf.minimum(q1_new_policy, q2_new_policy)

            actor_loss = tf.reduce_mean(self.getTemperature() * log_probabilities - q_new_policy)

            log_probabilities_detached = tf.stop_gradient(log_probabilities)
            temperature_loss = -tf.reduce_mean(self.getTemperature() * (log_probabilities_detached + self.targetEntropy))

        # Compute gradients and update policy network
        actor_grad = tape.gradient(actor_loss, self.policyModel.trainable_variables)
        self.policyModel.optimizer.apply_gradients(zip(actor_grad, self.policyModel.trainable_variables))
            
        temperature_grad = tape.gradient(temperature_loss, [self.log_alpha])
        self.optimizer.apply_gradients(zip(temperature_grad, [self.log_alpha]))

        del tape

        return critic_loss1, critic_loss2, actor_loss, temperature_loss
    
    @tf.function
    def behaviorCloning_train_step(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            new_action, _, _ = self.policyModel(state)
            bc_loss = tf.reduce_mean(tf.square(new_action - action))

        bc_grad = tape.gradient(bc_loss, self.policyModel.trainable_variables)
        self.policyModel.optimizer.apply_gradients(zip(bc_grad, self.policyModel.trainable_variables))

        return bc_loss
    
    def choose_action(self, state: floatMatrix, evaluate: bool = False) -> tf.Tensor:
        action, _, mean = self.policyModel(state.reshape(1, -1).astype(np.float32))

        if evaluate:
            return mean

        return action
    
    def addToReplayBuffer(self, state: floatMatrix, action: floatMatrix, reward: float, new_state: floatMatrix, done: bool):
        self.replay_buffer.add(state, action, reward, new_state, done)

    def resetReplayBuffer(self):
        self.replay_buffer.erase()

    def shouldTrain(self) -> bool:
        return self.replay_buffer.n_entries >= self.minBufferSize

    def train(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor] | None:
        if not self.shouldTrain():
            return 0.0, 0.0, 0.0, 0.0, self.getTemperature()

        states, actions, rewards, new_states, dones = self.replay_buffer.get_batch(self.batchSize)

        critic_loss1, critic_loss2, actor_loss, temperature_loss = self.train_step(states, actions, rewards, new_states, dones)

        # Soft update target networks
        self.updateTargetNetworks(self.tau, clipTemp=True)

        return critic_loss1.numpy(), critic_loss2.numpy(), actor_loss.numpy(), temperature_loss.numpy(), self.getTemperature()

    def behaviorCloningTrain(self) -> tf.Tensor:
        if not self.shouldTrain():
            return tf.convert_to_tensor(0.0)
        
        states, actions, _, _, _ = self.replay_buffer.get_batch(self.batchSize)

        return self.behaviorCloning_train_step(states, actions)