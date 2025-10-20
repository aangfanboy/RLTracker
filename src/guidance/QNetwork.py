import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import os
import sys

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from utils.math import floatMatrix

class ReplayBuffer:
    """
    An efficient Replay Buffer for off-policy RL algorithms like SAC.
    Implemented as a circular buffer using NumPy arrays.
    """
    def __init__(self, max_size: int, input_shape: int, n_actions: int):
        self.mem_size = max_size
        self.mem_cntr = 0  # Memory counter

        # Pre-allocate memory for transitions
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # Use bool for terminal flags to save memory
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state: floatMatrix, action: floatMatrix, reward: float, new_state: floatMatrix, done: bool):
        """Stores a new experience in the buffer."""
        # Find the first available index using the modulo operator
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        """Samples a batch of experiences from the buffer."""
        # Determine the current number of stored transitions
        max_mem = min(self.mem_cntr, self.mem_size)

        # Generate a batch of random, unique indices
        batch = np.random.choice(max_mem, batch_size, replace=False)

        # Retrieve the data at the sampled indices (fast vectorized operation)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones
    
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
    
class PolicyNetwork(tf.keras.Model):
    def __init__(self, trainingName: str, modelName: str, n_actions: int, log_std_min: float = -20, log_std_max: float = 2, learningRate: float = 0.0003):
        super(PolicyNetwork, self).__init__(name=modelName)
        self.trainingName = trainingName
        self.modelName = modelName
        self.n_actions = n_actions
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate) # type: ignore

        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.mu = tf.keras.layers.Dense(n_actions, activation=None)
        self.log_std = tf.keras.layers.Dense(n_actions, activation=None)

    def call(self, state: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        log_std = self.log_std(x)

        log_std: tf.Tensor = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)  # type: ignore # Limit log_std to reasonable values
        std: tf.Tensor = tf.exp(log_std) # type: ignore

        return mu, std # type: ignore
    
    def sample_action(self, state: tf.Tensor, training: bool = True) -> tuple[tf.Tensor, tf.Tensor]:
        mu, std = self(state)
        # Create Gaussian distribution
        dist = tfp.distributions.Normal(loc=mu, scale=std)

        # Sample action using reparameterization
        if training:
            epsilon = tf.random.normal(shape=mu.shape)
            action = tf.add(mu, tf.multiply(std, epsilon))
        else:
            action = mu

        action_tanh = tf.tanh(action)

        log_prob = tf.reduce_sum(dist.log_prob(action) - tf.math.log(1 - action_tanh**2 + 1e-6), axis=1)

        return action_tanh, log_prob
        
class Agent(tf.Module):
    def saveModels(self, path: str):
        """Saves the models to the specified path."""
        os.makedirs(path, exist_ok=True)

        self.criticModel1.save_weights(os.path.join(path, f"{self.trainingName}_critic1.weights.h5"))
        self.criticModel2.save_weights(os.path.join(path, f"{self.trainingName}_critic2.weights.h5"))
        self.policyModel.save_weights(os.path.join(path, f"{self.trainingName}_policy.weights.h5"))

        with open(os.path.join(path, f"{self.trainingName}_temperature.txt"), 'w') as f:
            f.write(str(self.log_alpha.numpy()))

    def loadModels(self, path: str):
        """Loads the models from the specified path."""
        self.criticModel1.load_weights(os.path.join(path, f"{self.trainingName}_critic1.weights.h5"))
        self.criticModel2.load_weights(os.path.join(path, f"{self.trainingName}_critic2.weights.h5"))
        self.policyModel.load_weights(os.path.join(path, f"{self.trainingName}_policy.weights.h5"))

        with open(os.path.join(path, f"{self.trainingName}_temperature.txt"), 'r') as f:
            temp_value = float(f.read())
            self.log_alpha.assign(tf.convert_to_tensor(temp_value))

    def __init__(self, trainingName: str, stateDims: int, nActions: int, gamma: float, learningRate: float = 0.0003, tau: float = 0.005, batchSize: int = 256, minBufferSize: int = 1000):
        self.trainingName = trainingName
        self.stateDims = stateDims
        self.nActions = nActions
        self.gamma = gamma
        self.learningRate = learningRate
        self.tau = tau
        self.batchSize = batchSize
        self.minBufferSize = max(minBufferSize, batchSize)

        self.log_alpha = tf.Variable(tf.convert_to_tensor(-8.111), dtype=tf.float32)  # Log temperature parameter
        self.targetEntropy = -nActions  # Target entropy for automatic temperature tuning

        self.replayBuffer = ReplayBuffer(max_size=1000000, input_shape=stateDims, n_actions=nActions)

        self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningRate) # type: ignore

        self.criticModel1 = QCriticNetwork(trainingName, "Critic1", learningRate=self.learningRate)
        self.criticModel2 = QCriticNetwork(trainingName, "Critic2", learningRate=self.learningRate)

        self.targetCriticModel1 = QCriticNetwork(trainingName, "Trget_critic1", learningRate=-1.0)
        self.targetCriticModel2 = QCriticNetwork(trainingName, "Target_critic2", learningRate=-1.0)

        self.updateTargetNetworks(tau=1.0)  # Hard update at the beginning

        self.policyModel = PolicyNetwork(trainingName, "Policy", nActions, learningRate=self.learningRate)

    def addToReplayBuffer(self, state: floatMatrix, action: floatMatrix, reward: float, new_state: floatMatrix, done: bool):
        self.replayBuffer.store_transition(state, action, reward, new_state, done)

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
            next_action, log_probabilities = self.policyModel.sample_action(new_state, training=True)
            next_state_action = tf.concat([new_state, next_action], axis=1)

            # Compute target Q-values using target networks
            target_q1 = self.targetCriticModel1(next_state_action)
            target_q2 = self.targetCriticModel2(next_state_action)

            # Correct calculation
            target_q_values = tf.minimum(target_q1, target_q2) - self.getTemperature() * log_probabilities
            target_value = reward + self.gamma * (1 - done) * target_q_values
            target_value = tf.stop_gradient(target_value)

            critic_loss1 = tf.reduce_mean((current_q1 - target_value) ** 2)
            critic_loss2 = tf.reduce_mean((current_q2 - target_value) ** 2)

        # Compute gradients and update critic networks
        critic_grad1 = tape.gradient(critic_loss1, self.criticModel1.trainable_variables)
        critic_grad2 = tape.gradient(critic_loss2, self.criticModel2.trainable_variables)

        self.criticModel1.optimizer.apply_gradients(zip(critic_grad1, self.criticModel1.trainable_variables))
        self.criticModel2.optimizer.apply_gradients(zip(critic_grad2, self.criticModel2.trainable_variables))

        del tape

        with tf.GradientTape() as tape:
            new_action, log_probabilities = self.policyModel.sample_action(state)
            state_new_action = tf.concat([state, new_action], axis=1)

            q1_new_policy = self.criticModel1(state_new_action)
            q2_new_policy = self.criticModel2(state_new_action)
            q_new_policy = tf.minimum(q1_new_policy, q2_new_policy)

            actor_loss = tf.reduce_mean(self.getTemperature() * log_probabilities - q_new_policy)

        # Compute gradients and update policy network
        actor_grad = tape.gradient(actor_loss, self.policyModel.trainable_variables)
        self.policyModel.optimizer.apply_gradients(zip(actor_grad, self.policyModel.trainable_variables))

        with tf.GradientTape() as tape:
            # Temperature loss for automatic entropy tuning
            new_action, log_probabilities = self.policyModel.sample_action(state)
            temperature_loss = tf.reduce_mean(-self.getTemperature() * (log_probabilities + self.targetEntropy))

        temperature_grad = tape.gradient(temperature_loss, [self.log_alpha])
        self.optimizer.apply_gradients(zip(temperature_grad, [self.log_alpha]))

        return critic_loss1, critic_loss2, actor_loss, temperature_loss
    
    def choose_action(self, observation: floatMatrix) -> tf.Tensor:
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        action, _ = self.policyModel.sample_action(state, training=False)

        return action

    def train(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor] | None:
        if self.replayBuffer.mem_cntr < self.minBufferSize:
            return None
        
        states, actions, rewards, new_states, dones = self.replayBuffer.sample_buffer(self.batchSize)

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        critic_loss1, critic_loss2, actor_loss, temperature_loss = self.train_step(states, actions, rewards, new_states, dones)

        # Soft update target networks
        self.updateTargetNetworks(self.tau, clipTemp=True)

        return critic_loss1, critic_loss2, actor_loss, temperature_loss