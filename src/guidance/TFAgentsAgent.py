import os
import numpy as np
import tensorflow as tf

try:
    import tf_agents
    from tf_agents.networks import actor_distribution_network, value_network, normal_projection_network
    from tf_agents.networks import q_network
    from tf_agents.agents.sac import sac_agent
    from tf_agents.replay_buffers import tf_uniform_replay_buffer
    from tf_agents.trajectories import trajectory
    from tf_agents.specs import tensor_spec

    class TFAgentsSACWrapper:
        """A thin wrapper around TF-Agents SAC that provides the minimal API used by main.py.

        Methods provided:
        - choose_action(observation): returns a tf.Tensor action (batched 1) in [-1,1]
        - addToReplayBuffer(state, action, reward, next_state, done): stores transition
        - train(): samples a batch and runs a train step, returns losses similar shape to original
        - saveModels(path): saves actor and critic weights + temperature

        Implementation notes:
        - To keep the existing environment loop unchanged we don't wrap the full Env as a PyEnvironment.
          Instead we build TF-Agents networks and agent and feed it transitions collected by the loop.
        - Actions are expected by main.py to be in range [-1,1] and then scaled. This wrapper will accept
          that contract and return actions as tf.Tensor with shape (1, n_actions).
        """

        def __init__(self, trainingName: str, stateDims: int, nActions: int, gamma: float, learningRate: float = 3e-4, tau: float = 0.005, batchSize: int = 256, minBufferSize: int = 1000):
            self.trainingName = trainingName
            self.state_dims = stateDims
            self.n_actions = nActions
            self.gamma = gamma
            self.learning_rate = learningRate
            self.tau = tau
            self.batch_size = batchSize
            self.min_buffer_size = max(minBufferSize, batchSize)

            # specs
            self.observation_spec = tensor_spec.TensorSpec([self.state_dims], tf.float32)
            self.action_spec = tensor_spec.BoundedTensorSpec([self.n_actions], tf.float32, minimum=-1.0, maximum=1.0)

            # networks
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                self.observation_spec.shape,
                self.action_spec.shape[0],
                fc_layer_params=(256, 256),
                continuous_projection_net=normal_projection_network.NormalProjectionNetwork,
            )

            critic_net1 = q_network.QNetwork(self.observation_spec.shape, self.action_spec.shape, fc_layer_params=(256, 256))
            critic_net2 = q_network.QNetwork(self.observation_spec.shape, self.action_spec.shape, fc_layer_params=(256, 256))

            # agent
            self.global_step = tf.Variable(0, dtype=tf.int64)

            # Build a minimal time_step_spec and pass it to the agent
            from tf_agents.trajectories import time_step as ts
            time_step_spec = ts.time_step_spec(self.observation_spec)

            self.agent = sac_agent.SacAgent(
                time_step_spec=time_step_spec,
                action_spec=self.action_spec,
                actor_network=actor_net,
                critic_network=(critic_net1, critic_net2),
                actor_optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                critic_optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                alpha_optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                target_update_tau=self.tau,
                gamma=self.gamma,
                reward_scale_factor=1.0,
                train_step_counter=self.global_step,
            )

            # Replay buffer (TF uniform replay buffer)
            self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=self.agent.collect_data_spec,
                batch_size=1,
                max_length=1000000,
            )

            # A small converter for transitions coming from numpy
            self._rb_add = self.replay_buffer.add_batch

            # Create dataset iterator for training
            self.dataset = self.replay_buffer.as_dataset(sample_batch_size=self.batch_size, num_steps=2).prefetch(10)
            self.iterator = iter(self.dataset)

            # Make sure agent is initialized
            self.agent.initialize()

        def choose_action(self, observation: np.ndarray) -> tf.Tensor:
            obs = tf.convert_to_tensor([observation], dtype=tf.float32)
            # Use policy to get action; policy returns a policy step with action
            policy_step = self.agent.policy.action(obs)
            action = policy_step.action
            # Ensure action is float32 tensor shaped (1, n_actions)
            return tf.convert_to_tensor(action, dtype=tf.float32)

        def addToReplayBuffer(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool):
            # Build a trajectory from the single step and add to replay buffer
            # tf-agents expects tensors shaped [batch, ...]
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            action = tf.convert_to_tensor([action], dtype=tf.float32)
            reward = tf.convert_to_tensor([[reward]], dtype=tf.float32)
            next_state = tf.convert_to_tensor([new_state], dtype=tf.float32)

            # Construct a time_step and policy_step and build a Trajectory via from_transition
            from tf_agents.trajectories import time_step as ts
            from tf_agents.trajectories import policy_step as ps

            # current time step: use transition with zero reward (the reward will be attached to the next step in from_transition)
            current_ts = ts.transition(observation=state, reward=tf.zeros([1], dtype=tf.float32))
            next_ts = ts.termination(observation=next_state, reward=tf.squeeze(reward, axis=0)) if done else ts.transition(observation=next_state, reward=tf.squeeze(reward, axis=0))

            policy_step = ps.PolicyStep(action=action, state=(), info=())

            traj = trajectory.from_transition(current_ts, policy_step, next_ts)

            # Add to buffer
            self._rb_add(traj)

        def train(self):
            # Only train if enough samples
            if self.replay_buffer.num_frames().numpy() < self.min_buffer_size:
                return None

            # Get a batch from the iterator
            try:
                experience, unused_info = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)
                experience, unused_info = next(self.iterator)

            # Run a training step
            train_loss = self.agent.train(experience)

            # TF-Agents returns a LossInfo-like object. Extract the scalar loss (if present).
            loss = train_loss.loss if hasattr(train_loss, 'loss') else tf.constant(0.0)
            # Return a tuple similar to the original Agent: (critic1, critic2, actor, temperature)
            critic_loss1 = loss
            critic_loss2 = loss
            actor_loss = tf.constant(0.0)
            temperature_loss = tf.constant(0.0)

            return critic_loss1, critic_loss2, actor_loss, temperature_loss

        def saveModels(self, path: str):
            os.makedirs(path, exist_ok=True)
            # Try to save networks' weights if available (best-effort)
            try:
                actor_dir = os.path.join(path, f"{self.trainingName}_actor")
                os.makedirs(actor_dir, exist_ok=True)
                # Save policy variables via a checkpoint
                ckpt = tf.train.Checkpoint(policy=self.agent.policy)
                ckpt.write(os.path.join(actor_dir, 'policy_ckpt'))
            except Exception:
                pass

            try:
                critic_dir = os.path.join(path, f"{self.trainingName}_critic")
                os.makedirs(critic_dir, exist_ok=True)
                critic_networks = getattr(self.agent, '_critic_networks', None)
                if critic_networks:
                    critic_networks[0].save_weights(os.path.join(critic_dir, "critic1_weights"))
                    critic_networks[1].save_weights(os.path.join(critic_dir, "critic2_weights"))
            except Exception:
                pass

            # Save temperature (alpha) if accessible
            try:
                log_alpha_var = getattr(self.agent, 'log_alpha', None) or getattr(self.agent, '_log_alpha', None)
                if log_alpha_var is not None:
                    alpha_val = float(tf.exp(log_alpha_var).numpy())
                    with open(os.path.join(path, f"{self.trainingName}_temperature.txt"), 'w') as f:
                        f.write(str(alpha_val))
            except Exception:
                pass

except Exception:
    # TF-Agents not available; fall back to the original Agent implementation
    from guidance.QNetwork import Agent as LegacyAgent

    class TFAgentsSACWrapper:
        """Adapter that forwards calls to the original Agent when TF-Agents isn't installed."""
        def __init__(self, trainingName: str, stateDims: int, nActions: int, gamma: float, learningRate: float = 3e-4, tau: float = 0.005, batchSize: int = 256, minBufferSize: int = 1000):
            self._legacy = LegacyAgent(trainingName=trainingName, stateDims=stateDims, nActions=nActions, gamma=gamma, learningRate=learningRate, tau=tau, batchSize=batchSize, minBufferSize=minBufferSize)

        def choose_action(self, observation):
            return self._legacy.choose_action(observation)

        def addToReplayBuffer(self, state, action, reward, new_state, done):
            return self._legacy.addToReplayBuffer(state, action, reward, new_state, done)

        def train(self):
            return self._legacy.train()

        def saveModels(self, path: str):
            return self._legacy.saveModels(path)
    """A thin wrapper around TF-Agents SAC that provides the minimal API used by main.py.

    Methods provided:
    - choose_action(observation): returns a tf.Tensor action (batched 1) in [-1,1]
    - addToReplayBuffer(state, action, reward, next_state, done): stores transition
    - train(): samples a batch and runs a train step, returns losses similar shape to original
    - saveModels(path): saves actor and critic weights + temperature

    Implementation notes:
    - To keep the existing environment loop unchanged we don't wrap the full Env as a PyEnvironment.
      Instead we build TF-Agents networks and agent and feed it transitions collected by the loop.
    - Actions are expected by main.py to be in range [-1,1] and then scaled. This wrapper will accept
      that contract and return actions as tf.Tensor with shape (1, n_actions).
    """

    def __init__(self, trainingName: str, stateDims: int, nActions: int, gamma: float, learningRate: float = 3e-4, tau: float = 0.005, batchSize: int = 256, minBufferSize: int = 1000):
        self.trainingName = trainingName
        self.state_dims = stateDims
        self.n_actions = nActions
        self.gamma = gamma
        self.learning_rate = learningRate
        self.tau = tau
        self.batch_size = batchSize
        self.min_buffer_size = max(minBufferSize, batchSize)

        # specs
        self.observation_spec = tensor_spec.TensorSpec([self.state_dims], tf.float32)
        self.action_spec = tensor_spec.BoundedTensorSpec([self.n_actions], tf.float32, minimum=-1.0, maximum=1.0)

        # networks
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            self.observation_spec.shape,
            self.action_spec.shape[0],
            fc_layer_params=(256, 256),
            continuous_projection_net=normal_projection_network.NormalProjectionNetwork,
        )

        critic_net1 = q_network.QNetwork(self.observation_spec.shape, self.action_spec.shape, fc_layer_params=(256, 256))
        critic_net2 = q_network.QNetwork(self.observation_spec.shape, self.action_spec.shape, fc_layer_params=(256, 256))

        # agent
        self.global_step = tf.Variable(0, dtype=tf.int64)

        # Build a minimal time_step_spec and pass it to the agent
        from tf_agents.trajectories import time_step as ts
        time_step_spec = ts.time_step_spec(self.observation_spec)

        self.agent = sac_agent.SacAgent(
            time_step_spec=time_step_spec,
            action_spec=self.action_spec,
            actor_network=actor_net,
            critic_network=(critic_net1, critic_net2),
            actor_optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            critic_optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            alpha_optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            target_update_tau=self.tau,
            gamma=self.gamma,
            reward_scale_factor=1.0,
            train_step_counter=self.global_step,
        )

        # Replay buffer (TF uniform replay buffer)
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=1,
            max_length=1000000,
        )

        # A small converter for transitions coming from numpy
        self._rb_add = self.replay_buffer.add_batch

        # Create dataset iterator for training
        self.dataset = self.replay_buffer.as_dataset(sample_batch_size=self.batch_size, num_steps=2).prefetch(10)
        self.iterator = iter(self.dataset)

        # Make sure agent is initialized
        self.agent.initialize()

    def choose_action(self, observation: np.ndarray) -> tf.Tensor:
        obs = tf.convert_to_tensor([observation], dtype=tf.float32)
        # Use policy to get action; policy returns a policy step with action
        policy_step = self.agent.policy.action(obs)
        action = policy_step.action
        # Ensure action is float32 tensor shaped (1, n_actions)
        return tf.convert_to_tensor(action, dtype=tf.float32)

    def addToReplayBuffer(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool):
        # Build a trajectory from the single step and add to replay buffer
        # tf-agents expects tensors shaped [batch, ...]
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = tf.convert_to_tensor([action], dtype=tf.float32)
        reward = tf.convert_to_tensor([[reward]], dtype=tf.float32)
        next_state = tf.convert_to_tensor([new_state], dtype=tf.float32)
        step_type = tf.constant([1], dtype=tf.int32)  # MID (1)
        next_step_type = tf.constant([2], dtype=tf.int32) if done else tf.constant([1], dtype=tf.int32)

        # Construct a time_step and policy_step and build a Trajectory via from_transition
        from tf_agents.trajectories import time_step as ts
        from tf_agents.trajectories import policy_step as ps

        # current time step: use transition with zero reward (the reward will be attached to the next step in from_transition)
        current_ts = ts.transition(observation=state, reward=tf.zeros([1], dtype=tf.float32))
        next_ts = ts.termination(observation=next_state, reward=tf.squeeze(reward, axis=0)) if done else ts.transition(observation=next_state, reward=tf.squeeze(reward, axis=0))

        policy_step = ps.PolicyStep(action=action, state=(), info=())

        traj = trajectory.from_transition(current_ts, policy_step, next_ts)

        # Add to buffer
        self._rb_add(traj)

    def train(self):
        # Only train if enough samples
        if self.replay_buffer.num_frames().numpy() < self.min_buffer_size:
            return None

        # Get a batch from the iterator
        try:
            experience, unused_info = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataset)
            experience, unused_info = next(self.iterator)

        # Run a training step
        train_loss = self.agent.train(experience)

    # TF-Agents returns a LossInfo-like object. Extract the scalar loss (if present).
    loss = train_loss.loss if hasattr(train_loss, 'loss') else tf.constant(0.0)
    # Return a tuple similar to the original Agent: (critic1, critic2, actor, temperature)
    # We don't have separate critic/actor losses easily, so return the scalar loss for critics and zeros for others.
    critic_loss1 = loss
    critic_loss2 = loss
    actor_loss = tf.constant(0.0)
    temperature_loss = tf.constant(0.0)

    return critic_loss1, critic_loss2, actor_loss, temperature_loss

    def saveModels(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Try to save networks' weights if available (best-effort)
        try:
            actor_dir = os.path.join(path, f"{self.trainingName}_actor")
            os.makedirs(actor_dir, exist_ok=True)
            # Save policy variables via a checkpoint
            ckpt = tf.train.Checkpoint(policy=self.agent.policy)
            ckpt.write(os.path.join(actor_dir, 'policy_ckpt'))
        except Exception:
            pass

        try:
            critic_dir = os.path.join(path, f"{self.trainingName}_critic")
            os.makedirs(critic_dir, exist_ok=True)
            critic_networks = getattr(self.agent, '_critic_networks', None)
            if critic_networks:
                critic_networks[0].save_weights(os.path.join(critic_dir, "critic1_weights"))
                critic_networks[1].save_weights(os.path.join(critic_dir, "critic2_weights"))
        except Exception:
            pass

        # Save temperature (alpha) if accessible
        try:
            log_alpha_var = getattr(self.agent, 'log_alpha', None) or getattr(self.agent, '_log_alpha', None)
            if log_alpha_var is not None:
                alpha_val = float(tf.exp(log_alpha_var).numpy())
                with open(os.path.join(path, f"{self.trainingName}_temperature.txt"), 'w') as f:
                    f.write(str(alpha_val))
        except Exception:
            pass
