# Versão simplificada e compatível do agente DDPG
import numpy as np
from task import Task
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers
import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)

class Actor:
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.build_model()

    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        
        # Hidden layers
        net = layers.Dense(64, activation='relu')(states)
        net = layers.Dense(128, activation='relu')(net)
        net = layers.Dense(64, activation='relu')(net)
        
        # Output layer - scaled to action range
        raw_actions = layers.Dense(self.action_size, activation='sigmoid')(net)
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)
        
        self.model = models.Model(inputs=states, outputs=actions)
        
        # Define optimizer and loss
        self.optimizer = optimizers.Adam(learning_rate=0.001)
        
        # Custom training step
        @tf.function
        def train_step(states, action_gradients):
            with tf.GradientTape() as tape:
                actions = self.model(states, training=True)
                loss = -tf.reduce_mean(action_gradients * actions)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss
        
        self.train_step = train_step

class Critic:
    """Critic (Value) Model."""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        # State pathway
        net_states = layers.Dense(64, activation='relu')(states)
        net_states = layers.Dense(128, activation='relu')(net_states)
        
        # Action pathway
        net_actions = layers.Dense(64, activation='relu')(actions)
        net_actions = layers.Dense(128, activation='relu')(net_actions)
        
        # Combine pathways
        net = layers.Concatenate()([net_states, net_actions])
        net = layers.Dense(128, activation='relu')(net)
        
        # Output layer
        Q_values = layers.Dense(1, name='q_values')(net)
        
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""
    
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class DDPG:
    """Reinforcement Learning agent that learns using DDPG."""
    
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor and Critic models
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target models
        self.update_target_models(tau=1.0)

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size=100000, batch_size=64)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01    # soft update parameter

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state, verbose=0)[0]
        return list(action + self.noise.sample())

    def step(self, action, reward, next_state, done):
        """Save experience and learn."""
        self.memory.add(self.last_state, action, reward, next_state, done)

        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        self.last_state = next_state

    def learn(self, experiences):
        """Update policy and value parameters using batch of experience tuples."""
        states = np.vstack([e.state for e in experiences])
        actions = np.array([e.action for e in experiences]).astype(np.float32)
        rewards = np.array([e.reward for e in experiences]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences])

        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target.model.predict(next_states, verbose=0)
        Q_targets_next = self.critic_target.model.predict([next_states, next_actions], verbose=0)

        # Compute Q targets for current states
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

        # Train critic model
        self.critic_local.model.fit([states, actions], Q_targets, 
                                   epochs=1, batch_size=len(experiences), 
                                   verbose=0)

        # Train actor model
        action_gradients = self._get_action_gradients(states, actions)
        self.actor_local.train_step(states, action_gradients)

        # Soft update target models
        self.update_target_models()

    def _get_action_gradients(self, states, actions):
        """Get action gradients from critic."""
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(actions_tensor)
            Q_values = self.critic_local.model([states_tensor, actions_tensor])
        
        action_gradients = tape.gradient(Q_values, actions_tensor)
        return action_gradients

    def update_target_models(self, tau=None):
        """Soft update of target model parameters."""
        if tau is None:
            tau = self.tau
            
        local_actor_weights = self.actor_local.model.get_weights()
        target_actor_weights = self.actor_target.model.get_weights()
        
        local_critic_weights = self.critic_local.model.get_weights()
        target_critic_weights = self.critic_target.model.get_weights()
        
        # Update actor target
        for i in range(len(local_actor_weights)):
            target_actor_weights[i] = tau * local_actor_weights[i] + (1 - tau) * target_actor_weights[i]
        self.actor_target.model.set_weights(target_actor_weights)
        
        # Update critic target
        for i in range(len(local_critic_weights)):
            target_critic_weights[i] = tau * local_critic_weights[i] + (1 - tau) * target_critic_weights[i]
        self.critic_target.model.set_weights(target_critic_weights)
