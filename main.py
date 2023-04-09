import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random

class QNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.q = layers.Dense(action_dim, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q = self.q(x)
        return q

class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer_size=100000, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, 
                 learning_rate=0.001, batch_size=32, tau=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.set_weights(self.q_network.get_weights())

        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.q_network(state)
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.replay_buffer) >= self.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(experience)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.sample(self.batch_size)

        q_next = self.target_network(next_state_batch)
        target_q = reward_batch + (1 - done_batch) * self.gamma * tf.reduce_max(q_next, axis=1)

        with tf.GradientTape() as tape:
            q_values = self.q_network(state_batch, training=True)
            q_action = tf.reduce_sum(tf.one_hot(action_batch, self.action_dim) * q_values, axis=1)
            loss = tf.reduce_mean(tf.square(target_q - q_action))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.update_target_network()

    def sample(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        state_batch = np.array([experience[0] for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        next_state_batch = np.array([experience[3] for experience in batch])
        done_batch = np.array([experience[4] for experience in batch])
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
