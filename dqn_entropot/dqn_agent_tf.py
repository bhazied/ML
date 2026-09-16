"""
Deep Q-Network (DQN) avec TensorFlow/Keras
Compatible avec WarehouseEnv (Gymnasium API).
"""

import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DQNNetwork(keras.Model):
    """Réseau de neurones qui estime une Q-value par action."""

    def __init__(self, action_size: int):
        super().__init__()
        self.dense_1 = layers.Dense(128, activation="relu")
        self.dense_2 = layers.Dense(128, activation="relu")
        self.output_layer = layers.Dense(action_size, activation="linear")

    def call(self, states, training=False):
        x = self.dense_1(states)
        x = self.dense_2(x)
        return self.output_layer(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        memory_size: int = 50_000,
        batch_size: int = 64,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.model = DQNNetwork(action_size)
        self.target_model = DQNNetwork(action_size)

        dummy_state = tf.zeros((1, state_size), dtype=tf.float32)
        self.model(dummy_state)
        self.target_model(dummy_state)
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.Huber()

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = tf.convert_to_tensor(np.asarray(state, dtype=np.float32)[None, :])
        q_values = self.model(state, training=False)
        return int(tf.argmax(q_values[0]).numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states = np.asarray([x[0] for x in batch], dtype=np.float32)
        actions = np.asarray([x[1] for x in batch], dtype=np.int32)
        rewards = np.asarray([x[2] for x in batch], dtype=np.float32)
        next_states = np.asarray([x[3] for x in batch], dtype=np.float32)
        terminated = np.asarray([x[4] for x in batch], dtype=np.float32)

        states = tf.convert_to_tensor(states)
        next_states = tf.convert_to_tensor(next_states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        terminated = tf.convert_to_tensor(terminated)

        next_q_values = self.target_model(next_states, training=False)
        targets = rewards + self.gamma * (1.0 - terminated) * tf.reduce_max(next_q_values, axis=1)

        with tf.GradientTape() as tape:
            all_q_values = self.model(states, training=True)
            action_mask = tf.one_hot(actions, self.action_size)
            predicted_q_values = tf.reduce_sum(all_q_values * action_mask, axis=1)
            loss = self.loss_fn(targets, predicted_q_values)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.numpy())

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, path="models/dqn_warehouse.keras"):
        self.model.save(path)


def train(env, agent, episodes=1_000, max_steps=250, target_update_every=20):
    history = {"rewards": [], "delivered": [], "losses": []}

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        losses = []

        for _ in range(max_steps):
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, terminated)
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)

            state = next_state
            episode_reward += reward
            if done:
                break

        if (episode + 1) % target_update_every == 0:
            agent.update_target_network()

        history["rewards"].append(episode_reward)
        history["delivered"].append(info.get("delivered", 0))
        history["losses"].append(float(np.mean(losses)) if losses else np.nan)

        if (episode + 1) % 25 == 0:
            avg_reward = np.mean(history["rewards"][-25:])
            avg_delivered = np.mean(history["delivered"][-25:])
            print(
                f"Episode {episode + 1:4d} | "
                f"reward moyen: {avg_reward:7.2f} | "
                f"articles moyens: {avg_delivered:.2f} | "
                f"epsilon: {agent.epsilon:.3f}"
            )

    return history
