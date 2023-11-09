from typing import Callable, List, Tuple
import tensorflow as tf

from common import ReplayHistoryType

loss_fn = tf.keras.losses.mean_squared_error

@tf.function
def training_step_dqnet_target_critic(
        batch: ReplayHistoryType,
        discount_rate: float,
        target_model: tf.keras.Model,
        actor_model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        n_outputs: int
):
    """
    Training step that uses:
    - DQN target network
    - Critic network & cost
    - Double DQN
    """
    states, actions, rewards, next_states, dones = batch

    next_Q_values, _ = actor_model(next_states, training=True) # type: ignore
    best_next_actions = tf.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs)
    next_best_Q, _ = target_model(next_states, training=True) # type: ignore 
    next_best_Q_values = tf.reduce_sum(next_best_Q * next_mask, axis=1)
    target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * next_best_Q_values)
    target_Q_values = tf.reshape(target_Q_values, [-1, 1])
    mask = tf.one_hot(actions, n_outputs)

    with tf.GradientTape() as tape:
        all_Q_values, values = actor_model(states, training=True) # type: ignore
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        actor_loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        critic_loss = tf.reduce_mean(loss_fn(target_Q_values, values))

        loss = actor_loss + critic_loss
    grads = tape.gradient(loss, actor_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))