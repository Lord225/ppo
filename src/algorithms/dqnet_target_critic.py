from typing import Callable, List, Tuple
import tensorflow as tf
import tensorboard as tb

from common import ReplayHistoryType

loss_fn = tf.keras.losses.mean_squared_error

@tf.function
def training_step_dqnet_target_critic(
        batch: ReplayHistoryType,
        discount_rate: float,
        target_model: tf.keras.Model,
        actor_model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        n_outputs: int,
        step: int,
        iters_per_episode: int,
        mini_batch_size: int,
):
    """
    Training step that uses:
    - DQN target network
    - Critic network & cost
    - Double DQN
    """
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch

    actor_loss_sum = tf.cast(0.0, dtype=tf.float32)
    critic_loss_sum = tf.cast(0.0, dtype=tf.float32)
    loss_sum = tf.cast(0.0, dtype=tf.float32)


    for _ in tf.range(iters_per_episode):
        # sample data
        idx = tf.random.uniform([mini_batch_size], minval=0, maxval=len(batch.states), dtype=tf.int64)
        
        states = tf.gather(batch_states, idx)
        actions = tf.gather(batch_actions, idx)
        rewards = tf.gather(batch_rewards, idx)
        next_states = tf.gather(batch_next_states, idx)
        dones = tf.gather(batch_dones, idx)
        
        next_Q_values, _ = actor_model(next_states, training=True) # type: ignore
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, n_outputs)
        next_best_Q, _ = target_model(next_states, training=True) # type: ignore 
        next_best_Q_values = tf.reduce_sum(next_best_Q * next_mask, axis=1)
        target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - tf.cast(dones, dtype=tf.float32)) * discount_rate * next_best_Q_values)
        target_Q_values = tf.reshape(target_Q_values, [-1, 1])
        mask = tf.one_hot(actions, n_outputs)

        with tf.GradientTape() as tape:
            all_Q_values, values = actor_model(states, training=True) # type: ignore
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            actor_loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
            critic_loss = tf.reduce_mean(loss_fn(target_Q_values, values))

            loss = actor_loss + critic_loss
        
        actor_loss_sum += actor_loss
        critic_loss_sum += critic_loss
        loss_sum += loss
        
        grads = tape.gradient(loss, actor_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))
    
    tf.summary.scalar('actor_loss', actor_loss_sum / iters_per_episode, step=step) # type: ignore
    tf.summary.scalar('critic_loss', critic_loss_sum / iters_per_episode, step=step) # type: ignore
    tf.summary.scalar('loss', loss_sum / iters_per_episode, step=step) # type: ignore

        