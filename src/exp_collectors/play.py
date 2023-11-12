from typing import Callable, Tuple
import tensorflow as tf
from algorithms.proximal_policy_optimalization import logprobabilities

from common import PPOReplayHistoryType, ReplayHistoryType


def get_episode_runner(tf_env_step: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]):
    @tf.function(reduce_retracing=True)
    def run_episode(
            initial_state: tf.Tensor,
            actor_model: tf.keras.Model,
            max_steps: tf.Tensor,
            epsilon: tf.Tensor,
            env_actions: tf.Tensor,
            ) -> Tuple[ReplayHistoryType, tf.Tensor]:
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        next_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        dones = tf.TensorArray(dtype=tf.bool, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps, dtype=tf.int32):
            # Convert state into a batched tensor (batch size = 1)
            states = states.write(t, state)

            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, _ = actor_model(state) # type: ignore

            action_probs_t = tf.nn.softmax(action_logits_t)

            # action = tf.cast( 
            # tf.squeeze(tf.where(
            #     tf.random.uniform([1]) < tf.cast(epsilon, tf.float32),
            #     # Random int, 0-4096
            #     tf.random.uniform([1], minval=0, maxval=env_actions, dtype=tf.int64),
            #     # argmax action
            #     tf.argmax(action_probs_t, axis=1)[0],
            # )), dtype=tf.int32)

            action = tf.squeeze(tf.random.categorical(action_probs_t, 1), axis=1)
            action = tf.cast(action, tf.int32)
            action = tf.squeeze(action)

            actions = actions.write(t, action)

            # Apply action to the environment to get next state and reward
            state, reward, done = tf_env_step(action) # type: ignore
            state.set_shape(initial_state_shape)

            next_states = next_states.write(t, state)

            dones = dones.write(t, tf.cast(done, tf.bool))

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool): # type: ignore
                break

        states = states.stack()
        actions = actions.stack()
        rewards = rewards.stack()
        next_states = next_states.stack()
        dones = dones.stack()

        return ReplayHistoryType(states, actions, rewards, next_states, dones), tf.reduce_sum(rewards) # type: ignore
    
    return run_episode

def get_ppo_runner(tf_env_step: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]):
    @tf.function(reduce_retracing=True)
    def run_episode(
            initial_state: tf.Tensor,
            actor_model: tf.keras.Model, 
            max_steps: int,
            env_actions: int,
            ):
        
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        dones = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps, dtype=tf.int32):

            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value_t = actor_model(state) # type: ignore

            action_probs_t = tf.nn.softmax(action_logits_t)

            action = tf.squeeze(tf.random.categorical(action_probs_t, 1), axis=1)
            action = tf.cast(action, tf.int32)
            action = tf.squeeze(action)

            state, reward, done = tf_env_step(action)

            state.set_shape(initial_state_shape)
            
            log_prob = logprobabilities(action_logits_t, action, env_actions)

            # store results
            states = states.write(t, state)
            actions = actions.write(t, action)
            rewards = rewards.write(t, reward)
            values = values.write(t, tf.squeeze(value_t))
            log_probs = log_probs.write(t, tf.squeeze(log_prob))
            dones = dones.write(t, tf.cast(done, tf.float32))

            if tf.cast(done, tf.bool): # type: ignore
                break

        states = states.stack()
        actions = actions.stack()
        rewards = rewards.stack()
        values = values.stack()
        log_probs = log_probs.stack()
        dones = dones.stack()

        return PPOReplayHistoryType(states, actions, rewards, values, log_probs, dones), tf.reduce_sum(rewards)
                
    return run_episode