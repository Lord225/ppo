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
        dones = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps, dtype=tf.int32):
            # Convert state into a batched tensor (batch size = 1)
            states = states.write(t, state)

            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, _ = actor_model(state) # type: ignore

            action_probs_t = tf.nn.softmax(action_logits_t)

            action = tf.cast( 
            tf.squeeze(tf.where(
                tf.random.uniform([1]) < tf.cast(epsilon, tf.float32),
                # Random int, 0-4096
                tf.random.uniform([1], minval=0, maxval=env_actions, dtype=tf.int64),
                # argmax action
                tf.cast(tf.squeeze(tf.random.categorical(action_probs_t, 1), axis=1), tf.int64)[0], # type: ignore
            )), dtype=tf.int32)

            
            

        

            actions = actions.write(t, action)

            # Apply action to the environment to get next state and reward
            state, reward, done = tf_env_step(action) # type: ignore
            state.set_shape(initial_state_shape)

            next_states = next_states.write(t, state)

            dones = dones.write(t, tf.cast(done, tf.float32))

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
            actor: tf.keras.Model,
            critic: tf.keras.Model,
            max_steps: int,
            env_actions: int,
            ):
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps, dtype=tf.int32):

            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t = actor(state) # type: ignore

            value_t = critic(state) # type: ignore

            action = tf.squeeze(tf.random.categorical(action_logits_t, 1), axis=1)
            action = tf.cast(action, tf.int32)
            action = tf.squeeze(action)

            next_state, reward, done = tf_env_step(action)

            next_state.set_shape(initial_state_shape)
            
            log_prob = logprobabilities(action_logits_t, action, env_actions)
            
            # store results
            states = states.write(t, tf.squeeze(state))
            actions = actions.write(t, action)
            rewards = rewards.write(t, reward)
            values = values.write(t, tf.squeeze(value_t))
            log_probs = log_probs.write(t, tf.squeeze(log_prob))

            state = next_state

            if tf.cast(done, tf.bool): # type: ignore
                break

        states = states.stack()
        actions = actions.stack()
        rewards = rewards.stack()
        values = values.stack()
        log_probs = log_probs.stack()

        return PPOReplayHistoryType(states, actions, rewards, values, log_probs), tf.reduce_sum(rewards)
                
    return run_episode




def get_curius_ppo_runner(tf_env_step: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]):
    @tf.function(reduce_retracing=True)
    def run_episode(
            initial_state: tf.Tensor,
            actor: tf.keras.Model,
            critic: tf.keras.Model,
            curiosity: tf.keras.Model,
            max_steps: int,
            env_actions: int,
            curius_coef: float,
            ):
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        next_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        curiosities = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps, dtype=tf.int32):

            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t = actor(state) # type: ignore

            value_t = critic(state) # type: ignore

            action = tf.squeeze(tf.random.categorical(action_logits_t, 1), axis=1)
            action = tf.cast(action, tf.int32)
            action = tf.squeeze(action)

            next_state, reward, done = tf_env_step(action)

            next_state.set_shape(initial_state_shape)
            
            # calculate curiosity
            curiosity_reward = curiosity(state, action)
            curiosity_reward = tf.squeeze(curiosity_reward)
            curiosity_reward = tf.cast(curiosity_reward, tf.float32)

            reward = reward + curius_coef * curiosity_reward # type: ignore
            
            log_prob = logprobabilities(action_logits_t, action, env_actions)
            
            # store results
            curiosities = curiosities.write(t, curiosity_reward)
            states = states.write(t, state)
            next_states = next_states.write(t, next_state)
            actions = actions.write(t, action)
            rewards = rewards.write(t, reward)
            values = values.write(t, tf.squeeze(value_t))
            log_probs = log_probs.write(t, tf.squeeze(log_prob))

            state = next_state

            if tf.cast(done, tf.bool): # type: ignore
                break

        states = states.stack()
        actions = actions.stack()
        rewards = rewards.stack()
        values = values.stack()
        log_probs = log_probs.stack()
        next_states = next_states.stack()

        return PPOReplayHistoryType(states, actions, rewards, values, log_probs), tf.reduce_sum(rewards), tf.reduce_sum(curiosities)
                
    return run_episode

