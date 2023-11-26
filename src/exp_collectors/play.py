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


@tf.function
def run_episode_selfplay(
    observation,
    mask,
    player1: tf.keras.Model,
    player2: tf.keras.Model,
    critic: tf.keras.Model,
    tf_env_step: Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]],
    max_steps: int,
    env_actions: int,
    epsilon: float,      
):
    # prepare buffers (numpy)
    states_p1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions_p1 = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards_p1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values_p1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    log_probs_p1 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    states_p2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions_p2 = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards_p2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values_p2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    log_probs_p2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    state = observation
    next_iteration_is_done = False

    initial_state_shape = state.shape
    initial_mask_shape = mask.shape

    index_p1 = 0
    index_p2 = 0

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        current_player = t % 2
        
        # Run the model and to get action probabilities and critic value
        if current_player == 0:
            action_logits_t = player1(state)
        else:
            action_logits_t = player2(state)

        value_t = critic(state)

        masked_action_logits_t = tf.where(tf.cast(mask, tf.bool), action_logits_t, -1e9)
        masked_action_random = tf.where(tf.cast(mask, tf.bool), tf.random.uniform([1, env_actions]), -1e9)

        # epsilon greedy
        action = tf.squeeze(tf.where(
                tf.random.uniform([1]) < tf.cast(epsilon, tf.float32),
                # Random action, use mask to make sure it is legal
                tf.cast(tf.squeeze(tf.random.categorical(masked_action_random, 1), axis=1), tf.int32)[0], # type: ignore
                # argmax action, use mask to make sure it is legal
                tf.cast(tf.squeeze(tf.argmax(masked_action_logits_t, axis=1)), tf.int32), # type: ignore
            ))
        
        # action = tf.squeeze(tf.random.categorical(masked_action_logits_t, 1), axis=1) 
        # action = tf.cast(action, tf.int32)
        # action = tf.squeeze(action)

        next_state, action_mask, reward, done = tf_env_step(action, next_iteration_is_done) # type: ignore
        next_state.set_shape(initial_state_shape)
        action_mask.set_shape(initial_mask_shape)

        log_prob = logprobabilities(action_logits_t, action, env_actions)

        # store results
        if current_player == 0:
            states_p1 = states_p1.write(index_p1, tf.squeeze(state))
            actions_p1 = actions_p1.write(index_p1, action)
            rewards_p1 = rewards_p1.write(index_p1, reward)
            values_p1 = values_p1.write(index_p1, tf.squeeze(value_t))
            log_probs_p1 = log_probs_p1.write(index_p1, tf.squeeze(log_prob))
            index_p1 += 1
        else:
            states_p2 = states_p2.write(index_p2, tf.squeeze(state))
            actions_p2 = actions_p2.write(index_p2, action)
            rewards_p2 = rewards_p2.write(index_p2, reward)
            values_p2 = values_p2.write(index_p2, tf.squeeze(value_t))
            log_probs_p2 = log_probs_p2.write(index_p2, tf.squeeze(log_prob))
            index_p2 += 1

        state = next_state
        mask = action_mask

        if next_iteration_is_done:
            break
    
        if tf.cast(done, tf.bool): # type: ignore
            next_iteration_is_done = True

    states_p1 = states_p1.stack()
    actions_p1 = actions_p1.stack()
    rewards_p1 = rewards_p1.stack()
    values_p1 = values_p1.stack()
    log_probs_p1 = log_probs_p1.stack()

    states_p2 = states_p2.stack()
    actions_p2 = actions_p2.stack()
    rewards_p2 = rewards_p2.stack()
    values_p2 = values_p2.stack()
    log_probs_p2 = log_probs_p2.stack()

    return (states_p1, states_p2, actions_p1, actions_p2, rewards_p1, rewards_p2, values_p1, values_p2, log_probs_p1, log_probs_p2)

# run one episode of selfplay and determine the winner
def evaluate_selfplay(
    observation,
    mask,
    player1: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    player2: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    critic: tf.keras.Model,
    tf_env_step: Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]],
    env_actions: int,
    max_steps: int
    ):

    state = observation
    next_iteration_is_done = False

    initial_state_shape = state.shape
    initial_mask_shape = mask.shape

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        current_player = t % 2
        # current_player_tensor = tf.reshape(current_player, [1])
        
        # Run the model and to get action probabilities and critic value
        if current_player == 0:
            action_logits_t = player1(state) # type: ignore
        else:
            action_logits_t = player2(state) # type: ignore

        # value_t = critic(state)

        masked_action_logits_t = tf.where(tf.cast(mask, tf.bool), action_logits_t, -1e9)

        # epsilon greedy
        action = int(tf.cast(tf.squeeze(tf.argmax(masked_action_logits_t, axis=1)), tf.int32)) # type: ignore

        next_state, action_mask, reward, done = tf_env_step(action, next_iteration_is_done) # type: ignore

        next_state.set_shape(initial_state_shape)
        action_mask.set_shape(initial_mask_shape)

        state = next_state
        mask = action_mask

        if next_iteration_is_done:
            # determine winner
            if current_player == 0:
                return 0, int(t)
            else:
                return 1, int(t)
            
        if tf.cast(done, tf.bool): # type: ignore
            next_iteration_is_done = True



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

