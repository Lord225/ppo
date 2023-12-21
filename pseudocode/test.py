from re import M
from typing import Callable, Tuple
import tensorflow as tf

EnvStep= Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]

def run_episode(initial_state: tf.Tensor, model: tf.keras.Model, max_steps: tf.Tensor, make_step: EnvStep):
    """
    Run a single episode of the environment using the given model.
    """
    states = tf.TensorArray(dtype=tf.float32)
    actions = tf.TensorArray(dtype=tf.int32)
    rewards = tf.TensorArray(dtype=tf.float32)
    next_states = tf.TensorArray(dtype=tf.float32)
    dones = tf.TensorArray(dtype=tf.float32)

    state = initial_state

    for t in tf.range(max_steps, dtype=tf.int32):
        states = states.write(t, state)

        action_logits_t = model(state)
        action = sample_action(action_logits_t)

        next_state, reward, done = make_step(action)

        actions = actions.write(t, action)
        rewards = rewards.write(t, reward)
        next_states = next_states.write(t, next_state)
        dones = dones.write(t, done)

        state = next_state

        if done:
            break

    return (states.stack(), actions.stack(), rewards.stack(), next_states.stack(), dones.stack())



@tf.function
def discounted_cumulative_sums_tf(x, discount_rate)-> tf.Tensor:
    size = tf.shape(x)[0]
    x = tf.reverse(x, axis=[0])
    buffer = tf.TensorArray(dtype=tf.float32, size=size)

    discounted_sum = tf.constant(0.0, dtype=tf.float32)

    for i in tf.range(size):
        discounted_sum = x[i] + discount_rate * discounted_sum
        buffer = buffer.write(i, discounted_sum)
    
    return tf.reverse(buffer.stack(), axis=[0])


def add_trajectory_to_memory(states, actions, rewards, values, logprobabilities):
    
    # code omitted
    
    # finish trajectory (assume last reward is 0 instead of V(s_T))
    rewards = tf.concat([rewards, [0.0]], axis=0)
    values = tf.concat([values, [0.0]], axis=0)

    # calucalte deltas between actual rewards and estimated values from critic
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

    advantages = discounted_cumulative_sums_tf(
        deltas, gamma * lam
    )
    returns = discounted_cumulative_sums_tf(
        rewards, gamma
    )[:-1]

    # code omitted 

@tf.function
def training_step_ppo(batch,
                      actor,
                      num_of_actions,
                      clip_ratio,
                      optimizer: tf.keras.optimizers.Optimizer,
                      ):
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer = batch
    with tf.GradientTape() as tape:
        logits = actor(observation_buffer)

        ratio = tf.exp(
            logprobabilities(logits, action_buffer, num_of_actions)
            - logprobability_buffer
        )
        
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))





# Run the model and to get action probabilities and critic value
action_logits_t = actor(state)

# sample an action from the action probability distribution
action = tf.squeeze(tf.random.categorical(action_logits_t, 1), axis=1)
action = tf.cast(action, tf.int32)
action = tf.squeeze(action)

# make a step in the environment
next_state, reward, done = tf_env_step(action)

next_state.set_shape(initial_state_shape)

action_one_hot = tf.one_hot(action, env_actions)
action_input_processed = tf.expand_dims(action_one_hot, axis=0)

# calculate curiosity
encoded_state = encoder(state)
encoded_next_state = encoder(tf.expand_dims(next_state, 0))
encoded_predicted_state = curiosity([encoded_state, action_input_processed])
encoded_predicted_state = tf.squeeze(encoded_predicted_state, axis=0)

curiosity_reward = tf.reduce_sum(tf.square(encoded_next_state - encoded_predicted_state))

# add curiosity reward to the reward
reward = reward + curius_coef * curiosity_reward



@tf.function
def sample_function():
    buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    for i in tf.range(10):
        buffer = buffer.write(i, i)
    
    return buffer.stack()


def step(action):
    state, reward, done, _, _ = env.step(int(action))
    state = observation_transformer(state)
    return (np.array(state, np.float32), np.array(reward, np.float32), np.array(done, np.int32))

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32)])
def tf_env_step(action):
    return tf.numpy_function(step, [action], (tf.float32, tf.float32, tf.int32))





parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="resume from a model") 

env = enviroments.get_packman_stack_frames()

params = argparse.Namespace()

params.env_name = env.spec.id
params.version = "v3.0"
params.DRY_RUN = False

params.actor_lr  = 3e-8
params.critic_lr = 1e-5
# define other parameters

# init everything

def run():
    memory = PPOReplayMemory(20_000, params.observation_space)

    env_step = enviroments.make_tensorflow_env_step(env, lambda x: enviroments.pacman_transform_observation_stack_big(x))
    env_reset = enviroments.make_tensorflow_env_reset(env, lambda x: enviroments.pacman_transform_observation_stack_big(x))

    runner = get_ppo_runner(env_step)
    runner = tf.function(runner)

    t = tqdm.tqdm(range(params.episodes))
    for episode in t:
        initial_state = env_reset()

        (states, actions, rewards, values, log_probs), total_rewards = runner(initial_state, 
                                                                              actor, critic, 
                                                                              max_steps_per_episode, 
                                                                              action_space)

        memory.add(states, actions, rewards, values, log_probs)
        
        if len(memory) >= params.batch_size and int(episode) % params.train_interval == 0:
            batch = memory.sample(batch_size)
            
            training_step_ppo(batch, actor, action_space, clip_ratio, policy_optimizer, episode)
            
            # other optimization steps
                
        if episode % params.save_freq == 0 and episode > 0: 
            save_model(episode, actor, critic, params)
run()



def training_step_DQN(batch,model, num_of_actions, discount_rate, optimizer):
    states, actions, rewards, next_states, dones = batch

    # calculate target Q values
    next_Q_values = model(next_states)
    max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
    # get target Q values, special case for last step (done))
    target_Q_values = rewards + (1 - dones) * discount_rate * max_next_Q_values

    mask = tf.one_hot(actions, num_of_actions)

    with tf.GradientTape() as tape:
        # get current Q values from model
        all_Q_values = model(states)
        # select only the Q values for the actions that were taken by agent
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        # get the loss
        loss = mse(target_Q_values, Q_values)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))