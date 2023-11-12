from collections import deque
import tensorflow as tf
import tensorboard
import os
import argparse

import tqdm
from algorithms.dqnet_target_critic import training_step_dqnet_target_critic
from common import splash_screen
import config


parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="resume from a model") 

args = parser.parse_args()


import enviroments

env = enviroments.get_packman_stack_frames()

params = argparse.Namespace()

params.env_name = env.spec.id
params.version = "v1.5"
params.DRY_RUN = False


params.lr = 0.0002
params.action_space = env.action_space.n # type: ignore
params.observation_space_raw = env.observation_space.shape
params.observation_space = (85, 50, 3*2)

params.episodes = 100000
params.max_steps_per_episode = 1000

params.discount_rate = 0.8
params.eps_decay_len = 4000
params.eps_min = 0.1

params.batch_size = 2048
params.iters_per_episode = 20
params.mini_batch_size = 128
params.train_interval = 1
params.target_update_freq = 150
params.save_freq = 250
if args.resume is not None:
    params.resumed = 'resumed from: ' + os.path.basename(args.resume)

splash_screen(params)

def get_model() -> tf.keras.Model:
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume)
        print("loaded model from", args.resume)
        return model # type: ignore
    

    inputs = tf.keras.Input(shape=params.observation_space)

    x = tf.keras.layers.Conv2D(32, 4, strides=2, activation="elu")(inputs)
    x = tf.keras.layers.Conv2D(64, 2, strides=2, activation="elu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="elu")(x)
    x = tf.keras.layers.Conv2D(128, 2, strides=1, activation="elu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="elu")(x)
    x = tf.keras.layers.Dense(256, activation="elu")(x)

    outputs = tf.keras.layers.Dense(params.action_space)(x)
    critic = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=[outputs, critic])

    return model 

actor_model = get_model()
target_model = get_model()
target_model.set_weights(actor_model.get_weights())


from memory import ReplayMemory


optimizer = tf.keras.optimizers.Adam(params.lr)

from exp_collectors.play import get_episode_runner 
import time

def run():
    running_avg = deque(maxlen=200)

    memory = ReplayMemory(10_000, params.observation_space)

    env_step = enviroments.make_tensorflow_env_step(env, lambda x: enviroments.pacman_transform_observation_stack(x, target_size=params.observation_space[:2]))
    env_reset = enviroments.make_tensorflow_env_reset(env, lambda x: enviroments.pacman_transform_observation_stack(x, target_size=params.observation_space[:2]))

    runner = get_episode_runner(env_step)
    runner = tf.function(runner)

    action_space = tf.constant(params.action_space, dtype=tf.int32)
    discount_rate = tf.constant(params.discount_rate, dtype=tf.float32)

    iters_per_episode = tf.constant(params.iters_per_episode, dtype=tf.float32)
    mini_batch_size = tf.constant(params.mini_batch_size, dtype=tf.int64)
    max_steps_per_episode = tf.constant(params.max_steps_per_episode, dtype=tf.int64)


    t = tqdm.tqdm(range(params.episodes))
    for episode in t:
        epsilon = max(1 - episode / params.eps_decay_len, params.eps_min)

        # run episode
        state = env_reset()
        state = tf.constant(state, dtype=tf.float32)

        epsilon = tf.constant(epsilon, dtype=tf.float32)

        (states, action_probs, returns, next_states, dones), total_rewards = runner(state,
                                                                                    actor_model,
                                                                                    max_steps_per_episode,
                                                                                    epsilon,
                                                                                    tf.constant(params.action_space, dtype=tf.int64),
                                                                                )  # type: ignore

        # time this line of code

        memory.add(states, action_probs, returns, next_states, dones)
        running_avg.append(total_rewards)
        avg = sum(running_avg)/len(running_avg)

        # log
        tf.summary.scalar('reward', total_rewards, step=episode)
        tf.summary.scalar('reward_avg', avg, step=episode)
        tf.summary.scalar('lenght', states.shape[0], step=episode)

        t.set_description(f"Reward: {total_rewards:.2f} - Avg: {avg:.2f} - Iterations: {states.shape[0]}")


        #train
        if len(memory) > params.batch_size+1 and episode % params.train_interval == 0:
            batch = memory.sample(params.batch_size)

            episode = tf.constant(episode, dtype=tf.int64)
    
            training_step_dqnet_target_critic(batch, 
                                              discount_rate, 
                                              target_model, 
                                              actor_model, 
                                              optimizer,
                                              action_space,
                                              episode,
                                              iters_per_episode,
                                              mini_batch_size,
                                              )
            
        # update target network
        if episode % params.target_update_freq == 0:
            target_model.set_weights(actor_model.get_weights())

        # save model
        if episode % params.save_freq == 0 and episode > 0:
            actor_model.save(f"{config.MODELS_DIR}{params.env_name}{params.version}_{config.RUN_NAME}_{episode}.h5")



run()