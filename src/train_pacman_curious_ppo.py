from collections import deque
import tensorflow as tf
import tensorboard
import os
import argparse
import tqdm
from algorithms.proximal_policy_optimalization import training_step_critic, training_step_curiosty, training_step_ppo
from common import splash_screen
import config
import enviroments
import os
import numpy as np
from tensorboard.plugins.hparams import api as hp

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="resume from a model") 

args = parser.parse_args()


env = enviroments.get_pacman_stack_frames_big()

params = argparse.Namespace()

params.env_name = env.spec.id
params.version = "v5.0"
params.DRY_RUN = False

params.actor_lr  = 1e-6
params.critic_lr = 1e-4

params.action_space = env.action_space.n # type: ignore
params.observation_space_raw = env.observation_space.shape
params.observation_space = (85, 50, 6)

params.episodes = 100000
params.max_steps_per_episode = 1200

params.discount_rate = 0.99

params.eps_decay_len = 1000
params.eps_min = 0.1

params.clip_ratio = 0.20
params.lam = 0.98


# params.curius_coef = 0.013
params.curius_coef = 0.08

params.batch_size = 4000
params.batch_size_curius = 300

params.train_interval = 1
params.iters = 100


params.save_freq = 500
if args.resume is not None:
    params.resumed = 'resumed from: ' + os.path.basename(args.resume)

splash_screen(params)

def get_actor():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'.actor.h5')
        print("loaded model from", args.resume)
        return model

    observation_input = tf.keras.Input(shape=params.observation_space, dtype=tf.float32)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(observation_input)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.elu)(x)

    logits = tf.keras.layers.Dense(params.action_space)(x)
    return tf.keras.Model(inputs=observation_input, outputs=logits)

def get_critic():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'.critic.h5')
        print("loaded model from", args.resume)
        return model

    observation_input = tf.keras.Input(shape=params.observation_space, dtype=tf.float32)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(observation_input)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.elu)(x)


    value = tf.squeeze(tf.keras.layers.Dense(1)(x))
    return tf.keras.Model(inputs=observation_input, outputs=value)

def get_curiosity():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'.curiosity.h5')
        print("loaded model from", args.resume)
        return model
    
    observation_input = tf.keras.Input(shape=params.observation_space, dtype=tf.float32)
    action_input = tf.keras.Input(shape=params.action_space, dtype=tf.float32)

    # convoluted nn that will predict next state
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation=tf.nn.elu)(observation_input)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=3, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, action_input])
    x = tf.keras.layers.Dense(512, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Reshape((1, 1, 512))(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation=tf.nn.elu)(x) 
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation=tf.nn.elu)(x) # 31, 31, 32
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation=tf.nn.elu)(x) # 63, 63, 6
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation=tf.nn.elu)(x) # 127, 127, 6
    x = tf.keras.layers.Cropping2D(((21, 21), (39, 38)))(x)
    # concat with observation
    x = tf.keras.layers.Concatenate()([x, observation_input])
    x = tf.keras.layers.Conv2D(32, 3, strides=1, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(16, 3, strides=1, activation=tf.nn.elu)(x)
    x = tf.keras.layers.Conv2D(6, 3, strides=1, activation=tf.nn.elu)(x)

    return tf.keras.Model(inputs=[observation_input, action_input], outputs=x)


actor = get_actor()
critic = get_critic()
curiosity = get_curiosity()


policy_optimizer = tf.keras.optimizers.Adam(learning_rate=params.actor_lr)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=params.critic_lr)
curiosity_optimizer = tf.keras.optimizers.Adam(learning_rate=params.critic_lr)

from memory import PPOReplayMemory
from exp_collectors.play import get_curius_ppo_runner

def log_stats(stats, step):
    # list of kl, policy_loss, mean_ratio, mean_clipped_ratio, mean_advantage, mean_logprob
    tf.summary.scalar('kl', np.mean([x[0] for x in stats]), step=step)
    tf.summary.scalar('loss', np.mean([x[1] for x in stats]), step=step)
    tf.summary.scalar('mean_ratio', np.mean([x[2] for x in stats]), step=step)
    tf.summary.scalar('mean_clipped_ratio', np.mean([x[3] for x in stats]), step=step)
    tf.summary.scalar('mean_logprob', np.mean([x[5] for x in stats]), step=step)


def run():
    running_avg = deque(maxlen=200)

    memory = PPOReplayMemory(15_000, params.observation_space, gamma=params.discount_rate, lam=params.lam, gather_next_states=True)

    env_step = enviroments.make_tensorflow_env_step(env, lambda x: enviroments.pacman_transform_grayscale_observation_stack_big(x))
    env_reset = enviroments.make_tensorflow_env_reset(env, lambda x: enviroments.pacman_transform_grayscale_observation_stack_big(x))

    runner = get_curius_ppo_runner(env_step)
    runner = tf.function(runner)

    action_space = tf.constant(params.action_space, dtype=tf.int32)
    max_steps_per_episode = tf.constant(params.max_steps_per_episode, dtype=tf.int64)
    batch_size = tf.constant(params.batch_size, dtype=tf.int64)
    batch_size_curius = tf.constant(params.batch_size_curius, dtype=tf.int64)
    clip_ratio = tf.constant(params.clip_ratio, dtype=tf.float32)

    t = tqdm.tqdm(range(params.episodes))
    for episode in t:
        initial_state = env_reset()

        for _ in range(50):
            initial_state, _, _ = env_step(0) # type: ignore

        initial_state = tf.constant(initial_state, dtype=tf.float32)
        
        curius_coef = tf.constant(params.curius_coef, dtype=tf.float32)

        (states, actions, rewards, values, log_probs, next_states), total_rewards, curiosity_mean, curiosity_std = runner(initial_state, actor, critic, curiosity, max_steps_per_episode, action_space, curius_coef) # type: ignore
        
        memory.add(states, actions, rewards, values, log_probs, next_states)
        
        curiosity_sum = curiosity_mean*params.curius_coef*states.shape[0]
        
        running_avg.append(total_rewards-curiosity_sum)
        avg = sum(running_avg)/len(running_avg)

        tf.summary.scalar('reward', total_rewards-curiosity_sum, step=episode)
        tf.summary.scalar('reward_avg', avg, step=episode)
        tf.summary.scalar('lenght', states.shape[0], step=episode)
        tf.summary.scalar('curiosity_mean', curiosity_mean, step=episode)
        tf.summary.scalar('curiosity_std', curiosity_std, step=episode)
        tf.summary.scalar('curiosity_sum', curiosity_sum, step=episode)

        t.set_description(f"Reward: {total_rewards:.2f} - Reward(Raw): {total_rewards-curiosity_sum:.2f}  - Avg: {avg:.2f} - Iterations: {states.shape[0]} curiosity: {curiosity_mean:.2f} curiosity epoisode: {curiosity_sum:.2f}")

        episode = tf.constant(episode, dtype=tf.int64)

        if len(memory) >= 15_000 and int(episode) % params.train_interval == 0:
            stats = [] # kl, policy_loss, mean_ratio, mean_clipped_ratio, mean_advantage, mean_logprob
            for _ in range(params.iters):
                batch = memory.sample(batch_size)
                history = training_step_ppo(batch, actor, action_space, clip_ratio, policy_optimizer, episode)
                stats.append(history)

            log_stats(stats, episode)

            for _ in range(params.iters):
                batch = memory.sample_critic(batch_size)
                training_step_critic(batch, critic, value_optimizer, episode)

            for _ in range(params.iters):
                batch = memory.sample_curiosity(batch_size_curius)
                training_step_curiosty(batch, curiosity, curiosity_optimizer, action_space,  episode)

            memory.reset()


        if episode % params.save_freq == 0 and episode > 0: 
            NAME = f"{config.MODELS_DIR}{params.env_name}{params.version}_{config.RUN_NAME}_{episode}"
            
            actor.save(f"{NAME}.actor.h5") # type: ignore
            critic.save(f"{NAME}.critic.h5") # type: ignore
            curiosity.save(f"{NAME}.curiosity.h5") # type: ignore

run()