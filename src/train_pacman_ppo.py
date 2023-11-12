from collections import deque
import tensorflow as tf
import tensorboard
import os
import argparse
import tqdm
from algorithms.proximal_policy_optimalization import training_step_ppo
from common import splash_screen
import config
import enviroments



parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="resume from a model") 

args = parser.parse_args()


env = enviroments.get_packman_stack_frames()

params = argparse.Namespace()

params.env_name = env.spec.id
params.version = "v2.0"
params.DRY_RUN = True

params.actor_lr = 0.0002
params.critic_lr = 0.001
params.action_space = env.action_space.n # type: ignore
params.observation_space_raw = env.observation_space.shape
params.observation_space = (50, 50, 3*2)

params.episodes = 100000
params.max_steps_per_episode = 1000

params.discount_rate = 0.8
params.eps_decay_len = 4000
params.eps_min = 0.1

params.clip_ratio = 0.2
params.lam = 0.97

params.batch_size = 128

params.actor_train_interval = 1
params.critic_train_interval = 1

params.save_freq = 100
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

    x = tf.keras.layers.Conv2D(32, 4, strides=2, activation="elu")(inputs)
    x = tf.keras.layers.Conv2D(64, 2, strides=2, activation="elu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="elu")(x)
    x = tf.keras.layers.Conv2D(128, 2, strides=1, activation="elu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="elu")(x)
    x = tf.keras.layers.Dense(256, activation="elu")(x)

    critic = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=[outputs, critic])

    return model



model = get_model()


optimizer_actor = tf.keras.optimizers.Adam(params.actor_lr)
optimizer_critic = tf.keras.optimizers.Adam(params.critic_lr)


from memory import PPOReplayMemory


from exp_collectors.play import get_ppo_runner

def run():
    running_avg = deque(maxlen=200)

    memory = PPOReplayMemory(10_000, params.observation_space)

    env_step = enviroments.make_tensorflow_env_step(env, lambda x: enviroments.pacman_transform_observation_stack(x, target_size=params.observation_space[:2]))
    env_reset = enviroments.make_tensorflow_env_reset(env, lambda x: enviroments.pacman_transform_observation_stack(x, target_size=params.observation_space[:2]))

    runner = get_ppo_runner(env_step)
    runner = tf.function(runner)

    action_space = tf.constant(params.action_space, dtype=tf.int32)
    max_steps_per_episode = tf.constant(params.max_steps_per_episode, dtype=tf.int64)
    batch_size = tf.constant(params.batch_size, dtype=tf.int64)
    clip_ratio = tf.constant(params.clip_ratio, dtype=tf.float32)

    t = tqdm.tqdm(range(params.episodes))
    for episode in t:
        epsilon = max(1 - episode / params.eps_decay_len, params.eps_min)
        
        state = env_reset()
        state = tf.constant(state, dtype=tf.float32)

        epsilon = tf.constant(epsilon, dtype=tf.float32)

        (states, actions, rewards, values, log_probs, dones), total_rewards = runner(state,
                                                                              model,
                                                                              max_steps_per_episode,
                                                                              action_space,
        ) # type: ignore

        memory.add(states, actions, rewards, values, log_probs, dones)
        running_avg.append(total_rewards)
        avg = sum(running_avg)/len(running_avg)

        tf.summary.scalar('reward', total_rewards, step=episode)
        tf.summary.scalar('reward_avg', avg, step=episode)
        tf.summary.scalar('lenght', states.shape[0], step=episode)

        t.set_description(f"Reward: {total_rewards:.2f} - Avg: {avg:.2f} - Iterations: {states.shape[0]}")

        # train
        if len(memory) > params.batch_size+1 and episode % params.actor_train_interval == 0:
            batch = memory.sample(batch_size)

            episode = tf.constant(episode, dtype=tf.int64)

            training_step_ppo(batch,
                              model,
                              action_space,
                              clip_ratio,
                              optimizer_actor,
                              episode,
            )

        if episode % params.save_freq == 0 and episode > 0:
            model.save(f"{config.MODELS_DIR}{params.env_name}{params.version}_{config.RUN_NAME}_{episode}.h5")


run()