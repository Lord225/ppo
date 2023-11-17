from collections import deque
import tensorflow as tf
import tensorboard
import os
import argparse
import tqdm
from algorithms.proximal_policy_optimalization import training_step_critic, training_step_ppo
from common import splash_screen
import config
import enviroments
import os


parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="resume from a model") 



args = parser.parse_args()


env = enviroments.get_pole()

params = argparse.Namespace()

params.env_name = env.spec.id
params.version = "v1.0"
params.DRY_RUN = False

params.actor_lr  = 0.001
params.critic_lr = 0.001

params.action_space = env.action_space.n # type: ignore
params.observation_space_raw = env.observation_space.shape
params.observation_space = (4,)

params.episodes = 100000
params.max_steps_per_episode = 200

params.discount_rate = 0.99

params.eps_decay_len = 1000
params.eps_min = 0.1

params.clip_ratio = 0.2
params.lam = 0.97

params.batch_size = 1024

params.train_interval = 1
params.iters = 1


params.save_freq = 500
if args.resume is not None:
    params.resumed = 'resumed from: ' + os.path.basename(args.resume)

splash_screen(params)

def get_actor():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'.actor.h5')
        print("loaded model from", args.resume + '.actor.h5')
        return model

    observation_input = tf.keras.Input(shape=params.observation_space, dtype=tf.float32)
    x = tf.keras.layers.Dense(64, activation=tf.tanh)(observation_input)
    x = tf.keras.layers.Dense(64, activation=tf.tanh)(x)
    logits = tf.keras.layers.Dense(params.action_space)(x)
    return tf.keras.Model(inputs=observation_input, outputs=logits)

def get_critic():
    if args.resume is not None:
        model = tf.keras.models.load_model(args.resume+'.critic.h5')
        print("loaded model from", args.resume + '.critic.h5')
        return model

    observation_input = tf.keras.Input(shape=params.observation_space, dtype=tf.float32)
    x = tf.keras.layers.Dense(64, activation=tf.tanh)(observation_input)
    x = tf.keras.layers.Dense(64, activation=tf.tanh)(x)
    value = tf.squeeze(tf.keras.layers.Dense(1)(x))
    return tf.keras.Model(inputs=observation_input, outputs=value)

actor = get_actor()
critic = get_critic()

policy_optimizer = tf.keras.optimizers.Adam(learning_rate=params.actor_lr)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=params.critic_lr)


from memory import PPOReplayMemory
from exp_collectors.play import get_ppo_runner


def run():
    running_avg = deque(maxlen=200)

    memory = PPOReplayMemory(8000, params.observation_space)

    env_step = enviroments.make_tensorflow_env_step(env, lambda x: x)
    env_reset = enviroments.make_tensorflow_env_reset(env, lambda x: x)

    runner = get_ppo_runner(env_step)
    runner = tf.function(runner)

    action_space = tf.constant(params.action_space, dtype=tf.int32)
    max_steps_per_episode = tf.constant(params.max_steps_per_episode, dtype=tf.int64)
    batch_size = tf.constant(params.batch_size, dtype=tf.int64)
    clip_ratio = tf.constant(params.clip_ratio, dtype=tf.float32)

    t = tqdm.tqdm(range(params.episodes))
    for episode in t:
        initial_state = env_reset()

        (states, actions, rewards, values, log_probs), total_rewards = runner(initial_state, actor, critic, max_steps_per_episode, action_space) # type: ignore

        memory.add(states, actions, rewards, values, log_probs)

        running_avg.append(total_rewards)
        avg = sum(running_avg)/len(running_avg)

        tf.summary.scalar('reward', total_rewards, step=episode)
        tf.summary.scalar('reward_avg', avg, step=episode)
        tf.summary.scalar('lenght', states.shape[0], step=episode)

        t.set_description(f"Reward: {total_rewards:.2f} - Avg: {avg:.2f} - Iterations: {states.shape[0]} size: {len(memory)}")

        episode = tf.constant(episode, dtype=tf.int64)
        
        if len(memory) >= params.batch_size and int(episode) % params.train_interval == 0:
            batch = memory.sample(batch_size)
            for _ in range(params.iters):
                training_step_ppo(batch, actor, action_space, clip_ratio, policy_optimizer, episode)

            batch = memory.sample_critic(batch_size)
            for _ in range(params.iters):
                training_step_critic(batch, critic, value_optimizer, episode)

            #memory.reset()

        if episode % params.save_freq == 0 and episode > 0: 
            NAME = f"{config.MODELS_DIR}{params.env_name}{params.version}_{config.RUN_NAME}_{episode}"
            
            actor.save(f"{NAME}.h5.actor") # type: ignore
            critic.save(f"{NAME}.h5.critic") # type: ignore

run()