import argparse
from collections import deque
import random
import tensorboard
from gym import Env
from pettingzoo.classic import connect_four_v3
import tensorflow as tf
import tqdm

from algorithms.proximal_policy_optimalization import training_step_critic, training_step_critic_selfplay, training_step_ppo_selfplay
from common import splash_screen
from exp_collectors.play import evaluate_selfplay, run_episode_selfplay
from memory import PPOReplayMemory
import config
import numpy as np


env = connect_four_v3.env()

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None, help="resume from a model") 

args = parser.parse_args()

params = argparse.Namespace()

params.env_name = "connect_four_v3"
params.version = "v5.0"
params.DRY_RUN = False

params.actor_lr  = 1e-7
params.critic_lr = 1e-5

params.action_space = 7
params.observation_space = (6, 7, 2)

params.episodes = 5_000_000
params.max_steps_per_episode = 100

params.discount_rate = 0.98
params.eps_decay_len = 5000
params.eps_min = 0.1

params.clip_ratio = 0.20
params.lam = 0.95

params.batch_size = 2000

params.train_interval = 1
params.iters = 80

params.save_freq = 3000
params.eval_freq = 1000
params.copy_player_freq = 1000
params.update_historic_freq = 5000
params.train_on_historic_freq = 4

splash_screen(params)

def step(action, action_none):
    if action_none == True:
        action = None
    else:
        action = int(action)

    env.step(action)
    observation, reward, termination, truncation, _ = env.last()

    done = termination or truncation
    state = observation['observation'] # type: ignore
    # current player
    player = 0 if env.agent_selection == 'player_0' else 1

    # add player to state
    state = np.append(state, np.full((6, 7, 1), player), axis=2)
    state = np.array(state, np.float32)

    # default reward for players for playing long
    reward += -0.025


    mask = observation["action_mask"] # type: ignore
    return (np.array(state, np.float32), np.array(mask, np.float32), np.array(reward, np.float32), np.array(done, np.int32))

@tf.function(
    input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.bool)], # type: ignore
)
def tf_env_step(action, action_none):
    return tf.numpy_function(step, [action, action_none], (tf.float32, tf.float32, tf.float32, tf.int32))

# define models
def get_model():
    # observation input
    observation_input = tf.keras.layers.Input(shape=(6, 7, 3), name="observation_input")

    # convolutional layers
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation="elu")(observation_input)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation="elu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="elu")(x)
    x = tf.keras.layers.Dense(512, activation="elu")(x)
    x = tf.keras.layers.Dense(256, activation="elu")(x)
    output = tf.keras.layers.Dense(7)(x)

    return tf.keras.Model(inputs=observation_input, outputs=output)

def get_critic():
    # observation input
    observation_input = tf.keras.layers.Input(shape=(6, 7, 3), name="observation_input")

    # convolutional layers
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation="elu")(observation_input)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation="elu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation="elu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="elu")(x)
    x = tf.keras.layers.Dense(512, activation="elu")(x)
    x = tf.keras.layers.Dense(256, activation="elu")(x)
    output = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(inputs=observation_input, outputs=output)

player1 = get_model()
player2 = get_model()

historic_player = get_model()
historic_player.set_weights(player1.get_weights())

critic1 = get_critic()
critic2 = get_critic()


# try running an episode
# todo
# - telemetry
# - train loop
# - params
# - tensorboard & setup
# - checkpointing

optimizer1 = tf.keras.optimizers.Adam(learning_rate=params.actor_lr)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=params.actor_lr)
optimizer_critic1 = tf.keras.optimizers.Adam(learning_rate=params.critic_lr)
optimizer_critic2 = tf.keras.optimizers.Adam(learning_rate=params.critic_lr)

memoryp1 = PPOReplayMemory(10_000, (6, 7, 3)) # exp as p1
memoryp2 = PPOReplayMemory(10_000, (6, 7, 3)) # exp as p2

t = tqdm.tqdm(range(params.episodes))

training_step_ppo_selfplay_p1 = tf.function(training_step_ppo_selfplay)
training_step_ppo_selfplay_p2 = tf.function(training_step_ppo_selfplay)

training_step_critic1 = tf.function(training_step_critic_selfplay)
training_step_critic2 = tf.function(training_step_critic_selfplay)

def log_stats(stats, step):
    # list of kl, policy_loss, mean_ratio, mean_clipped_ratio, mean_advantage, mean_logprob
    tf.summary.scalar('kl', np.mean([x[0] for x in stats]), step=step)
    tf.summary.scalar('loss', np.mean([x[1] for x in stats]), step=step)
    tf.summary.scalar('mean_ratio', np.mean([x[2] for x in stats]), step=step)
    tf.summary.scalar('mean_clipped_ratio', np.mean([x[3] for x in stats]), step=step)
    tf.summary.scalar('mean_logprob', np.mean([x[5] for x in stats]), step=step)


def reset_env():
    env.reset()
    observation, reward, termination, truncation, _ = env.last()
    state = observation['observation'] # type: ignore
    mask = observation["action_mask"] # type: ignore
    mask = np.array(mask, np.float32)

    # current player
    player = 0 if env.agent_selection == 'player_0' else 1

    # add 3rd plane of all zeros or ones depending on player
    state = np.append(state, np.full((6, 7, 1), player), axis=2)
    state = np.array(state, np.float32)

    return (state, mask)

action_space = tf.constant(params.action_space, dtype=tf.int32)
max_steps_per_episode = tf.constant(params.max_steps_per_episode, dtype=tf.int32)
batch_size = tf.constant(params.batch_size, dtype=tf.int64)
clip_ratio = tf.constant(params.clip_ratio, dtype=tf.float32)

running_avgp1 = deque(maxlen=200)
running_avgp2 = deque(maxlen=200)

for i in t:
    env.reset()

    episode = tf.constant(i, dtype=tf.int64)

    epsilon = max(1 - i / params.eps_decay_len, params.eps_min)
    epsilon = tf.constant(epsilon, dtype=tf.float32)

    state, mask = reset_env()


    if np.random.uniform() < 0.5:
        # run with self as p1
        output = run_episode_selfplay(state, mask, player1, player2, critic1, critic2, tf_env_step, max_steps_per_episode, action_space, epsilon) # type: ignore
        P1 = True
    else: 
        # run with self as p2
        output = run_episode_selfplay(state, mask, player2, player1, critic1, critic2, tf_env_step, max_steps_per_episode, action_space, epsilon) # type: ignore
        P1 = False
    
    (states_p1, states_p2, actions_p1, actions_p2, rewards_p1, rewards_p2, values_p1, values_p2, log_probs_p1, log_probs_p2) = output # type: ignore

    # take only player1 data
    # if P1:
    #     memoryp1.add(states_p1, actions_p1, rewards_p1, values_p1, log_probs_p1)
    #     reward_sum = sum(rewards_p1)
    #     running_avg.append(reward_sum)
    # else:
    #     memoryp2.add(states_p2, actions_p2, rewards_p2, values_p2, log_probs_p2)
    #     reward_sum = sum(rewards_p2)
    #     running_avg.append(reward_sum)

    # take both players data
    # memoryp1.add(states_p1, actions_p1, rewards_p1, values_p1, log_probs_p1)
    # memoryp2.add(states_p2, actions_p2, rewards_p2, values_p2, log_probs_p2)
    
    # memoryp1 will contain data only from p1 perspective
    if P1:
        memoryp1.add(states_p1, actions_p1, rewards_p1, values_p1, log_probs_p1)
        memoryp2.add(states_p2, actions_p2, rewards_p2, values_p2, log_probs_p2)
        reward_sump1 = sum(rewards_p1)
        reward_sump2 = sum(rewards_p2)
    else:
        memoryp1.add(states_p2, actions_p2, rewards_p2, values_p2, log_probs_p2)
        memoryp2.add(states_p1, actions_p1, rewards_p1, values_p1, log_probs_p1)
        reward_sump2 = sum(rewards_p1)
        reward_sump1 = sum(rewards_p2)
       

    running_avgp1.append(reward_sump1)
    running_avgp2.append(reward_sump2)

    avgp1 = sum(running_avgp1)/len(running_avgp1)
    avgp2 = sum(running_avgp2)/len(running_avgp2)
    

    t.set_description(f"Avg: {avgp1:.2f} - size: {len(memoryp1)}, {len(memoryp2)}")

    # tensorboard
    tf.summary.scalar('reward', reward_sump1, step=i)
    tf.summary.scalar('reward_avg', avgp1, step=i)
    tf.summary.scalar('reward_p2', reward_sump2, step=i)
    tf.summary.scalar('reward_avg_p2', avgp2, step=i)
    tf.summary.scalar('lenght', states_p1.shape[0] + states_p2.shape[0], step=i)

    # train
    if len(memoryp1) > batch_size and len(memoryp2) > batch_size and i % params.train_interval == 0:
        stats = []

        for j in range(params.iters):
            batch1 = memoryp1.sample(batch_size)
            #batch2 = memoryp1.sample(batch_size)

            history = training_step_ppo_selfplay_p1(
                batch1,
               # batch2,
                player1,
                action_space,
                clip_ratio,
                optimizer1,
                episode,
            ) # type: ignore
            stats.append(history)
            
            batch1 = memoryp2.sample(batch_size)
            #batch2 = memoryp2.sample(batch_size)

            history = training_step_ppo_selfplay_p2(
                batch1,
              #  batch2,
                player2,
                action_space,
                clip_ratio,
                optimizer2,
                episode,
            ) # type: ignore
            stats.append(history)

        log_stats(stats, episode)

        stats = []

        for j in range(params.iters):
            # batch1 = memoryp1.sample_critic(batch_size)
            # batch2 = memoryp2.sample_critic(batch_size)
            # training_step_critic_selfplay(batch1, batch2, critic, optimizer_critic, episode)

            batch1 = memoryp1.sample_critic(batch_size)
            h = training_step_critic1(batch1, critic1, optimizer_critic1, episode)
            stats.append(h)
            batch2 = memoryp2.sample_critic(batch_size)
            h = training_step_critic2(batch2, critic2, optimizer_critic2, episode)
            stats.append(h)

        tf.summary.scalar('critic_loss', np.mean(stats), step=i)
        # clear memory after training
        memoryp1.reset()
        memoryp2.reset()

    # copy player1 to player2
    if i % params.copy_player_freq == 0:
        t.set_description(f"New player...  ")
        # CHECK who is better
        if avgp1 > avgp2:
            player2.set_weights(player1.get_weights())
            critic2.set_weights(critic1.get_weights())
        else:
            player1.set_weights(player2.get_weights())
            critic1.set_weights(critic2.get_weights())
        
        # clear memory
        # memoryp1.reset()
        # memoryp2.reset()

    # copy player1 to historic_player
    if i % params.update_historic_freq == 0:
        t.set_description(f"History update... ")
        historic_player.set_weights(player1.get_weights())


    # save
    if i % params.save_freq == 0 and i > 0:
        t.set_description(f"Saving... ")
        NAME = f"{config.MODELS_DIR}{params.env_name}{params.version}_{config.RUN_NAME}_{episode}"

        player1.save(f"{NAME}.p1.h5") # type: ignore
        player2.save(f"{NAME}.critic.h5")


    # eval
    # if i % params.eval_freq == 0 and i > 0:
    #     t.set_description(f"Evaluating ")
    #     EVAL_EPISODES = 50

    #     history_random = []
    #     history_historic = []
    #     history_self = []
        
    #     # play aginst random
    #     for j in range(EVAL_EPISODES):
    #         state, mask = reset_env()
            
    #         output = evaluate_selfplay(
    #             state,
    #             mask, 
    #             lambda obs: player1(obs, training=False), # type: ignore 
    #             lambda obs: np.random.uniform(size=(1, 7)), # type: ignore 
    #             critic,
    #             tf_env_step, # type: ignore
    #             action_space,
    #             max_steps_per_episode,
    #         ) # type: ignore

    #         history_random.append(output)
    #     t.set_description(f"Evaluating. ")

    #     # play aginst historic
    #     for j in range(EVAL_EPISODES):
    #         state, mask = reset_env()
            
    #         output = evaluate_selfplay(
    #             state,
    #             mask, 
    #             lambda obs: player1(obs, training=False), # type: ignore
    #             lambda obs: historic_player(obs, training=False), # type: ignore
    #             critic,
    #             tf_env_step, # type: ignore
    #             action_space,
    #             max_steps_per_episode,
    #         )

    #         history_historic.append(output)
    #     t.set_description(f"Evaluating.. ")
        
    #     # play aginst self
    #     for j in range(EVAL_EPISODES):
    #         state, mask = reset_env()
            
    #         output = evaluate_selfplay(
    #             state,
    #             mask, 
    #             lambda obs: player1(obs, training=False), # type: ignore
    #             lambda obs: player2(obs, training=False), # type: ignore
    #             critic,
    #             tf_env_step, # type: ignore
    #             action_space,
    #             max_steps_per_episode,
    #         )

    #         history_self.append(output)
    #     t.set_description(f"Evaluating... ")
    #     # history is tupe - win side, length

    #     # print some basic info - win rate, avg length ect 
    #     random_wins = sum([x[0] for x in history_random])
    #     historic_wins = sum([x[0] for x in history_historic])
    #     self_wins = sum([x[0] for x in history_self])

    #     # normalize by number of episodes
    #     random_wins = random_wins/EVAL_EPISODES
    #     historic_wins = historic_wins/EVAL_EPISODES
    #     self_wins = self_wins/EVAL_EPISODES

    #     random_avg_length = sum([x[1] for x in history_random])/len(history_random)
    #     historic_avg_length = sum([x[1] for x in history_historic])/len(history_historic)
    #     self_avg_length = sum([x[1] for x in history_self])/len(history_self)

    #     tf.summary.scalar('eval_random_wins', random_wins, step=i)
    #     tf.summary.scalar('eval_historic_wins', historic_wins, step=i)
    #     tf.summary.scalar('eval_self_wins', self_wins, step=i)

    #     tf.summary.scalar('eval_random_avg_length', random_avg_length, step=i)
    #     tf.summary.scalar('eval_historic_avg_length', historic_avg_length, step=i)
    #     tf.summary.scalar('eval_self_avg_length', self_avg_length, step=i)
