import tensorflow as tf

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, agent, observation_preprocess, save_all: bool = False, curiosity = None):
    history_rewards = []
    history = []

    with tqdm() as t:
        while True:
            state, _ = env.reset()
            state = observation_preprocess(state)

            rewards = []
            history_episode = []
            i = 0

            while True:
                # check how obs looks like (two images)
                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(state[:,:, 0])
                # ax[1].imshow(state[:,:, 1])
                # plt.show()
                print(i)
                i += 1
                state = tf.expand_dims(state, 0)
                action_logits_t = agent(state)

                action = tf.squeeze(tf.random.categorical(action_logits_t, 1), axis=1)
                action = tf.cast(action, tf.int32)
                action = tf.squeeze(action)

                new_state, reward, done, _, _ = env.step(int(action))

                if curiosity is not None:
                    action_one_hot = tf.one_hot(action, env.action_space.n)
                    action_one_hot = tf.expand_dims(action_one_hot, 0)
                    predicted_state = curiosity([state, action_one_hot])

                    plt.imshow(predicted_state[0,:,:,5])
                    plt.show()

                state = observation_preprocess(new_state)

                history_episode.append((state, action, reward, done))
            
                rewards.append(reward)

                if done:
                    break

                env.render()

            history_rewards.append((sum(rewards), len(rewards), np.mean(rewards), np.std(rewards)))
            history.append(history_episode)

            total_mean = np.mean([x[0] for x in history_rewards])
            total_std = np.std([x[3] for x in history_rewards])

            t.set_description(f"Reward: {sum(rewards):.2f} ({total_mean:.2f} +/- {total_std:.2f}) - Iterations: {len(rewards)}")
            t.set_postfix_str(f'{sum(len(x[0]) for x in history_episode)}')
            t.update(1)



            
            

            




