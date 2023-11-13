import tensorflow as tf

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, agent, observation_preprocess, save_all: bool = False):
    history_rewards = []
    history = []

    with tqdm() as t:
        while True:
            state, _ = env.reset()
            state = observation_preprocess(state)

            rewards = []
            history_episode = []

            while True:
                # check how obs looks like
                #plt.imshow(state[:,:,:3])
                #plt.show()
                
                state = tf.expand_dims(state, 0)
                action_logits_t, _ = agent(state)

                action_probs_t = tf.nn.softmax(action_logits_t)

                action = tf.argmax(action_probs_t, axis=1)[0]

                state, reward, done, _, _ = env.step(int(action))

                state = observation_preprocess(state)

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



            
            

            



