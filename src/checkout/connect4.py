from pettingzoo.classic import connect_four_v3
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="path to model")
parser.add_argument("--model2", type=str, default=None, help="path to model (player 2)")

args = parser.parse_args()

# load the environment (human)
env = connect_four_v3.env(
    render_mode="human",
)

# load agent from arguments
import tensorflow as tf

if args.model is None:
    raise ValueError("model argument is required")

model = tf.keras.models.load_model(args.model)

if args.model2 is not None:
    model2 = tf.keras.models.load_model(args.model2)
else:
    model2 = model
optimal_ratio = 0
total = 0
while True:
    env.reset()

    # run the episode
    for agent in env.agent_iter():
        observation, reward, term, tru, info = env.last()

        if term or tru:
            print("game over")
            break
        
        player = 0 if agent == 'player_0' else 1
        
        state = observation['observation'] # type: ignore

        state = np.append(state, np.full((6, 7, 1), player), axis=2)
        state = np.array(state, np.float32)
        state = state.reshape(1, 6, 7, 3)

        
        if player == 0:
            logits = model(state) # type: ignore
        else:
            logits = model2(state) # type: ignore
        
        legal_moves = observation['action_mask'] # type: ignore
        
        # random categorical action, apply player mask
        logits = tf.where(legal_moves, logits, -1e10)
        action = int(tf.squeeze(tf.random.categorical(logits, 1), axis=1))

        argmax = np.argmax(logits)

        optimal_ratio += argmax == action
        total += 1

        print("player", agent, "turn, optimal: ", argmax, "action: ", action)
        
        print("logits: ", ', '.join([f'{float(x):.2f}' for x in tf.squeeze(logits).numpy()]))
        print("optimal ratio: ", optimal_ratio/total)
        
        # step the environment
        if np.random.rand() < 0.1:
            env.step(action)
        else:
            env.step(argmax)

    


