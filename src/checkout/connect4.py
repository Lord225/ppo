from pettingzoo.classic import connect_four_v3
import argparse
import os
import sys

from tangled_up_in_unicode import age
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

while True:
    env.reset()
    # run the episode
    for agent in env.agent_iter():
        observation, reward, term, tru, info = env.last()

        if term or tru:
            break
            
        
        player = 0 if agent == 'player_0' else 1
        
        state = [observation['observation'].reshape(1, 6, 7, 2), np.array([player])] # type: ignore
        
        if player == 0:
            action = model(state)[0] # type: ignore
        else:
            action = model2(state)[0] # type: ignore
        
        legal_moves = observation['action_mask'] # type: ignore
        
        # categorical action
        action = np.argmax(action * legal_moves)

        # step the environment
        env.step(action)

    


