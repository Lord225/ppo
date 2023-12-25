import argparse
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="path to model")

args = parser.parse_args()

# load model and let it play in the environment
import tensorflow as tf
import enviroments

env = enviroments.get_packman_stack_frames(human=True)

model = tf.keras.models.load_model(args.model)

if args.model is None:
    raise ValueError("model argument is required")

state, _ = env.reset()
state = np.array(state)

state = enviroments.pacman_transform_observation_stack(state, (50, 50))


while True:
    env.render()
    state = tf.expand_dims(state, 0)
    action_logits_t, _ = model(state) # type: ignore
    action_probs_t = tf.nn.softmax(action_logits_t)
    action = tf.argmax(action_probs_t, axis=1)[0]
    state, _, done, _, _ = env.step(int(action)) # type: ignore
    state = np.array(state)
    state = enviroments.pacman_transform_observation_stack(state, (50, 50))
    if done:
        state, _ = env.reset()
        state = enviroments.pacman_transform_observation_stack(state, (50, 50))

    env.render()



