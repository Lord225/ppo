import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="path to model")

args = parser.parse_args()

import tensorflow as tf
import enviroments

env = enviroments.get_packman_stack_frames(human=True)

model = tf.keras.models.load_model(args.model)

if args.model is None:
    raise ValueError("model argument is required")

state, _ = env.reset()

from checkout.common import run_episode

run_episode(env, model, lambda x: enviroments.pacman_transform_observation_stack_big_gray(x))