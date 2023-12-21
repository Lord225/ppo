import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="path to model")
parser.add_argument("--curius", type=str, default=None, help="path to curius model")

args = parser.parse_args()

import tensorflow as tf
import enviroments

env = enviroments.get_pacman_stack_frames_big(human=False)

model = tf.keras.models.load_model(args.model)

if args.model is None:
    raise ValueError("model argument is required")

if args.curius is not None:
    curiosity = tf.keras.models.load_model(args.curius)
else:
    curiosity = None


state, _ = env.reset()

from checkout.common import run_episode

run_episode(env, model, lambda x: enviroments.pacman_transform_grayscale_observation_stack_big(x), curiosity=curiosity)