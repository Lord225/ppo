import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import enviroments

env = enviroments.get_pacman_stack_frames_big()

state, _ = env.reset()

for i in range(100):
    state, _, _, _, _ = env.step(env.action_space.sample())

import matplotlib.pyplot as plt

# just plot the first frame


s = enviroments.pacman_transform_grayscale_observation_stack_big(state)

# plot all six frames
fig, axs = plt.subplots(2, 3)

for i in range(2):
    for j in range(3):
        axs[i, j].imshow(s[:, :, i*3+j], cmap='hsv', vmin=0, vmax=1)
        axs[i, j].axis('off')
        axs[i, j].set_title(f"Frame {i*3+j}")

plt.show()
