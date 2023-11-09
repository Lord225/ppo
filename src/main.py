import tensorflow as tf
import tensorboard as tb


from env import get_packman


env = get_packman()


print(env.action_space)
print(env.observation_space.shape)