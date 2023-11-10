from typing import Any, Callable, Tuple
import gym
import numpy as np
import tensorflow as tf
import cv2 

def get_packman(human: bool = False):
    return gym.make("ALE/MsPacman-v5", obs_type="image", frameskip=4, render_mode="human" if human else "rgb_array")

def pacman_transform_observation(observation, target_size):
    # use numpy to resize image to target size
    observation = cv2.resize(observation, target_size)
    return observation.astype(np.float32)/255
    
    
ObservationTransformer = Callable[[Any], np.ndarray]

def make_tensorflow_env_step(env: gym.Env, observation_transformer: ObservationTransformer)-> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    def step(action):
        state, reward, done, _, _ = env.step(int(action))
        state = observation_transformer(state)
        return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

    @tf.function
    def tf_env_step(action):
        return tf.numpy_function(step, [action], (tf.float32, tf.float32, tf.int32))
    return tf_env_step # type: ignore

def make_tensorflow_env_reset(env: gym.Env, observation_transformer: ObservationTransformer) -> Callable[[], tf.Tensor]:
    def reset():
        state, _ = env.reset()
        return observation_transformer(state)
    
    @tf.function
    def tf_env_reset():
        return tf.numpy_function(reset, [], tf.float32)
    return tf_env_reset # type: ignore