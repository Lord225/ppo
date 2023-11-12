from typing import Any, Callable, Tuple
import gym
import numpy as np
import tensorflow as tf
import cv2 

from gym.wrappers import frame_stack

def get_packman(human: bool = False):
    return gym.make("ALE/MsPacman-v5", obs_type="image", frameskip=4, render_mode="human" if human else "rgb_array")

def get_packman_stack_frames(human: bool = False):
    env = get_packman(human)
    env = frame_stack.FrameStack(env, 2)
    return env

def pacman_transform_observation(observation, target_size):
    observation = cv2.resize(observation, target_size)
    return observation.astype(np.float32)/255

def pacman_transform_observation_stack(observation, target_size):
    observation = observation.__array__()
    # reshpae from (2, 210, 160, 3) to (210, 160, 2, 3)
    observation = np.transpose(observation, (1, 2, 0, 3))
    # reshape from (210, 160, 2, 3) to (210, 160, 6)
    observation = np.reshape(observation, (observation.shape[0], observation.shape[1], -1))
    # resize to target size
    observation = cv2.resize(observation, target_size)
    return observation.astype(np.float32)/255

def pacman_transform_observation_stack_grayscale(observation, target_size):
    observation = observation.__array__()
    # grayscale
    observation = np.mean(observation, axis=3)
    # reshpae from (2, 210, 160, 3) to (210, 160, 2, 3)
    observation = np.transpose(observation, (1, 2, 0, 3))
    # reshape from (210, 160, 2, 3) to (210, 160, 6)
    observation = np.reshape(observation, (observation.shape[0], observation.shape[1], -1))
    # resize to target size
    observation = cv2.resize(observation, target_size, interpolation=cv2.INTER_NEAREST)
    # cut lower part of the image
    observation = observation[:85, :, :] # shape (85, 50, 2)
    return observation.astype(np.float32)/255


    
ObservationTransformer = Callable[[Any], np.ndarray]

def make_tensorflow_env_step(env: gym.Env, observation_transformer: ObservationTransformer)-> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    def step(action):
        state, reward, done, _, _ = env.step(int(action))
        state = observation_transformer(state)
        return (state.astype(np.float32), np.array(reward, np.float32), np.array(done, np.int32))

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32)] # type: ignore
    )
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