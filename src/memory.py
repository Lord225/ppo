import scipy
import tensorflow as tf
from common import ReplayHistoryType


class ReplayMemory:
    def __init__(self, max_size, state_shape):
        states_shape = (max_size,) + state_shape
        self.states_buffer = tf.Variable(tf.zeros(states_shape,  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.actions_buffer = tf.Variable(tf.zeros((max_size), dtype=tf.int32), trainable=False, dtype=tf.int32)
        self.returns_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.next_states_buffer = tf.Variable(tf.zeros(states_shape, dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.dones_buffer = tf.Variable(tf.zeros((max_size), dtype=tf.float32), trainable=False, dtype=tf.float32)

        self.max_size = max_size
        self.count = 0
        self.real_size = 0

    def add(self, states, actions, returns, next_states, dones):
        batch_size = len(states)
        indices = tf.range(self.count, self.count + batch_size) % self.max_size
        self.states_buffer = tf.tensor_scatter_nd_update(self.states_buffer, indices[:, None], states)
        self.actions_buffer = tf.tensor_scatter_nd_update(self.actions_buffer, indices[:, None], actions)
        self.returns_buffer = tf.tensor_scatter_nd_update(self.returns_buffer, indices[:, None], returns)
        self.next_states_buffer = tf.tensor_scatter_nd_update(self.next_states_buffer, indices[:, None], next_states)
        self.dones_buffer = tf.tensor_scatter_nd_update(self.dones_buffer, indices[:, None], dones)
        self.count = (self.count + batch_size) % self.max_size
        self.real_size = min(self.real_size + batch_size, self.max_size)

    @tf.function
    def sample(self, batch_size) -> ReplayHistoryType:
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        return ReplayHistoryType(
            tf.gather(self.states_buffer, indices),
            tf.gather(self.actions_buffer, indices),
            tf.gather(self.returns_buffer, indices),
            tf.gather(self.next_states_buffer, indices),
            tf.gather(self.dones_buffer, indices)
        )

    def __len__(self):
        return self.real_size


class PPOReplayMemory:
    def __init__(self, max_size, state_shape, gamma=0.99, lam=0.95):
        states_shape = (max_size,) + state_shape
        self.states_buffer = tf.Variable(tf.zeros(states_shape,  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.actions_buffer = tf.Variable(tf.zeros((max_size), dtype=tf.int32), trainable=False, dtype=tf.int32)
        self.advantages_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.rewards_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.returns_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.value_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.logprobability_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)

        self.max_size = max_size
        self.count = 0
        self.trajectory_start_index = 0

        self.gamma = gamma
        self.lam = lam

    @tf.function
    def discounted_cumulative_sums_tf(self, x, discount):
        x = tf.cast(x[::-1], dtype=tf.float32) 
        n = tf.shape(x)[0]
        cumulative = tf.TensorArray(dtype=tf.float32, size=n) 
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = x[i] # type: ignore
            discounted_sum = reward + discount * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = cumulative.write(i, discounted_sum)

        returns = returns.stack()[::-1] # type: ignore

        # Normalize returns
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-7)

        return returns
    
    def add(self, states, actions, rewards, values, logprobs, dones):
        batch_size = len(states)
        indices = tf.range(self.count, self.count + batch_size) % self.max_size
        self.states_buffer = tf.tensor_scatter_nd_update(self.states_buffer, indices[:, None], states)
        self.actions_buffer = tf.tensor_scatter_nd_update(self.actions_buffer, indices[:, None], actions)
        self.rewards_buffer = tf.tensor_scatter_nd_update(self.rewards_buffer, indices[:, None], rewards)
        self.value_buffer = tf.tensor_scatter_nd_update(self.value_buffer, indices[:, None], values)
        self.logprobability_buffer = tf.tensor_scatter_nd_update(self.logprobability_buffer, indices[:, None], logprobs)
        self.dones_buffer = tf.tensor_scatter_nd_update(self.dones_buffer, indices[:, None], dones)
        self.count = (self.count + batch_size) % self.max_size

    def trajectory_end(self, last_state, last_value):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        last_state = tf.expand_dims(last_state, 0)