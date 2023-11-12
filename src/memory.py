import scipy
from sympy import Q
import tensorflow as tf
from common import PPOReplayHistoryType, ReplayHistoryType


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
       
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        discounted_sum = tf.constant(0.0)

        for i in tf.range(n):
            reward = x[i] # type: ignore
            discounted_sum = reward + discount * discounted_sum
            returns = returns.write(i, discounted_sum)

        returns = returns.stack()[::-1] # type: ignore

        # Normalize returns
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-7)

        return returns
    
    @tf.function
    def add_tf(self,
            batch, 
            states_buffer,
            actions_buffer,
            rewards_buffer,
            value_buffer,
            logprobability_buffer,
            advantages_buffer,
            returns_buffer,
            count,
            max_size,
            gamma,
            lam,
            ):
        states, actions, rewards, values, logprobs, dones = batch
        batch_size = len(states)
        indices = tf.range(count, count + batch_size) % max_size

        states_buffer = tf.tensor_scatter_nd_update(states_buffer, indices[:, None], states)
        actions_buffer = tf.tensor_scatter_nd_update(actions_buffer, indices[:, None], actions)
        rewards_buffer = tf.tensor_scatter_nd_update(rewards_buffer, indices[:, None], rewards)
        value_buffer = tf.tensor_scatter_nd_update(value_buffer, indices[:, None], values)
        logprobability_buffer = tf.tensor_scatter_nd_update(logprobability_buffer, indices[:, None], logprobs)
        
        advantages = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        returns = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        for i in tf.range(batch_size, dtype=tf.int32):
            if dones[i]:
                advantages = advantages.write(i, 0.0)
                returns = returns.write(i, rewards[i])
            else:
                next_value = value_buffer[(count + i + 1) % max_size]
                next_advantage = advantages_buffer[(count + i + 1) % max_size] # type: ignore
                delta = rewards[i] + gamma * next_value - values[i]
                advantages = advantages.write(i, delta + gamma * lam * next_advantage)
                returns = returns.write(i, rewards[i] + gamma * next_value)

        advantages = advantages.stack()
        returns = returns.stack()

        advantages_buffer = tf.tensor_scatter_nd_update(advantages_buffer, indices[:, None], advantages)
        returns_buffer = tf.tensor_scatter_nd_update(returns_buffer, indices[:, None], returns)

        count = (count + batch_size) % max_size

        return states_buffer, actions_buffer, rewards_buffer, value_buffer, logprobability_buffer, advantages_buffer, returns_buffer, logprobability_buffer, count, 
    def add(self, states, actions, rewards, values, logprobs, dones):
        (states_buffer, actions_buffer, rewards_buffer, value_buffer, logprobability_buffer, advantages_buffer, returns_buffer, logprobability_buffer, count) = self.add_tf((states, actions, rewards, values, logprobs, dones),
                    self.states_buffer,
                    self.actions_buffer,
                    self.rewards_buffer,
                    self.value_buffer,
                    self.logprobability_buffer,
                    self.advantages_buffer,
                    self.returns_buffer,
                    self.count,
                    self.max_size,
                    self.gamma,
                    self.lam,
                    ) # type: ignore
        
        self.states_buffer = states_buffer
        self.actions_buffer = actions_buffer
        self.rewards_buffer = rewards_buffer
        self.value_buffer = value_buffer
        self.logprobability_buffer = logprobability_buffer
        self.advantages_buffer = advantages_buffer
        self.returns_buffer = returns_buffer
        self.count = count
    
    @tf.function
    def sample(self, batch_size) -> PPOReplayHistoryType:
        assert self.count >= batch_size, "buffer contains less samples than batch size"

        indices = tf.range(self.count - batch_size, self.count) % self.max_size

        advantages = self.advantages_buffer[indices] # type: ignore
        advantages = (advantages - tf.math.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-7)

        return PPOReplayHistoryType(
            tf.gather(self.states_buffer, indices),
            tf.gather(self.actions_buffer, indices),
            advantages,
            tf.gather(self.returns_buffer, indices),
            tf.gather(self.logprobability_buffer, indices)
        )
        

import unittest

class PPOReplayTest(unittest.TestCase):
    def test_discounted_cumulative_sums_tf(self):
        memory = PPOReplayMemory(100, (4,), gamma=0.99, lam=0.95)

        rewards = tf.constant([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=tf.float32)
        expected = tf.constant([0.125, 0.25, 0.5, 1.0, 0.0, 0.0], dtype=tf.float32)
        expected_normalize = (expected - tf.math.reduce_mean(expected)) / (tf.math.reduce_std(expected) + 1e-7)
        actual = memory.discounted_cumulative_sums_tf(rewards, 0.5)

        self.assertTrue(tf.reduce_all(tf.math.abs(actual - expected_normalize) < 1e-4))

    def test_add(self):
        memory = PPOReplayMemory(2, (4,), gamma=0.99, lam=0.95)

        states = tf.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=tf.float32)
        actions = tf.constant([0, 1], dtype=tf.int32)
        rewards = tf.constant([0.0, 1.0], dtype=tf.float32)
        values = tf.constant([0.0, 1.0], dtype=tf.float32)
        logprobs = tf.constant([0.0, 1.0], dtype=tf.float32)
        dones = tf.constant([False, True], dtype=tf.bool)

        memory.add(states, actions, rewards, values, logprobs, dones)

        self.assertTrue(tf.reduce_all(memory.states_buffer[0:2] == states)) # type: ignore
        self.assertTrue(tf.reduce_all(memory.actions_buffer[0:2] == actions)) # type: ignore
        self.assertTrue(tf.reduce_all(memory.rewards_buffer[0:2] == rewards))# type: ignore
        self.assertTrue(tf.reduce_all(memory.value_buffer[0:2] == values))# type: ignore
        self.assertTrue(tf.reduce_all(memory.logprobability_buffer[0:2] == logprobs))# type: ignore
        self.assertTrue(tf.reduce_all(memory.advantages_buffer[0:2] == [0.99, 0.0]))# type: ignore
        self.assertTrue(tf.reduce_all(memory.returns_buffer[0:2] == [0.99, 1.0]))# type: ignore
    

# run test if main
if __name__ == '__main__':
    unittest.main()
