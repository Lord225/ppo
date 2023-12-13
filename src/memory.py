import tensorflow as tf
from common import HistorySampleCriticType, HistorySampleCuriosityType, HistorySampleType, PPOReplayHistoryType, ReplayHistoryType


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



@tf.function
def discounted_cumulative_sums_tf(x, discount_rate)-> tf.Tensor:
    size = tf.shape(x)[0]
    x = tf.reverse(x, axis=[0])
    buffer = tf.TensorArray(dtype=tf.float32, size=size)

    discounted_sum = tf.constant(0.0, dtype=tf.float32)

    for i in tf.range(size):
        discounted_sum = x[i] + discount_rate * discounted_sum # type: ignore
        buffer = buffer.write(i, discounted_sum)
    
    return tf.reverse(buffer.stack(), axis=[0]) # type: ignore

class PPOReplayMemory:
    def __init__(self, max_size, state_shape, next_state_shape=None, gamma=0.99, lam=0.95, gather_next_states=False):
        states_shape = (max_size,) + state_shape
        if next_state_shape is None:
            next_state_shape = states_shape
        
        self.states_buffer = tf.Variable(tf.zeros(states_shape,  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.advantages_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.actions_buffer = tf.Variable(tf.zeros((max_size), dtype=tf.int32), trainable=False, dtype=tf.int32)
        self.rewards_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.return_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)
        self.logprobability_buffer = tf.Variable(tf.zeros((max_size),  dtype=tf.float32), trainable=False, dtype=tf.float32)

        if gather_next_states:
            self.next_states_buffer = tf.Variable(tf.zeros(next_state_shape, dtype=tf.float32), trainable=False, dtype=tf.float32)
        else:
            self.next_states_buffer = None

        self.gamma = gamma
        self.lam = lam
        self.max_size = max_size
        self.count = 0
        self.real_size = 0

    def reset(self):
        self.real_size = 0
        self.count = 0

    
    @tf.function(reduce_retracing=True)
    def add_tf(self, states, actions, rewards, values, logprobabilities,
               states_buffer, advantages_buffer, actions_buffer, rewards_buffer, return_buffer, logprobability_buffer, next_states_buffer,
               gamma, lam, max_size, count):
        
        size = len(states)
        indices = tf.range(count, count + size) % max_size
        
        states_buffer = tf.tensor_scatter_nd_update(states_buffer, indices[:, None], states)
        actions_buffer = tf.tensor_scatter_nd_update(actions_buffer, indices[:, None], actions)
        rewards_buffer = tf.tensor_scatter_nd_update(rewards_buffer, indices[:, None], rewards)
        logprobability_buffer = tf.tensor_scatter_nd_update(logprobability_buffer, indices[:, None], logprobabilities)

        if next_states_buffer is not None:
            next_states_buffer = tf.tensor_scatter_nd_update(next_states_buffer, indices[:, None], states)
        
        count = (count + size) % max_size

        # finish trajectory
        rewards = tf.concat([rewards, [0.0]], axis=0)
        values = tf.concat([values, [0.0]], axis=0)

        deltas = rewards[:-1] + gamma * values[1:] - values[:-1] # type: ignore

        advantages = discounted_cumulative_sums_tf(
            deltas, gamma * lam
        )
        returns = discounted_cumulative_sums_tf(
            rewards, gamma
        )[:-1] # type: ignore

        advantages_buffer = tf.tensor_scatter_nd_update(advantages_buffer, indices[:, None], advantages)
        return_buffer = tf.tensor_scatter_nd_update(return_buffer, indices[:, None], returns)

        return states_buffer, advantages_buffer, actions_buffer, rewards_buffer, return_buffer, logprobability_buffer, next_states_buffer, count
        

    def add(self, observations, actions, rewards, values, logprobabilities, next_observations=None):
        count = tf.constant(self.count, dtype=tf.int32)
        max_size = tf.constant(self.max_size, dtype=tf.int32)
        gamma = tf.constant(self.gamma, dtype=tf.float32)
        lam = tf.constant(self.lam, dtype=tf.float32)

        states_buffer, advantages_buffer, actions_buffer, rewards_buffer, return_buffer, logprobability_buffer, next_states_buffer, new_count = self.add_tf(
                    observations, 
                    actions, 
                    rewards, 
                    values, 
                    logprobabilities,
                    self.states_buffer, 
                    self.advantages_buffer, 
                    self.actions_buffer, 
                    self.rewards_buffer, 
                    self.return_buffer, 
                    self.logprobability_buffer,
                    self.next_states_buffer,
                    gamma, 
                    lam, 
                    max_size, 
                    count) # type: ignore
        
        self.states_buffer = states_buffer
        self.advantages_buffer = advantages_buffer
        self.actions_buffer = actions_buffer
        self.rewards_buffer = rewards_buffer
        self.return_buffer = return_buffer
        self.logprobability_buffer = logprobability_buffer
        self.next_states_buffer = next_states_buffer
        
        self.count = int(new_count) # type: ignore
        self.real_size = min(self.real_size + len(observations), self.max_size)
    

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        advantages = tf.gather(self.advantages_buffer, indices)
        advantages = (advantages - tf.math.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-7)

        return HistorySampleType(
            tf.gather(self.states_buffer, indices),
            tf.gather(self.actions_buffer, indices),
            tf.gather(self.logprobability_buffer, indices),
            advantages,
        )


    def sample_critic(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        return HistorySampleCriticType(
            tf.gather(self.states_buffer, indices),
            tf.gather(self.return_buffer, indices),
        )
    
    def sample_curiosity(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"
        assert self.next_states_buffer is not None, "next_states_buffer is not gathered"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        return HistorySampleCuriosityType(
            tf.gather(self.states_buffer, indices),
            tf.gather(self.actions_buffer, indices),
            tf.gather(self.next_states_buffer, indices),
        )
    
    def sample_encoded_curiosity(self, batch_size, encoder):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"
        assert self.next_states_buffer is not None, "next_states_buffer is not gathered"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        return HistorySampleCuriosityType(
            encoder(tf.gather(self.states_buffer, indices)),
            tf.gather(self.actions_buffer, indices),
            encoder(tf.gather(self.next_states_buffer, indices)),
        )

    def sample_autoencoder(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        indices = tf.random.uniform((batch_size,), 0, self.real_size, dtype=tf.int32)

        return tf.gather(self.states_buffer, indices)
         
    def __len__(self) -> int:
        return self.real_size
    

    
import unittest

# class PPOReplayTest(unittest.TestCase):
#     def test_discounted_cumulative_sums_tf(self):
#         memory = PPOReplayMemory(100, (4,), gamma=0.99, lam=0.95)

#         rewards = tf.constant([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=tf.float32)
#         expected = tf.constant([0.125, 0.25, 0.5, 1.0, 0.0, 0.0], dtype=tf.float32)
#         expected_normalize = (expected - tf.math.reduce_mean(expected)) / (tf.math.reduce_std(expected) + 1e-7)
#         actual = memory.discounted_cumulative_sums_tf(rewards, 0.5)

#         self.assertTrue(tf.reduce_all(tf.math.abs(actual - expected_normalize) < 1e-4))

#     def test_add(self):
#         memory = PPOReplayMemory(2, (4,), gamma=0.99, lam=0.95)

#         states = tf.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=tf.float32)
#         actions = tf.constant([0, 1], dtype=tf.int32)
#         rewards = tf.constant([0.0, 1.0], dtype=tf.float32)
#         values = tf.constant([0.0, 1.0], dtype=tf.float32)
#         logprobs = tf.constant([0.0, 1.0], dtype=tf.float32)
#         dones = tf.constant([False, True], dtype=tf.bool)

#         memory.add(states, actions, rewards, values, logprobs, dones)

#         self.assertTrue(tf.reduce_all(memory.states_buffer[0:2] == states)) # type: ignore
#         self.assertTrue(tf.reduce_all(memory.actions_buffer[0:2] == actions)) # type: ignore
#         self.assertTrue(tf.reduce_all(memory.rewards_buffer[0:2] == rewards))# type: ignore
#         self.assertTrue(tf.reduce_all(memory.value_buffer[0:2] == values))# type: ignore
#         self.assertTrue(tf.reduce_all(memory.logprobability_buffer[0:2] == logprobs))# type: ignore
#         self.assertTrue(tf.reduce_all(memory.advantages_buffer[0:2] == [0.99, 0.0]))# type: ignore
#         self.assertTrue(tf.reduce_all(memory.returns_buffer[0:2] == [0.99, 1.0]))# type: ignore
    

# run test if main
if __name__ == '__main__':
    unittest.main()
