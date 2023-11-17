import tensorflow as tf


@tf.function
def logprobabilities(logits, actions, num_actions):
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(tf.one_hot(actions, num_actions) * logprobabilities_all, axis=1)
    return logprobability

@tf.function
def clip(values, clip_ratio):
    return tf.minimum(tf.maximum(values, 1-clip_ratio), 1+clip_ratio)

@tf.function
def training_step_ppo(batch,
                      actor,
                      num_of_actions,
                      clip_ratio,
                      optimizer: tf.keras.optimizers.Optimizer,
                      step: int
                      ):
    
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer = batch
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        logits = actor(observation_buffer)
        
        ratio = tf.exp(
            logprobabilities(logits, action_buffer, num_of_actions)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(logprobability_buffer - logprobabilities(logits, action_buffer, num_of_actions))
    kl = tf.reduce_mean(kl)

    # tf.summary.scalar('kl', kl, step=step) # type: ignore
    # tf.summary.scalar('loss', policy_loss, step=step)
    # tf.summary.scalar('mean_ratio', tf.reduce_mean(ratio), step=step)
    # tf.summary.scalar('mean_clipped_ratio', tf.reduce_mean(min_advantage), step=step)
    # tf.summary.scalar('mean_advantage', tf.reduce_mean(advantage_buffer), step=step)
    # tf.summary.scalar('mean_logprob', tf.reduce_mean(logprobability_buffer), step=step)

    mean_ratio = tf.reduce_mean(ratio)
    mean_clipped_ratio = tf.reduce_mean(min_advantage)
    mean_advantage = tf.reduce_mean(advantage_buffer)
    mean_logprob = tf.reduce_mean(logprobability_buffer)

    return kl, policy_loss, mean_ratio, mean_clipped_ratio, mean_advantage, mean_logprob

@tf.function
def training_step_critic(
        batch,
        critic,
        optimizer: tf.keras.optimizers.Optimizer,
        step: int
):
    observation_buffer, target_buffer = batch
    with tf.GradientTape() as tape:
        values = critic(observation_buffer)
        loss = tf.reduce_mean(tf.square(target_buffer - values))

    gradients = tape.gradient(loss, critic.trainable_variables)
    optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

    tf.summary.scalar('critic_loss', loss, step=step) # type: ignore
