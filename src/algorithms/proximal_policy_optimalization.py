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
                      actor_model,
                      num_of_actions,
                      clip_ratio,
                      optimizer: tf.keras.optimizers.Optimizer,
                      step: int
                      ):
    
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer = batch
    with tf.GradientTape() as tape:
        logits = actor_model(observation_buffer)

        ratio = tf.exp(logprobabilities(logits, action_buffer, num_of_actions) - logprobability_buffer)

        clipped_ratio = clip(ratio, clip_ratio)

        loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, clipped_ratio * advantage_buffer))

    gradients = tape.gradient(loss, actor_model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, actor_model.trainable_variables))

    kl = tf.reduce_mean(logprobability_buffer - logprobabilities(logits, action_buffer, num_of_actions))
    kl = tf.reduce_mean(kl)

    tf.summary.scalar('kl', kl, step=step) # type: ignore
    tf.summary.scalar('loss', loss, step=step)
    tf.summary.scalar('mean_ratio', tf.reduce_mean(ratio), step=step)
    tf.summary.scalar('mean_clipped_ratio', tf.reduce_mean(clipped_ratio), step=step)
    tf.summary.scalar('mean_advantage', tf.reduce_mean(advantage_buffer), step=step)
    tf.summary.scalar('mean_logprob', tf.reduce_mean(logprobability_buffer), step=step)

@tf.function
def training_step_critic(
        batch,
        critic_model,
        optimizer: tf.keras.optimizers.Optimizer,
        step: int
):
    observation_buffer, target_buffer = batch
    with tf.GradientTape() as tape:
        values = critic_model(observation_buffer)
        loss = tf.reduce_mean(tf.square(target_buffer - values))

    gradients = tape.gradient(loss, critic_model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, critic_model.trainable_variables))

    tf.summary.scalar('critic_loss', loss, step=step) # type: ignore
