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

def training_step_ppo_selfplay(
        batch1,
        batch2,
        actor,
        num_of_actions,
        clip_ratio,
        optimizer: tf.keras.optimizers.Optimizer,
        step: int
):# -> tuple[Any, Any, Any, Any, Any, Any]:
    observation_bufferp1, action_bufferp1, logprobability_bufferp1, advantage_bufferp1 = batch1
    observation_bufferp2, action_bufferp2, logprobability_bufferp2, advantage_bufferp2 = batch2

    # mix two batches and add new buffer that will contain 0 for player 1 and 1 for player 2
    # observation_buffer = tf.concat([observation_bufferp1, observation_bufferp2], axis=0)
    # action_buffer = tf.concat([action_bufferp1, action_bufferp2], axis=0)
    # logprobability_buffer = tf.concat([logprobability_bufferp1, logprobability_bufferp2], axis=0)
    # advantage_buffer = tf.concat([advantage_bufferp1, advantage_bufferp2], axis=0)
    # create new buffer that will contain 0 for player 1 and 1 for player 2, (batchsize,)
    # batchsize1 = tf.shape(observation_bufferp1)[0]
    # batchsize2 = tf.shape(observation_bufferp2)[0]
    # player_buffer = tf.concat([tf.zeros((batchsize1, 1)), tf.ones((batchsize2, 1))], axis=0)

    #observation_buffer, action_buffer, logprobability_buffer, advantage_buffer = batch1

    # concat both batches
    observation_buffer = tf.concat([observation_bufferp1, observation_bufferp2], axis=0)
    action_buffer = tf.concat([action_bufferp1, action_bufferp2], axis=0)
    logprobability_buffer = tf.concat([logprobability_bufferp1, logprobability_bufferp2], axis=0)
    advantage_buffer = tf.concat([advantage_bufferp1, advantage_bufferp2], axis=0)
    


    with tf.GradientTape() as tape: 
        logits = actor(observation_buffer)
        
        ratio = tf.exp(
            logprobabilities(logits, action_buffer, num_of_actions)
            - logprobability_buffer # type: ignore
        )  
        min_advantage = tf.where(
            advantage_buffer > 0, # type: ignore
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
        
        policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
        optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(logprobability_buffer - logprobabilities(logits, action_buffer, num_of_actions)) # type: ignore
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

#@tf.function
def training_step_critic_selfplay(
        batch1,
        batch2,
        critic,
        optimizer: tf.keras.optimizers.Optimizer,
        step: int
):
    #observation_buffer, target_buffer = batch
    observation_bufferp1, target_bufferp1 = batch1
    observation_bufferp2, target_bufferp2 = batch2

    # mix two batches and add new buffer that will contain 0 for player 1 and 1 for player 2
    observation_buffer = tf.concat([observation_bufferp1, observation_bufferp2], axis=0)
    target_buffer = tf.concat([target_bufferp1, target_bufferp2], axis=0)
    # create new buffer that will contain 0 for player 1 and 1 for player 2, (batchsize,)
    batchsize1 = tf.shape(observation_bufferp1)[0]
    batchsize2 = tf.shape(observation_bufferp2)[0]
    player_buffer = tf.concat([tf.zeros((batchsize1, 1)), tf.ones((batchsize2, 1))], axis=0)


    
    with tf.GradientTape() as tape:
        values = critic(observation_buffer)
        loss = tf.reduce_mean(tf.square(target_buffer - values))

        gradients = tape.gradient(loss, critic.trainable_variables)
        optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

    #tf.summary.scalar('critic_loss', loss, step=step) # type: ignore

    return tf.reduce_mean(loss)

@tf.function
def training_step_curiosty(
        batch,
        curiosity,
        optimizer: tf.keras.optimizers.Optimizer,
        num_of_actions,
        step: int
):
    observation_buffer, action_buffer, next_observation_buffer = batch

    action_buffer = tf.one_hot(action_buffer, num_of_actions, dtype=tf.float32)

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(next_observation_buffer - curiosity([observation_buffer, action_buffer])))

    gradients = tape.gradient(loss, curiosity.trainable_variables)
    optimizer.apply_gradients(zip(gradients, curiosity.trainable_variables))

    tf.summary.scalar('curiosity_loss', loss, step=step) # type: ignore

@tf.function
def training_step_selfplay_curiosty(
        batch1,
        batch2,
        curiosity,
        optimizer: tf.keras.optimizers.Optimizer,
        num_of_actions,
        step: int
):
    observation_buffer1, action_buffer1, next_observation_buffer1 = batch1
    observation_buffer2, action_buffer2, next_observation_buffer2 = batch2

    observation_buffer = tf.concat([observation_buffer1, observation_buffer2], axis=0)
    action_buffer = tf.concat([action_buffer1, action_buffer2], axis=0)
    next_observation_buffer = tf.concat([next_observation_buffer1, next_observation_buffer2], axis=0)

    action_buffer = tf.one_hot(action_buffer, num_of_actions, dtype=tf.float32)

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(next_observation_buffer - curiosity([observation_buffer, action_buffer])))

    gradients = tape.gradient(loss, curiosity.trainable_variables)
    optimizer.apply_gradients(zip(gradients, curiosity.trainable_variables))

    tf.summary.scalar('curiosity_loss', loss, step=step) # type: ignore



@tf.function
def training_step_autoencoder(
    batch,
    autoencoder,
    optimizer: tf.keras.optimizers.Optimizer,
    step: int
):
    observation_buffer = batch

    # train autoencoder

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(observation_buffer - autoencoder(observation_buffer)))

    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

    tf.summary.scalar('autoencoder_loss', loss, step=step) # type: ignore

