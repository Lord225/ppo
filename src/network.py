

import tensorflow as tf


from typing import Tuple


def ms_packman_head(input_shape: Tuple):

    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_layer)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(9, activation='softmax')(x)

    return tf.keras.Model(inputs=input_layer, outputs=output_layer)





