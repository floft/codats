"""
For invertible datasets in datasets.py, we need to two mapping functions when
evaluating the learned mappings
"""

import tensorflow as tf


def jumpingmean1_low_to_high(x):
    return x + tf.constant(2.5, dtype=tf.float32)


def jumpingmean1_high_to_low(x):
    return x - tf.constant(2.5, dtype=tf.float32)


def jumpingmean2_low_to_high(x):
    return x + tf.constant(5.0, dtype=tf.float32)


def jumpingmean2_high_to_low(x):
    return x - tf.constant(5.0, dtype=tf.float32)


def scale1_low_to_high(x):
    # Since not centered around 0, must get rid of bias before scaling
    # And, since there's no noise, the bias is the first value.
    bias = tf.slice(x, [0, 0, 0], [tf.shape(x)[0], 1, 1])
    return (x - bias) * tf.constant(1.5, dtype=tf.float32) + bias


def scale1_high_to_low(x):
    bias = tf.slice(x, [0, 0, 0], [tf.shape(x)[0], 1, 1])
    return (x - bias) * tf.constant(1.0/1.5, dtype=tf.float32) + bias


def scale2_low_to_high(x):
    bias = tf.slice(x, [0, 0, 0], [tf.shape(x)[0], 1, 1])
    return (x - bias) * tf.constant(2.0, dtype=tf.float32) + bias


def scale2_high_to_low(x):
    bias = tf.slice(x, [0, 0, 0], [tf.shape(x)[0], 1, 1])
    return (x - bias) * tf.constant(1.0/2.0, dtype=tf.float32) + bias


map_to_source = {
    "line1low": jumpingmean1_high_to_low,
    "line2low": jumpingmean2_high_to_low,
    "sine1low": jumpingmean1_high_to_low,
    "sine2low": jumpingmean2_high_to_low,
    "lineslope1low": scale1_high_to_low,
    "lineslope2low": scale1_high_to_low,
    "sineslope1low": scale2_high_to_low,
    "sineslope2low": scale2_high_to_low,
}

map_to_target = {
    "line1low": jumpingmean1_low_to_high,
    "line2low": jumpingmean2_low_to_high,
    "sine1low": jumpingmean1_low_to_high,
    "sine2low": jumpingmean2_low_to_high,
    "lineslope1low": scale1_low_to_high,
    "lineslope2low": scale1_low_to_high,
    "sineslope1low": scale2_low_to_high,
    "sineslope2low": scale2_low_to_high,
}
