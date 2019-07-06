"""
For invertible datasets in datasets.py, we need to two mapping functions when
evaluating the learned mappings
"""

import tensorflow as tf


def jumpingmean1_low_to_high(x):
    return x + tf.constant(5.0, dtype=tf.float32)


def jumpingmean1_high_to_low(x):
    return x - tf.constant(5.0, dtype=tf.float32)


def jumpingmean2_low_to_high(x):
    return x + tf.constant(7.5, dtype=tf.float32)


def jumpingmean2_high_to_low(x):
    return x - tf.constant(7.5, dtype=tf.float32)


map_to_source = {
    "line1low": jumpingmean1_high_to_low,
    "line2low": jumpingmean2_high_to_low,
    "sine1low": jumpingmean1_high_to_low,
    "sine2low": jumpingmean2_high_to_low,
}

map_to_target = {
    "line1low": jumpingmean1_low_to_high,
    "line2low": jumpingmean2_low_to_high,
    "sine1low": jumpingmean1_low_to_high,
    "sine2low": jumpingmean2_low_to_high,
}
