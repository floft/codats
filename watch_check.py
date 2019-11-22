#!/usr/bin/env python3
"""
Check the watch data for inf/nan's, and count how many examples we have
"""
import tensorflow as tf

from absl import app
from absl import flags

from load_datasets import load_da

FLAGS = flags.FLAGS


def assert_all_false(tensor):
    assert tf.reduce_sum(tf.cast(tensor, tf.int8)) == 0


def check_x(x):
    assert_all_false(tf.math.is_nan(x))
    assert_all_false(tf.math.is_inf(x))


def check_not_none(value):
    assert value is not None, "dataset file probably doesn't exist"


def main(argv):
    # We use test=True so we concat the validation data into the training data.
    # Otherwise, we never check/count the real test data.
    sources, target = load_da("watch_noother", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", "1",
        test=True)
    train_count = 0
    test_count = 0

    for i, source in enumerate(sources):
        check_not_none(source.train_evaluation)

        for x, y in source.train_evaluation:
            check_x(x)
            train_count += len(x)

        check_not_none(source.test_evaluation)

        for x, y in source.test_evaluation:
            check_x(x)
            test_count += len(x)

    print("Training examples:", train_count)
    print("Testing examples:", test_count)

    # Note: we included all users in the sources, so this isn't really necessary
    check_not_none(target.train_evaluation)

    for x, y in target.train_evaluation:
        check_x(x)

    check_not_none(target.test_evaluation)

    for x, y in target.test_evaluation:
        check_x(x)


if __name__ == "__main__":
    app.run(main)
