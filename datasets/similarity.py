#!/usr/bin/env python3
"""
Check how similar two domain adaptation problems are using dynamic time warping

Note: sets CUDA_VISIBLE_DEVICES= so that it doesn't use the GPU.
"""
import os
import random
import numpy as np

from absl import app
from absl import flags
from dtw import dtw
from numpy.linalg import norm

import datasets

from datasets import calc_normalization, apply_normalization

FLAGS = flags.FLAGS

flags.DEFINE_enum("source", None, datasets.names(), "What dataset to use as the source")
flags.DEFINE_enum("target", None, datasets.names(), "What dataset to use as the target")
flags.DEFINE_integer("sample_size", 100, "How many pairs to sample on which distance is computed, split evenly among classes")
flags.DEFINE_integer("repeat", 3, "How times to repeat the sampling for computing the standard deviation")

flags.mark_flag_as_required("source")
flags.mark_flag_as_required("target")
flags.DEFINE_boolean("test", False, "Calculate using test sets rather than training sets")


def pairs_ignore_order(len1, len2):
    """
    Compute indices of pairs from arrays 1 and 2 but don't add permutations of
    the orders

    For example, if len1 = 2 and len2 = 3, then we have indices [0,1] and
    [0,1,2]. All pairs are [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)], but we
    wish to ignore the permutation (1,0).

    This can be done by nested loops with the outer loop being the smaller of
    the two arrays and the inner being the larger. Then when iterating over the
    larger, we ignore all smaller entries than the outer loop index.

    However, we must keep track of which array index is which. We want array 1
    to end up with pair[0] and array 2 to be pair[1] so that these pair values
    are less than the len1 and len2 and are thus actually in the array.
    """
    pairs = []

    if len1 > len2:
        for i in range(len2):
            for j in range(i, len1):
                pairs.append((j, i))
    else:
        for i in range(len1):
            for j in range(i, len2):
                pairs.append((i, j))

    return pairs


def compute_distance(num_classes, source_labels, source_data, target_labels,
        target_data, take_sample=True):
    """ Compute average distance from pairs of data between the two domains on
    a per-class basis """
    number = 0
    total_dist = 0.0

    for class_num in range(0, num_classes):
        # Get only data for this class
        source_thisclass = np.argmax(source_labels, axis=1) == class_num
        source_data_thisclass = source_data[source_thisclass]

        target_thisclass = np.argmax(target_labels, axis=1) == class_num
        target_data_thisclass = target_data[target_thisclass]

        # For all possible pairings ignoring order
        pairs = pairs_ignore_order(len(source_data_thisclass), len(target_data_thisclass))

        # Random sample (so this doesn't take forever)
        if take_sample:
            num_samples = min(FLAGS.sample_size // num_classes, len(pairs))
            pairs = random.sample(pairs, num_samples)

        for pair in pairs:
            s = source_data_thisclass[pair[0]]
            t = target_data_thisclass[pair[1]]

            # See: https://github.com/pierre-rouanet/dtw
            dist, cost, acc_cost, path = dtw(s, t, dist=lambda x, y: norm(x - y, ord=1))

            number += 1
            total_dist += dist

    avg_dist = total_dist / number

    return avg_dist


def main(argv):
    # Don't bother using the GPU for this
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Get data
    source_dataset, target_dataset = datasets.load_da(FLAGS.source, FLAGS.target)

    if not FLAGS.test:
        source_data = source_dataset.train_data
        source_labels = source_dataset.train_labels
        target_data = target_dataset.train_data
        target_labels = target_dataset.train_labels
    else:
        source_data = source_dataset.test_data
        source_labels = source_dataset.test_labels
        target_data = target_dataset.test_data
        target_labels = target_dataset.test_labels

    num_classes = source_dataset.num_classes
    assert target_dataset.num_classes == num_classes, \
        "Currently don't support differing numbers of classes between domains"

    if FLAGS.normalize != "none":
        source_data = apply_normalization(source_data, calc_normalization(source_data, FLAGS.normalize))
        target_data = apply_normalization(target_data, calc_normalization(target_data, FLAGS.normalize))

    # Compute distance - mean +/- std
    distances = []

    for i in range(FLAGS.repeat):
        distances.append(compute_distance(num_classes, source_labels,
            source_data, target_labels, target_data))

    distances = np.array(distances, dtype=np.float32)

    print(distances.mean(), "+/-", distances.std(ddof=0))


if __name__ == "__main__":
    app.run(main)
