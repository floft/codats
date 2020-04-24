#!/usr/bin/env python3
"""
Check the datasets statistics for each dataset (from .tfrecord files)

Note: sets CUDA_VISIBLE_DEVICES= so that it doesn't use the GPU.

Run something like the following to save the result:
    ./dataset_statistics.py | tee dataset_statistics.txt
"""
import os
import numpy as np

from absl import app
from absl import flags

from datasets import datasets
from load_datasets import load_da
from print_dictionary import print_dictionary

FLAGS = flags.FLAGS


def get_data(dataset):
    """ Get all the data in a dataset (i.e. must not use tf.data's repeat()) """
    xs = []
    ys = []

    for x, y in dataset:
        # Convert each modality/epoch to a numpy before adding
        xs.append(x.numpy())
        ys.append(y.numpy())

    # Stack each modality
    x = np.vstack(xs)
    y = np.hstack(ys)

    return x, y


def calc_individual_stats(x, only_total=False):
    original_shape = x.shape

    # Reshape from [num_examples, time_steps, num_features] to
    # [num_examples*time_steps, num_features]
    if len(x.shape) > 1:  # i.e. this is x, not y
        x = x.reshape(-1, x.shape[-1])

    # Then, compute statistics along axis 0, leaving us with an array of length
    # num_features for each
    results = {}

    if not only_total:
        results["min"] = np.min(x, axis=0)
        results["max"] = np.max(x, axis=0)
        results["mean"] = np.mean(x, axis=0)
        results["std"] = np.std(x, axis=0)

    results["total_min"] = np.min(x)
    results["total_max"] = np.max(x)
    results["total_mean"] = np.mean(x)
    results["total_std"] = np.std(x)
    results["shape"] = original_shape
    results["dtype"] = x.dtype

    return results


def calc_stats(data):
    x, y = data
    assert len(x) == len(y)

    results = {}

    results["x"] = calc_individual_stats(x)
    results["y"] = calc_individual_stats(y, only_total=True)

    return results


def print_stats(dataset_name, dataset, test=False):
    if test:
        test_name = "test"
    else:
        test_name = "valid"

    print_dictionary(calc_stats(get_data(dataset.train_evaluation)),
        dataset_name + "_" + "train")
    print_dictionary(calc_stats(get_data(dataset.test_evaluation)),
        dataset_name + "_" + test_name)

    print()


def main(argv):
    # Don't bother using the GPU for this
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    for dataset_name in datasets.list_datasets():
        for user in datasets.get_dataset_users(dataset_name):
            # Note: test=False so we only look at the training samples, where
            # train=80% of training set, test=20% of training set, i.e. the
            # validation set
            test = False

            sources, _ = load_da(dataset_name, str(user), "", test=test)
            assert len(sources) == 1
            dataset = sources[0]
            print_stats(dataset_name+"_"+str(user), dataset, test=test)


if __name__ == "__main__":
    app.run(main)
