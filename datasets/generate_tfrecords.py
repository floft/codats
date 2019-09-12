#!/usr/bin/env python3
"""
Create all the tfrecord files

For some source domain A and target domain B, where C and D are A and B but in
alphabetical order:
    C_and_D_A_{train,valid,test}.tfrecord
    C_and_D_B_{train,valid,test}.tfrecord

For example for MNIST to MNIST-M or MNIST-M to MNIST (since both ways use the
same data):
    mnist_and_mnistm_mnist_{train,valid,test}.tfrecord
    mnist_and_mnistm_mnistm_{train,valid,test}.tfrecord

We do this because otherwise for some domains like SynNumbers to SVHN we use
nearly all of my 32 GiB of RAM just loading the datasets and it takes a while
as well.

Note: probably want to run this prefixed with CUDA_VISIBLE_DEVICES= so that it
doesn't use the GPU (if you're running other jobs). Does this by default if
parallel=True since otherwise it'll error.
"""
import os
import sys
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

import datasets

from datasets import calc_normalization, apply_normalization

# Hack to import from ../pool.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pool import run_job_pool
from tfrecord import write_tfrecord, tfrecord_filename_simple

FLAGS = flags.FLAGS

flags.DEFINE_boolean("parallel", True, "Run multiple in parallel")
flags.DEFINE_integer("jobs", 0, "Parallel jobs (if parallel=True), 0 = # of CPU cores")


def write(filename, x, y, domain):
    if x is not None and y is not None and domain is not None:
        if not os.path.exists(filename):
            write_tfrecord(filename, x, y, domain)
        else:
            print("Skipping:", filename, "(already exists)")
    else:
        print("Skipping:", filename, "(no data)")


def shuffle_together_calc(length, seed=None):
    """ Generate indices of numpy array shuffling, then do x[p] """
    rand = np.random.RandomState(seed)
    p = rand.permutation(length)
    return p


def to_numpy(value):
    """ Make sure value is numpy array """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return value


def valid_split(data, labels, domain, seed=None, validation_size=1000):
    """ Split training data into train/valid as is commonly done, taking 1000
    random (labeled, even if target domain) samples for a validation set """
    assert len(data) == len(labels), "len(data) != len(labels)"
    p = shuffle_together_calc(len(data), seed=seed)
    data = to_numpy(data)[p]
    labels = to_numpy(labels)[p]
    domain = to_numpy(domain)[p]

    # If data to small then 1000 examples is too much, so shrink validation size
    # to max of 20%
    orig_valid_size = validation_size
    validation_size = min(validation_size, int(0.2*len(data)))
    if orig_valid_size != validation_size:
        print("Warning: using smaller validation set size:", validation_size)

    valid_data = data[:validation_size]
    valid_labels = labels[:validation_size]
    valid_domain = domain[:validation_size]
    train_data = data[validation_size:]
    train_labels = labels[validation_size:]
    train_domain = domain[validation_size:]

    return valid_data, valid_labels, valid_domain, \
        train_data, train_labels, train_domain


def save_one(source, target, dataset_name, dataset, seed,
        already_normalized=False):
    """ Save single dataset """
    valid_data, valid_labels, valid_domain, \
        train_data, train_labels, train_domain = \
        valid_split(dataset.train_data, dataset.train_labels,
            dataset.train_domain, seed=seed)

    # Calculate normalization only on the training data
    if FLAGS.normalize != "none" and not already_normalized:
        normalization = calc_normalization(train_data, FLAGS.normalize)

        # Apply the normalization to the training, validation, and testing data
        train_data = apply_normalization(train_data, normalization)
        valid_data = apply_normalization(valid_data, normalization)
        test_data = apply_normalization(dataset.test_data, normalization)
    else:
        test_data = dataset.test_data

    # Saving
    write(tfrecord_filename_simple(dataset_name, "train"),
        train_data, train_labels, train_domain)
    write(tfrecord_filename_simple(dataset_name, "valid"),
        valid_data, valid_labels, valid_domain)
    write(tfrecord_filename_simple(dataset_name, "test"),
        test_data, dataset.test_labels, dataset.test_domain)


def save_adaptation(source, target, seed=0):
    """ Save single source-target pair datasets """
    print("Adaptation from", source, "to", target)

    source_dataset, target_dataset = datasets.load_da(source, target)

    already_normalized = False

    # UCI HAR datasets already normalized and bounded
    if "ucihar" in source:
        already_normalized = True

    save_one(source, target, source, source_dataset, seed=0,
        already_normalized=already_normalized)

    if target is not None:
        save_one(source, target, target, target_dataset, seed=1,
            already_normalized=already_normalized)


def main(argv):
    # Only list one direction since the other direction uses the same data
    adaptation_problems = [
        ("ucihar_1,2,3", "ucihar_t4"),
        ("uwave_1,2,3", "uwave_t4"),
        ("sleep_0,1,2", "sleep_t3"),
    ]

    # Save tfrecord files for each of the adaptation problems
    if FLAGS.parallel:
        # TensorFlow will error from all processes trying to use ~90% of the
        # GPU memory on all parallel jobs, which will fail, so do this on the
        # CPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if FLAGS.jobs == 0:
            cores = None
        else:
            cores = FLAGS.jobs

        run_job_pool(save_adaptation, adaptation_problems, cores=cores)
    else:
        for source, target in adaptation_problems:
            save_adaptation(source, target)


if __name__ == "__main__":
    app.run(main)
