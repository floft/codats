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

# Hack to import from ../pool.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pool import run_job_pool
from tfrecord import write_tfrecord, tfrecord_filename

FLAGS = flags.FLAGS

flags.DEFINE_boolean("parallel", True, "Run multiple in parallel")
flags.DEFINE_integer("jobs", 0, "Parallel jobs (if parallel=True), 0 = # of CPU cores")


def write(filename, x, y):
    if x is not None and y is not None:
        if not os.path.exists(filename):
            write_tfrecord(filename, x, y)
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


def valid_split(images, labels, seed=None, validation_size=1000):
    """ Split training data into train/valid as is commonly done, taking 1000
    random (labeled, even if target domain) samples for a validation set """
    assert len(images) == len(labels), "len(images) != len(labels)"
    p = shuffle_together_calc(len(images), seed=seed)
    images = to_numpy(images)[p]
    labels = to_numpy(labels)[p]

    valid_images = images[:validation_size]
    valid_labels = labels[:validation_size]
    train_images = images[validation_size:]
    train_labels = labels[validation_size:]

    return valid_images, valid_labels, train_images, train_labels


def save_one(source, target, dataset_name, dataset, seed):
    """ Save single dataset """
    valid_images, valid_labels, \
        train_images, train_labels = \
        valid_split(dataset.train_images, dataset.train_labels,
            seed=seed)

    write(tfrecord_filename(source, target, dataset_name, "train"),
        train_images, train_labels)
    write(tfrecord_filename(source, target, dataset_name, "valid"),
        valid_images, valid_labels)
    write(tfrecord_filename(source, target, dataset_name, "test"),
        dataset.test_images, dataset.test_labels)


def save_adaptation(source, target, seed=0):
    """ Save single source-target pair datasets """
    print("Adaptation from", source, "to", target)

    source_dataset, target_dataset = datasets.load_da(source, target)

    save_one(source, target, source, source_dataset, seed=0)

    if target is not None:
        save_one(source, target, target, target_dataset, seed=1)


def main(argv):
    # Only list one direction since the other direction uses the same data
    adaptation_problems = [
        ("mnist", "usps"),
        ("svhn", "mnist"),
        ("svhn2", "mnist2"),
        ("mnist", "mnistm"),
        ("synnumbers", "svhn"),
        ("synsigns", "gtsrb"),
        # All combinations of these, so just make one file for each
        ("office_amazon", None),
        ("office_dslr", None),
        ("office_webcam", None),
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
