#!/usr/bin/env python3
"""
Process each dataset into .tfrecord files

Run (or see ../generate_tfrecords.sh script):

    python -m datasets.main <args>

Note: probably want to run this prefixed with CUDA_VISIBLE_DEVICES= so that it
doesn't use the GPU (if you're running other jobs). Does this by default if
parallel=True since otherwise it'll error.
"""
import os
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from sklearn.model_selection import train_test_split

from datasets import datasets
from pool import run_job_pool
from datasets.tfrecord import write_tfrecord, tfrecord_filename

FLAGS = flags.FLAGS

flags.DEFINE_boolean("parallel", True, "Run multiple in parallel")
flags.DEFINE_integer("jobs", 0, "Parallel jobs (if parallel=True), 0 = # of CPU cores")
flags.DEFINE_boolean("debug", False, "Whether to print debug information")


def write(filename, x, y):
    if x is not None and y is not None:
        if not os.path.exists(filename):
            write_tfrecord(filename, x, y)
        elif FLAGS.debug:
            print("Skipping:", filename, "(already exists)")
    elif FLAGS.debug:
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


def valid_split(data, labels, seed=None, validation_size=1000):
    """ (Stratified) split training data into train/valid as is commonly done,
    taking 1000 random (stratified) (labeled, even if target domain) samples for
    a validation set """
    percentage_size = int(0.2*len(data))
    if percentage_size > validation_size:
        test_size = validation_size
    else:
        if FLAGS.debug:
            print("Warning: using smaller validation set size", percentage_size)
        test_size = 0.2  # 20% maximum

    x_train, x_valid, y_train, y_valid = \
        train_test_split(data, labels, test_size=test_size,
            stratify=labels, random_state=seed)

    return x_valid, y_valid, x_train, y_train


def save_dataset(dataset_name, output_dir, seed=0):
    """ Save single dataset """
    train_filename = os.path.join(output_dir,
        tfrecord_filename(dataset_name, "train"))
    valid_filename = os.path.join(output_dir,
        tfrecord_filename(dataset_name, "valid"))
    test_filename = os.path.join(output_dir,
        tfrecord_filename(dataset_name, "test"))

    # Skip if they already exist
    if os.path.exists(train_filename) \
            and os.path.exists(valid_filename) \
            and os.path.exists(test_filename):
        if FLAGS.debug:
            print("Skipping:", train_filename, valid_filename, test_filename,
               "already exist")
        return

    if FLAGS.debug:
        print("Saving dataset", dataset_name)
    dataset, dataset_class = datasets.load(dataset_name)

    # Skip if already normalized/bounded, e.g. UCI HAR datasets
    already_normalized = dataset_class.already_normalized

    # Split into training/valid datasets
    valid_data, valid_labels, train_data, train_labels = \
        valid_split(dataset.train_data, dataset.train_labels, seed=seed)

    # Calculate normalization only on the training data
    if FLAGS.normalize != "none" and not already_normalized:
        normalization = datasets.calc_normalization(train_data, FLAGS.normalize)

        # Apply the normalization to the training, validation, and testing data
        train_data = datasets.apply_normalization(train_data, normalization)
        valid_data = datasets.apply_normalization(valid_data, normalization)
        test_data = datasets.apply_normalization(dataset.test_data, normalization)
    else:
        test_data = dataset.test_data

    # Saving
    write(train_filename, train_data, train_labels)
    write(valid_filename, valid_data, valid_labels)
    write(test_filename, test_data, dataset.test_labels)


def main(argv):
    output_dir = os.path.join("datasets", "tfrecords")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all possible datasets we can generate
    adaptation_problems = datasets.names()

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

        run_job_pool(save_dataset,
            [(d, output_dir) for d in adaptation_problems], cores=cores)
    else:
        for dataset_name in adaptation_problems:
            save_dataset(dataset_name, output_dir)


if __name__ == "__main__":
    app.run(main)
