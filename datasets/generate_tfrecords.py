#!/usr/bin/env python3
"""
Create all the tfrecord files

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
from sklearn.model_selection import train_test_split

import datasets

from datasets import calc_normalization, apply_normalization

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


def valid_split(data, labels, seed=None, validation_size=1000):
    """ (Stratified) split training data into train/valid as is commonly done,
    taking 1000 random (stratified) (labeled, even if target domain) samples for
    a validation set """
    percentage_size = int(0.2*len(data))
    if percentage_size > validation_size:
        test_size = validation_size
    else:
        print("Warning: using smaller validation set size", percentage_size)
        test_size = 0.2  # 20% maximum

    x_train, x_valid, y_train, y_valid = \
        train_test_split(data, labels, test_size=test_size,
            stratify=labels, random_state=seed)

    return x_valid, y_valid, x_train, y_train


def save_dataset(dataset_name, seed=0):
    """ Save single dataset """
    print("Saving dataset", dataset_name)
    dataset = datasets.load(dataset_name)

    # UCI HAR datasets already normalized and bounded
    already_normalized = False

    if "ucihar" in dataset_name:
        already_normalized = True

    # Split into training/valid datasets
    valid_data, valid_labels, train_data, train_labels = \
        valid_split(dataset.train_data, dataset.train_labels, seed=seed)

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
    write(tfrecord_filename(dataset_name, "train"), train_data, train_labels)
    write(tfrecord_filename(dataset_name, "valid"), valid_data, valid_labels)
    write(tfrecord_filename(dataset_name, "test"), test_data, dataset.test_labels)


def main(argv):
    # See pick_multi_source.py
    adaptation_problems = [
        "ucihar_1",
        "ucihar_2",
        "ucihar_3",
        "ucihar_4",
        "ucihar_5",
        "ucihar_6",
        "ucihar_7",
        "ucihar_8",
        "ucihar_9",
        "ucihar_10",
        "ucihar_11",
        "ucihar_12",
        "ucihar_13",
        "ucihar_14",
        "ucihar_15",
        "ucihar_16",
        "ucihar_17",
        "ucihar_18",
        "ucihar_19",
        "ucihar_20",
        "ucihar_21",
        "ucihar_22",
        "ucihar_23",
        "ucihar_24",
        "ucihar_25",
        "ucihar_26",
        "ucihar_27",
        "ucihar_28",
        "ucihar_29",
        "ucihar_30",
        "uwave_1",
        "uwave_2",
        "uwave_3",
        "uwave_4",
        "uwave_5",
        "uwave_6",
        "uwave_7",
        "uwave_8",
        "ucihhar_0",
        "ucihhar_1",
        "ucihhar_2",
        "ucihhar_3",
        "ucihhar_4",
        "ucihhar_5",
        "ucihhar_6",
        "ucihhar_7",
        "ucihhar_8",
        "wisdm_0",
        "wisdm_1",
        "wisdm_2",
        "wisdm_3",
        "wisdm_4",
        "wisdm_5",
        "wisdm_6",
        "wisdm_7",
        "wisdm_8",
        "wisdm_9",
        "wisdm_10",
        "wisdm_11",
        "wisdm_12",
        "wisdm_13",
        "wisdm_14",
        "wisdm_15",
        "wisdm_16",
        "wisdm_17",
        "wisdm_18",
        "wisdm_19",
        "wisdm_20",
        "wisdm_21",
        "wisdm_22",
        "wisdm_23",
        "wisdm_24",
        "wisdm_25",
        "wisdm_26",
        "wisdm_27",
        "wisdm_28",
        "wisdm_29",
        "wisdm_30",
        "wisdm_31",
        "wisdm_32",
        "wisdm_33",
        "wisdm_34",
        "wisdm_35",
        "wisdm_36",
        "wisdm_37",
        "wisdm_38",
        "wisdm_39",
        "wisdm_40",
        "wisdm_41",
        "wisdm_42",
        "wisdm_43",
        "wisdm_44",
        "wisdm_45",
        "wisdm_46",
        "wisdm_47",
        "wisdm_48",
        "wisdm_49",
        "wisdm_50",
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

        run_job_pool(save_dataset, [(d,) for d in adaptation_problems], cores=cores)
    else:
        for dataset_name in adaptation_problems:
            save_dataset(dataset_name)


if __name__ == "__main__":
    app.run(main)
