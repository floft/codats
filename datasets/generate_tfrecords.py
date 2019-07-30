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
    """ Split training data into train/valid as is commonly done, taking 1000
    random (labeled, even if target domain) samples for a validation set """
    assert len(data) == len(labels), "len(data) != len(labels)"
    p = shuffle_together_calc(len(data), seed=seed)
    data = to_numpy(data)[p]
    labels = to_numpy(labels)[p]

    valid_data = data[:validation_size]
    valid_labels = labels[:validation_size]
    train_data = data[validation_size:]
    train_labels = labels[validation_size:]

    return valid_data, valid_labels, train_data, train_labels


def save_one(source, target, dataset_name, dataset, seed):
    """ Save single dataset """
    valid_data, valid_labels, \
        train_data, train_labels = \
        valid_split(dataset.train_data, dataset.train_labels,
            seed=seed)

    # Calculate normalization only on the training data
    if FLAGS.normalize != "none":
        normalization = calc_normalization(train_data, FLAGS.normalize)

        # Apply the normalization to the training, validation, and testing data
        train_data = apply_normalization(train_data, normalization)
        valid_data = apply_normalization(valid_data, normalization)
        test_data = apply_normalization(dataset.test_data, normalization)
    else:
        test_data = dataset.test_data

    # Saving
    write(tfrecord_filename(source, target, dataset_name, "train"),
        train_data, train_labels)
    write(tfrecord_filename(source, target, dataset_name, "valid"),
        valid_data, valid_labels)
    write(tfrecord_filename(source, target, dataset_name, "test"),
        test_data, dataset.test_labels)


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
        ("utdata_wrist", None),
        ("utdata_pocket", None),
        ("utdata_wrist", "utdata_pocket"),

        ("uwave_days_first", None),
        ("uwave_days_second", None),
        ("uwave_days_first", "uwave_days_second"),
        ("uwave_users_first", None),
        ("uwave_users_second", None),
        ("uwave_users_first", "uwave_users_second"),

        # ("positive_slope", "positive_slope_low"),
        # ("positive_slope", "positive_slope_noise"),
        # ("positive_sine", "positive_sine_low"),
        # ("positive_sine", "positive_sine_noise"),

        # ("freq_low", "freq_high"),
        # ("freq_low", "freq_low_amp_noise"),
        # ("freq_low", "freq_low_freq_noise"),
        # ("freq_low", "freq_low_freqamp_noise"),
        # ("freq_high", "freq_high_amp_noise"),
        # ("freq_high", "freq_high_freq_noise"),
        # ("freq_high", "freq_high_freqamp_noise"),
        # ("freq_low", "freq_high_freqamp_noise"),
        # ("freq_high", "freq_low_freqamp_noise"),

        # ("freqshift_low", "freqshift_high"),
        # ("freqscale_low", "freqscale_high"),

        # ("line1low", "line1high"),
        # ("line2low", "line2high"),
        # ("sine1low", "sine1high"),
        # ("sine2low", "sine2high"),
        # ("sine3low", "sine3high"),
        # ("sine4low", "sine4high"),

        # ("lineslope1low", "lineslope1high"),
        # ("lineslope2low", "lineslope2high"),
        # ("sineslope1low", "sineslope1high"),
        # ("sineslope2low", "sineslope2high"),
        # ("sineslope3low", "sineslope3high"),
        # ("sineslope4low", "sineslope4high"),

        # ("freqshift_b0", None),
        # ("freqshift_b1", None),
        # ("freqshift_b2", None),
        # ("freqshift_b3", None),
        # ("freqshift_b4", None),
        # ("freqshift_b5", None),
        # ("freqshift_a", "freqshift_b0"),
        # ("freqshift_a", "freqshift_b1"),
        # ("freqshift_a", "freqshift_b2"),
        # ("freqshift_a", "freqshift_b3"),
        # ("freqshift_a", "freqshift_b4"),
        # ("freqshift_a", "freqshift_b5"),

        ("freqshift_phase_b0", None),
        ("freqshift_phase_b1", None),
        ("freqshift_phase_b2", None),
        ("freqshift_phase_b3", None),
        ("freqshift_phase_b4", None),
        ("freqshift_phase_b5", None),
        ("freqshift_phase_b6", None),
        ("freqshift_phase_b7", None),
        ("freqshift_phase_b8", None),
        ("freqshift_phase_b9", None),
        ("freqshift_phase_b10", None),
        ("freqshift_phase_a", "freqshift_phase_b0"),
        ("freqshift_phase_a", "freqshift_phase_b1"),
        ("freqshift_phase_a", "freqshift_phase_b2"),
        ("freqshift_phase_a", "freqshift_phase_b3"),
        ("freqshift_phase_a", "freqshift_phase_b4"),
        ("freqshift_phase_a", "freqshift_phase_b5"),
        ("freqshift_phase_a", "freqshift_phase_b6"),
        ("freqshift_phase_a", "freqshift_phase_b7"),
        ("freqshift_phase_a", "freqshift_phase_b8"),
        ("freqshift_phase_a", "freqshift_phase_b9"),
        ("freqshift_phase_a", "freqshift_phase_b10"),

        # ("freqscale_b0", None),
        # ("freqscale_b1", None),
        # ("freqscale_b2", None),
        # ("freqscale_b3", None),
        # ("freqscale_b4", None),
        # ("freqscale_b5", None),
        # ("freqscale_a", "freqscale_b0"),
        # ("freqscale_a", "freqscale_b1"),
        # ("freqscale_a", "freqscale_b2"),
        # ("freqscale_a", "freqscale_b3"),
        # ("freqscale_a", "freqscale_b4"),
        # ("freqscale_a", "freqscale_b5"),

        ("freqscale_phase_b0", None),
        ("freqscale_phase_b1", None),
        ("freqscale_phase_b2", None),
        ("freqscale_phase_b3", None),
        ("freqscale_phase_b4", None),
        ("freqscale_phase_b5", None),
        ("freqscale_phase_b6", None),
        ("freqscale_phase_b7", None),
        ("freqscale_phase_b8", None),
        ("freqscale_phase_b9", None),
        ("freqscale_phase_b10", None),
        ("freqscale_phase_a", "freqscale_phase_b0"),
        ("freqscale_phase_a", "freqscale_phase_b1"),
        ("freqscale_phase_a", "freqscale_phase_b2"),
        ("freqscale_phase_a", "freqscale_phase_b3"),
        ("freqscale_phase_a", "freqscale_phase_b4"),
        ("freqscale_phase_a", "freqscale_phase_b5"),
        ("freqscale_phase_a", "freqscale_phase_b6"),
        ("freqscale_phase_a", "freqscale_phase_b7"),
        ("freqscale_phase_a", "freqscale_phase_b8"),
        ("freqscale_phase_a", "freqscale_phase_b9"),
        ("freqscale_phase_a", "freqscale_phase_b10"),

        # ("jumpmean_b0", None),
        # ("jumpmean_b1", None),
        # ("jumpmean_b2", None),
        # ("jumpmean_b3", None),
        # ("jumpmean_b4", None),
        # ("jumpmean_b5", None),
        # ("jumpmean_a", "jumpmean_b0"),
        # ("jumpmean_a", "jumpmean_b1"),
        # ("jumpmean_a", "jumpmean_b2"),
        # ("jumpmean_a", "jumpmean_b3"),
        # ("jumpmean_a", "jumpmean_b4"),
        # ("jumpmean_a", "jumpmean_b5"),

        # ("jumpmean_phase_b0", None),
        # ("jumpmean_phase_b1", None),
        # ("jumpmean_phase_b2", None),
        # ("jumpmean_phase_b3", None),
        # ("jumpmean_phase_b4", None),
        # ("jumpmean_phase_b5", None),
        # ("jumpmean_phase_a", "jumpmean_phase_b0"),
        # ("jumpmean_phase_a", "jumpmean_phase_b1"),
        # ("jumpmean_phase_a", "jumpmean_phase_b2"),
        # ("jumpmean_phase_a", "jumpmean_phase_b3"),
        # ("jumpmean_phase_a", "jumpmean_phase_b4"),
        # ("jumpmean_phase_a", "jumpmean_phase_b5"),
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
