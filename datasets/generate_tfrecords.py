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
from sklearn.model_selection import train_test_split

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
    """ (Stratified) split training data into train/valid as is commonly done,
    taking 1000 random (stratified) (labeled, even if target domain) samples for
    a validation set """
    percentage_size = int(0.2*len(data))
    if percentage_size > validation_size:
        test_size = validation_size
    else:
        print("Warning: using smaller validation set size", percentage_size)
        test_size = 0.2  # 20% maximum

    x_train, x_valid, y_train, y_valid, domain_train, domain_valid = \
        train_test_split(data, labels, domain, test_size=test_size,
            stratify=labels, random_state=seed)

    return x_valid, y_valid, domain_valid, \
        x_train, y_train, domain_train


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
    # See pick_multi_source.py
    adaptation_problems = [
        ("ucihar_n13_0", None),
        ("ucihar_n13_1", None),
        ("ucihar_n13_10", None),
        ("ucihar_n13_11", None),
        ("ucihar_n13_12", None),
        ("ucihar_n13_13", None),
        ("ucihar_n13_14", None),
        ("ucihar_n13_2", None),
        ("ucihar_n13_3", None),
        ("ucihar_n13_4", None),
        ("ucihar_n13_5", None),
        ("ucihar_n13_6", None),
        ("ucihar_n13_7", None),
        ("ucihar_n13_8", None),
        ("ucihar_n13_9", None),
        ("ucihar_n19_0", None),
        ("ucihar_n19_1", None),
        ("ucihar_n19_10", None),
        ("ucihar_n19_11", None),
        ("ucihar_n19_12", None),
        ("ucihar_n19_13", None),
        ("ucihar_n19_14", None),
        ("ucihar_n19_2", None),
        ("ucihar_n19_3", None),
        ("ucihar_n19_4", None),
        ("ucihar_n19_5", None),
        ("ucihar_n19_6", None),
        ("ucihar_n19_7", None),
        ("ucihar_n19_8", None),
        ("ucihar_n19_9", None),
        ("ucihar_n1_0", None),
        ("ucihar_n1_1", None),
        ("ucihar_n1_2", None),
        ("ucihar_n1_3", None),
        ("ucihar_n1_4", None),
        ("ucihar_n1_5", None),
        ("ucihar_n1_6", None),
        ("ucihar_n1_7", None),
        ("ucihar_n1_8", None),
        ("ucihar_n1_9", None),
        ("ucihar_n25_0", None),
        ("ucihar_n25_1", None),
        ("ucihar_n25_10", None),
        ("ucihar_n25_11", None),
        ("ucihar_n25_12", None),
        ("ucihar_n25_13", None),
        ("ucihar_n25_14", None),
        ("ucihar_n25_2", None),
        ("ucihar_n25_3", None),
        ("ucihar_n25_4", None),
        ("ucihar_n25_5", None),
        ("ucihar_n25_6", None),
        ("ucihar_n25_7", None),
        ("ucihar_n25_8", None),
        ("ucihar_n25_9", None),
        ("ucihar_n7_0", None),
        ("ucihar_n7_1", None),
        ("ucihar_n7_10", None),
        ("ucihar_n7_11", None),
        ("ucihar_n7_12", None),
        ("ucihar_n7_13", None),
        ("ucihar_n7_14", None),
        ("ucihar_n7_2", None),
        ("ucihar_n7_3", None),
        ("ucihar_n7_4", None),
        ("ucihar_n7_5", None),
        ("ucihar_n7_6", None),
        ("ucihar_n7_7", None),
        ("ucihar_n7_8", None),
        ("ucihar_n7_9", None),
        ("ucihar_t1", None),
        ("ucihar_t2", None),
        ("ucihar_t3", None),
        ("ucihar_t4", None),
        ("ucihar_t5", None),
        ("ucihhar_n1_0", None),
        ("ucihhar_n1_1", None),
        ("ucihhar_n1_2", None),
        ("ucihhar_n1_3", None),
        ("ucihhar_n1_4", None),
        ("ucihhar_n1_5", None),
        ("ucihhar_n1_6", None),
        ("ucihhar_n2_0", None),
        ("ucihhar_n2_1", None),
        ("ucihhar_n2_10", None),
        ("ucihhar_n2_11", None),
        ("ucihhar_n2_2", None),
        ("ucihhar_n2_3", None),
        ("ucihhar_n2_4", None),
        ("ucihhar_n2_5", None),
        ("ucihhar_n2_6", None),
        ("ucihhar_n2_7", None),
        ("ucihhar_n2_8", None),
        ("ucihhar_n2_9", None),
        ("ucihhar_n3_0", None),
        ("ucihhar_n3_1", None),
        ("ucihhar_n3_10", None),
        ("ucihhar_n3_11", None),
        ("ucihhar_n3_12", None),
        ("ucihhar_n3_13", None),
        ("ucihhar_n3_2", None),
        ("ucihhar_n3_3", None),
        ("ucihhar_n3_4", None),
        ("ucihhar_n3_5", None),
        ("ucihhar_n3_6", None),
        ("ucihhar_n3_7", None),
        ("ucihhar_n3_8", None),
        ("ucihhar_n3_9", None),
        ("ucihhar_n4_0", None),
        ("ucihhar_n4_1", None),
        ("ucihhar_n4_10", None),
        ("ucihhar_n4_11", None),
        ("ucihhar_n4_12", None),
        ("ucihhar_n4_13", None),
        ("ucihhar_n4_14", None),
        ("ucihhar_n4_2", None),
        ("ucihhar_n4_3", None),
        ("ucihhar_n4_4", None),
        ("ucihhar_n4_5", None),
        ("ucihhar_n4_6", None),
        ("ucihhar_n4_7", None),
        ("ucihhar_n4_8", None),
        ("ucihhar_n4_9", None),
        ("ucihhar_n5_0", None),
        ("ucihhar_n5_1", None),
        ("ucihhar_n5_10", None),
        ("ucihhar_n5_11", None),
        ("ucihhar_n5_12", None),
        ("ucihhar_n5_13", None),
        ("ucihhar_n5_14", None),
        ("ucihhar_n5_2", None),
        ("ucihhar_n5_3", None),
        ("ucihhar_n5_4", None),
        ("ucihhar_n5_5", None),
        ("ucihhar_n5_6", None),
        ("ucihhar_n5_7", None),
        ("ucihhar_n5_8", None),
        ("ucihhar_n5_9", None),
        ("ucihhar_t0", None),
        ("ucihhar_t1", None),
        ("ucihhar_t2", None),
        ("ucihhar_t3", None),
        ("ucihhar_t4", None),
        ("uwave_n1_0", None),
        ("uwave_n1_1", None),
        ("uwave_n1_2", None),
        ("uwave_n1_3", None),
        ("uwave_n1_4", None),
        ("uwave_n1_5", None),
        ("uwave_n1_6", None),
        ("uwave_n1_7", None),
        ("uwave_n2_0", None),
        ("uwave_n2_1", None),
        ("uwave_n2_10", None),
        ("uwave_n2_11", None),
        ("uwave_n2_12", None),
        ("uwave_n2_13", None),
        ("uwave_n2_2", None),
        ("uwave_n2_3", None),
        ("uwave_n2_4", None),
        ("uwave_n2_5", None),
        ("uwave_n2_6", None),
        ("uwave_n2_7", None),
        ("uwave_n2_8", None),
        ("uwave_n2_9", None),
        ("uwave_n3_0", None),
        ("uwave_n3_1", None),
        ("uwave_n3_10", None),
        ("uwave_n3_11", None),
        ("uwave_n3_12", None),
        ("uwave_n3_13", None),
        ("uwave_n3_2", None),
        ("uwave_n3_3", None),
        ("uwave_n3_4", None),
        ("uwave_n3_5", None),
        ("uwave_n3_6", None),
        ("uwave_n3_7", None),
        ("uwave_n3_8", None),
        ("uwave_n3_9", None),
        ("uwave_n4_0", None),
        ("uwave_n4_1", None),
        ("uwave_n4_10", None),
        ("uwave_n4_11", None),
        ("uwave_n4_12", None),
        ("uwave_n4_13", None),
        ("uwave_n4_14", None),
        ("uwave_n4_2", None),
        ("uwave_n4_3", None),
        ("uwave_n4_4", None),
        ("uwave_n4_5", None),
        ("uwave_n4_6", None),
        ("uwave_n4_7", None),
        ("uwave_n4_8", None),
        ("uwave_n4_9", None),
        ("uwave_n5_0", None),
        ("uwave_n5_1", None),
        ("uwave_n5_10", None),
        ("uwave_n5_11", None),
        ("uwave_n5_12", None),
        ("uwave_n5_13", None),
        ("uwave_n5_2", None),
        ("uwave_n5_3", None),
        ("uwave_n5_4", None),
        ("uwave_n5_5", None),
        ("uwave_n5_6", None),
        ("uwave_n5_7", None),
        ("uwave_n5_8", None),
        ("uwave_n5_9", None),
        ("uwave_t1", None),
        ("uwave_t2", None),
        ("uwave_t3", None),
        ("uwave_t4", None),
        ("uwave_t5", None),
        ("wisdm_n117_0", None),
        ("wisdm_n117_1", None),
        ("wisdm_n117_10", None),
        ("wisdm_n117_11", None),
        ("wisdm_n117_12", None),
        ("wisdm_n117_13", None),
        ("wisdm_n117_14", None),
        ("wisdm_n117_2", None),
        ("wisdm_n117_3", None),
        ("wisdm_n117_4", None),
        ("wisdm_n117_5", None),
        ("wisdm_n117_6", None),
        ("wisdm_n117_7", None),
        ("wisdm_n117_8", None),
        ("wisdm_n117_9", None),
        ("wisdm_n1_0", None),
        ("wisdm_n1_1", None),
        ("wisdm_n1_10", None),
        ("wisdm_n1_11", None),
        ("wisdm_n1_12", None),
        ("wisdm_n1_13", None),
        ("wisdm_n1_14", None),
        ("wisdm_n1_2", None),
        ("wisdm_n1_3", None),
        ("wisdm_n1_4", None),
        ("wisdm_n1_5", None),
        ("wisdm_n1_6", None),
        ("wisdm_n1_7", None),
        ("wisdm_n1_8", None),
        ("wisdm_n1_9", None),
        ("wisdm_n30_0", None),
        ("wisdm_n30_1", None),
        ("wisdm_n30_10", None),
        ("wisdm_n30_11", None),
        ("wisdm_n30_12", None),
        ("wisdm_n30_13", None),
        ("wisdm_n30_14", None),
        ("wisdm_n30_2", None),
        ("wisdm_n30_3", None),
        ("wisdm_n30_4", None),
        ("wisdm_n30_5", None),
        ("wisdm_n30_6", None),
        ("wisdm_n30_7", None),
        ("wisdm_n30_8", None),
        ("wisdm_n30_9", None),
        ("wisdm_n59_0", None),
        ("wisdm_n59_1", None),
        ("wisdm_n59_10", None),
        ("wisdm_n59_11", None),
        ("wisdm_n59_12", None),
        ("wisdm_n59_13", None),
        ("wisdm_n59_14", None),
        ("wisdm_n59_2", None),
        ("wisdm_n59_3", None),
        ("wisdm_n59_4", None),
        ("wisdm_n59_5", None),
        ("wisdm_n59_6", None),
        ("wisdm_n59_7", None),
        ("wisdm_n59_8", None),
        ("wisdm_n59_9", None),
        ("wisdm_n88_0", None),
        ("wisdm_n88_1", None),
        ("wisdm_n88_10", None),
        ("wisdm_n88_11", None),
        ("wisdm_n88_12", None),
        ("wisdm_n88_13", None),
        ("wisdm_n88_14", None),
        ("wisdm_n88_2", None),
        ("wisdm_n88_3", None),
        ("wisdm_n88_4", None),
        ("wisdm_n88_5", None),
        ("wisdm_n88_6", None),
        ("wisdm_n88_7", None),
        ("wisdm_n88_8", None),
        ("wisdm_n88_9", None),
        ("wisdm_t0", None),
        ("wisdm_t1", None),
        ("wisdm_t2", None),
        ("wisdm_t3", None),
        ("wisdm_t4", None),
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
