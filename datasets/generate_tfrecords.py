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
        ("ucihar_n13_1,2,4,6,7,9,10,12,15,21,22,23,27", None),
        ("ucihar_n13_1,4,5,6,8,11,13,15,17,18,20,24,26", None),
        ("ucihar_n13_1,4,5,7,8,11,12,16,18,22,23,26,28", None),
        ("ucihar_n13_1,5,6,10,11,12,13,14,15,16,22,27,28", None),
        ("ucihar_n13_1,5,7,11,13,15,16,20,21,23,26,27,28", None),
        ("ucihar_n13_1,7,10,11,13,15,17,18,24,25,26,28,30", None),
        ("ucihar_n13_2,4,6,8,10,14,15,16,22,23,27,28,29", None),
        ("ucihar_n13_2,4,7,8,13,15,17,18,19,20,25,26,29", None),
        ("ucihar_n13_2,5,6,7,10,12,14,16,17,18,23,25,30", None),
        ("ucihar_n13_3,4,6,8,13,14,16,17,18,20,21,22,24", None),
        ("ucihar_n13_3,5,6,9,12,18,19,20,21,23,24,27,29", None),
        ("ucihar_n13_3,8,9,10,12,13,14,15,20,23,26,29,30", None),
        ("ucihar_n13_4,8,11,14,16,19,20,21,24,25,26,29,30", None),
        ("ucihar_n13_4,9,11,13,14,16,17,18,21,23,25,28,30", None),
        ("ucihar_n13_7,8,11,12,13,14,16,17,20,21,23,24,26", None),
        ("ucihar_n19_1,2,3,4,7,8,13,15,16,17,18,19,20,21,25,26,27,28,29", None),
        ("ucihar_n19_1,2,4,5,6,7,9,10,12,15,16,17,18,20,21,22,23,27,29", None),
        ("ucihar_n19_1,3,4,5,6,8,11,12,13,15,17,18,20,21,23,24,26,27,29", None),
        ("ucihar_n19_1,3,4,5,7,8,11,12,13,16,17,18,21,22,23,26,28,29,30", None),
        ("ucihar_n19_1,3,6,7,10,11,13,15,16,17,18,19,20,21,24,25,26,28,30", None),
        ("ucihar_n19_1,5,6,7,11,13,14,15,16,19,20,21,22,23,24,25,26,27,28", None),
        ("ucihar_n19_1,5,6,8,9,10,11,12,13,14,15,16,19,21,22,23,24,27,28", None),
        ("ucihar_n19_2,3,4,6,8,9,10,12,13,14,15,17,20,23,25,26,27,29,30", None),
        ("ucihar_n19_2,4,5,6,8,9,10,11,13,14,15,16,20,22,23,24,27,28,29", None),
        ("ucihar_n19_2,5,6,7,9,10,12,14,15,16,17,18,23,24,25,26,27,29,30", None),
        ("ucihar_n19_3,4,5,6,8,9,11,13,14,16,17,18,19,21,22,23,25,28,30", None),
        ("ucihar_n19_3,4,6,7,8,9,11,13,14,15,16,17,18,20,21,22,24,26,28", None),
        ("ucihar_n19_3,4,7,8,11,12,13,14,15,16,17,18,20,21,23,24,26,28,30", None),
        ("ucihar_n19_3,5,6,9,10,11,12,13,16,18,19,20,21,22,23,24,26,27,29", None),
        ("ucihar_n19_4,6,7,8,9,11,12,13,14,16,19,20,21,23,24,25,26,29,30", None),
        ("ucihar_n1_10", None),
        ("ucihar_n1_12", None),
        ("ucihar_n1_16", None),
        ("ucihar_n1_17", None),
        ("ucihar_n1_20", None),
        ("ucihar_n1_21", None),
        ("ucihar_n1_23", None),
        ("ucihar_n1_26", None),
        ("ucihar_n1_27", None),
        ("ucihar_n1_4", None),
        ("ucihar_n25_1,2,3,4,6,7,8,10,11,13,14,15,16,17,18,19,20,21,22,25,26,27,28,29,30", None),
        ("ucihar_n25_1,2,3,4,6,8,9,10,11,12,13,14,15,17,18,20,21,22,23,24,25,26,27,29,30", None),
        ("ucihar_n25_1,2,3,5,6,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24,25,26,27,29,30", None),
        ("ucihar_n25_1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,26,27,28,30", None),
        ("ucihar_n25_1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,27,28,29,30", None),
        ("ucihar_n25_1,2,4,5,6,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,27,28,29", None),
        ("ucihar_n25_1,3,4,5,6,8,10,11,12,13,14,15,16,17,18,20,21,22,23,24,26,27,28,29,30", None),
        ("ucihar_n25_1,3,4,5,7,8,10,11,12,13,14,15,16,17,18,20,21,22,23,24,26,27,28,29,30", None),
        ("ucihar_n25_1,3,4,6,7,8,9,11,12,13,14,15,16,19,20,21,22,23,24,25,26,27,28,29,30", None),
        ("ucihar_n25_1,3,5,6,7,8,9,10,11,12,13,14,15,16,18,19,21,22,23,24,25,26,27,28,29", None),
        ("ucihar_n25_1,5,6,7,8,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30", None),
        ("ucihar_n25_2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,20,21,23,24,25,26,27,29,30", None),
        ("ucihar_n25_3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,24,25,26,27,28,30", None),
        ("ucihar_n25_3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,28,29,30", None),
        ("ucihar_n25_3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,26,27,28,29,30", None),
        ("ucihar_n7_1,4,13,18,20,24,26", None),
        ("ucihar_n7_1,7,10,15,18,24,25", None),
        ("ucihar_n7_1,7,15,20,23,26,27", None),
        ("ucihar_n7_2,5,10,12,14,16,25", None),
        ("ucihar_n7_4,11,19,20,24,26,29", None),
        ("ucihar_n7_4,6,9,12,21,22,27", None),
        ("ucihar_n7_4,9,14,17,18,21,23", None),
        ("ucihar_n7_5,7,11,18,23,26,28", None),
        ("ucihar_n7_6,10,13,16,22,27,28", None),
        ("ucihar_n7_6,18,20,21,23,24,27", None),
        ("ucihar_n7_6,8,10,14,15,16,29", None),
        ("ucihar_n7_7,8,12,16,20,23,24", None),
        ("ucihar_n7_7,8,17,18,19,20,25", None),
        ("ucihar_n7_8,16,17,18,20,21,24", None),
        ("ucihar_n7_8,9,14,20,23,26,30", None),
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
        ("ucihhar_n2_0,3", None),
        ("ucihhar_n2_0,5", None),
        ("ucihhar_n2_0,8", None),
        ("ucihhar_n2_1,4", None),
        ("ucihhar_n2_2,3", None),
        ("ucihhar_n2_3,5", None),
        ("ucihhar_n2_3,6", None),
        ("ucihhar_n2_4,5", None),
        ("ucihhar_n2_4,6", None),
        ("ucihhar_n2_4,8", None),
        ("ucihhar_n2_5,6", None),
        ("ucihhar_n2_6,8", None),
        ("ucihhar_n3_0,1,4", None),
        ("ucihhar_n3_0,2,3", None),
        ("ucihhar_n3_0,4,5", None),
        ("ucihhar_n3_0,4,8", None),
        ("ucihhar_n3_0,7,8", None),
        ("ucihhar_n3_1,5,6", None),
        ("ucihhar_n3_3,4,6", None),
        ("ucihhar_n3_3,4,8", None),
        ("ucihhar_n3_3,5,8", None),
        ("ucihhar_n3_3,6,8", None),
        ("ucihhar_n3_4,5,7", None),
        ("ucihhar_n3_4,6,7", None),
        ("ucihhar_n3_5,6,7", None),
        ("ucihhar_n3_6,7,8", None),
        ("ucihhar_n4_0,1,2,3", None),
        ("ucihhar_n4_0,1,4,8", None),
        ("ucihhar_n4_0,1,7,8", None),
        ("ucihhar_n4_0,2,3,8", None),
        ("ucihhar_n4_0,4,5,6", None),
        ("ucihhar_n4_0,4,7,8", None),
        ("ucihhar_n4_1,2,5,6", None),
        ("ucihhar_n4_1,3,4,8", None),
        ("ucihhar_n4_1,3,5,8", None),
        ("ucihhar_n4_3,4,5,6", None),
        ("ucihhar_n4_3,4,6,8", None),
        ("ucihhar_n4_3,5,6,7", None),
        ("ucihhar_n4_3,6,7,8", None),
        ("ucihhar_n4_4,5,6,7", None),
        ("ucihhar_n4_4,5,7,8", None),
        ("ucihhar_n5_0,1,2,3,8", None),
        ("ucihhar_n5_0,1,2,7,8", None),
        ("ucihhar_n5_0,1,4,5,6", None),
        ("ucihhar_n5_0,1,4,5,8", None),
        ("ucihhar_n5_0,1,4,7,8", None),
        ("ucihhar_n5_0,2,3,5,8", None),
        ("ucihhar_n5_0,4,5,6,7", None),
        ("ucihhar_n5_1,2,5,6,8", None),
        ("ucihhar_n5_1,3,4,5,8", None),
        ("ucihhar_n5_1,3,5,7,8", None),
        ("ucihhar_n5_2,3,4,5,6", None),
        ("ucihhar_n5_2,3,4,6,8", None),
        ("ucihhar_n5_2,3,5,6,7", None),
        ("ucihhar_n5_3,4,5,7,8", None),
        ("ucihhar_n5_3,5,6,7,8", None),
        ("ucihhar_t0", None),
        ("ucihhar_t1", None),
        ("ucihhar_t2", None),
        ("ucihhar_t3", None),
        ("ucihhar_t4", None),
        ("ucihm_n1_0", None),
        ("ucihm_n1_2", None),
        ("ucihm_n1_3", None),
        ("ucihm_n1_4", None),
        ("ucihm_n1_5", None),
        ("ucihm_n2_0,1", None),
        ("ucihm_n2_0,3", None),
        ("ucihm_n2_0,5", None),
        ("ucihm_n2_1,4", None),
        ("ucihm_n2_2,3", None),
        ("ucihm_n2_2,4", None),
        ("ucihm_n2_2,5", None),
        ("ucihm_n2_3,4", None),
        ("ucihm_n2_3,5", None),
        ("ucihm_n2_4,5", None),
        ("ucihm_n3_0,1,4", None),
        ("ucihm_n3_0,1,5", None),
        ("ucihm_n3_0,2,3", None),
        ("ucihm_n3_0,2,4", None),
        ("ucihm_n3_0,2,5", None),
        ("ucihm_n3_0,3,5", None),
        ("ucihm_n3_0,4,5", None),
        ("ucihm_n3_1,2,4", None),
        ("ucihm_n3_1,3,4", None),
        ("ucihm_n3_1,3,5", None),
        ("ucihm_n3_2,3,4", None),
        ("ucihm_n3_2,3,5", None),
        ("ucihm_n3_3,4,5", None),
        ("ucihm_n4_0,1,2,4", None),
        ("ucihm_n4_0,1,2,5", None),
        ("ucihm_n4_0,1,3,4", None),
        ("ucihm_n4_0,1,3,5", None),
        ("ucihm_n4_0,2,3,4", None),
        ("ucihm_n4_0,2,3,5", None),
        ("ucihm_n4_0,3,4,5", None),
        ("ucihm_n4_1,2,3,4", None),
        ("ucihm_n4_1,2,4,5", None),
        ("ucihm_n4_1,3,4,5", None),
        ("ucihm_n4_2,3,4,5", None),
        ("ucihm_n5_0,1,2,3,5", None),
        ("ucihm_n5_0,1,2,4,5", None),
        ("ucihm_n5_0,1,3,4,5", None),
        ("ucihm_n5_0,2,3,4,5", None),
        ("ucihm_n5_1,2,3,4,5", None),
        ("ucihm_t0", None),
        ("ucihm_t1", None),
        ("ucihm_t2", None),
        ("ucihm_t3", None),
        ("ucihm_t4", None),
        ("uwave_n1_1", None),
        ("uwave_n1_2", None),
        ("uwave_n1_3", None),
        ("uwave_n1_4", None),
        ("uwave_n1_5", None),
        ("uwave_n1_6", None),
        ("uwave_n1_7", None),
        ("uwave_n1_8", None),
        ("uwave_n2_1,2", None),
        ("uwave_n2_1,5", None),
        ("uwave_n2_1,7", None),
        ("uwave_n2_1,8", None),
        ("uwave_n2_3,5", None),
        ("uwave_n2_3,6", None),
        ("uwave_n2_4,5", None),
        ("uwave_n2_4,6", None),
        ("uwave_n2_4,7", None),
        ("uwave_n2_4,8", None),
        ("uwave_n2_5,6", None),
        ("uwave_n2_5,8", None),
        ("uwave_n2_6,7", None),
        ("uwave_n2_6,8", None),
        ("uwave_n3_1,2,5", None),
        ("uwave_n3_1,4,7", None),
        ("uwave_n3_1,5,6", None),
        ("uwave_n3_1,5,7", None),
        ("uwave_n3_1,6,8", None),
        ("uwave_n3_2,4,6", None),
        ("uwave_n3_2,6,7", None),
        ("uwave_n3_3,5,6", None),
        ("uwave_n3_3,6,7", None),
        ("uwave_n3_4,5,6", None),
        ("uwave_n3_4,5,8", None),
        ("uwave_n3_4,6,8", None),
        ("uwave_n3_5,7,8", None),
        ("uwave_n3_6,7,8", None),
        ("uwave_n4_1,2,5,7", None),
        ("uwave_n4_1,3,6,8", None),
        ("uwave_n4_1,5,6,7", None),
        ("uwave_n4_1,5,6,8", None),
        ("uwave_n4_2,4,5,6", None),
        ("uwave_n4_2,4,5,7", None),
        ("uwave_n4_2,4,6,7", None),
        ("uwave_n4_2,4,6,8", None),
        ("uwave_n4_2,5,6,8", None),
        ("uwave_n4_2,6,7,8", None),
        ("uwave_n4_3,4,5,6", None),
        ("uwave_n4_3,4,6,7", None),
        ("uwave_n4_3,5,6,7", None),
        ("uwave_n4_3,5,7,8", None),
        ("uwave_n4_4,5,6,8", None),
        ("uwave_n5_1,2,4,6,8", None),
        ("uwave_n5_1,2,5,6,8", None),
        ("uwave_n5_1,2,5,7,8", None),
        ("uwave_n5_1,2,6,7,8", None),
        ("uwave_n5_1,3,5,6,7", None),
        ("uwave_n5_1,3,5,7,8", None),
        ("uwave_n5_1,3,6,7,8", None),
        ("uwave_n5_1,4,5,7,8", None),
        ("uwave_n5_2,3,5,6,8", None),
        ("uwave_n5_2,4,5,6,7", None),
        ("uwave_n5_2,4,5,6,8", None),
        ("uwave_n5_2,4,6,7,8", None),
        ("uwave_n5_3,4,5,6,7", None),
        ("uwave_n5_3,4,5,6,8", None),
        ("uwave_t1", None),
        ("uwave_t2", None),
        ("uwave_t3", None),
        ("uwave_t4", None),
        ("uwave_t5", None),
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
