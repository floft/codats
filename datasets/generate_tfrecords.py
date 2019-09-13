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
        ("sleep_n11_0,14,10,12,25,6,15,4,7,13,20", "sleep_t2"),
        ("sleep_n11_1,10,13,3,24,0,25,6,16,15,12", "sleep_t4"),
        ("sleep_n11_10,19,12,20,16,6,0,25,9,13,1", "sleep_t3"),
        ("sleep_n11_10,8,0,14,6,2,18,22,21,17,24", "sleep_t1"),
        ("sleep_n11_13,12,14,10,20,15,2,0,16,4,24", "sleep_t1"),
        ("sleep_n11_16,5,1,12,8,19,18,14,23,11,22", "sleep_t2"),
        ("sleep_n11_17,13,10,20,19,7,6,11,16,22,12", "sleep_t0"),
        ("sleep_n11_18,15,13,8,25,5,0,7,3,12,10", "sleep_t4"),
        ("sleep_n11_2,17,4,23,0,13,9,21,10,12,6", "sleep_t3"),
        ("sleep_n11_23,1,3,13,14,11,6,0,16,10,15", "sleep_t4"),
        ("sleep_n11_23,1,9,24,11,25,6,16,14,22,15", "sleep_t0"),
        ("sleep_n11_25,8,10,12,19,22,7,23,17,2,11", "sleep_t1"),
        ("sleep_n11_3,13,12,8,25,4,10,23,18,16,22", "sleep_t0"),
        ("sleep_n11_4,11,22,18,2,10,19,12,9,15,24", "sleep_t3"),
        ("sleep_n11_5,6,9,16,7,19,23,12,14,13,18", "sleep_t2"),
        ("sleep_n16_0,14,10,12,25,6,15,4,7,13,20,23,9,24,11,17", "sleep_t2"),
        ("sleep_n16_1,10,13,3,24,0,25,6,16,15,12,21,2,18,5,8", "sleep_t4"),
        ("sleep_n16_10,19,12,20,16,6,0,25,9,13,1,23,22,17,2,14", "sleep_t3"),
        ("sleep_n16_10,8,0,14,6,2,18,22,21,17,24,20,12,4,3,9", "sleep_t1"),
        ("sleep_n16_13,12,14,10,20,15,2,0,16,4,24,11,5,7,17,19", "sleep_t1"),
        ("sleep_n16_16,5,1,12,8,19,18,14,23,11,22,24,13,20,3,7", "sleep_t2"),
        ("sleep_n16_17,13,10,20,19,7,6,11,16,22,12,18,2,15,23,14", "sleep_t0"),
        ("sleep_n16_18,15,13,8,25,5,0,7,3,12,10,14,2,20,9,22", "sleep_t4"),
        ("sleep_n16_2,17,4,23,0,13,9,21,10,12,6,24,11,15,1,25", "sleep_t3"),
        ("sleep_n16_23,1,3,13,14,11,6,0,16,10,15,21,22,24,20,19", "sleep_t4"),
        ("sleep_n16_23,1,9,24,11,25,6,16,14,22,15,4,17,5,13,8", "sleep_t0"),
        ("sleep_n16_25,8,10,12,19,22,7,23,17,2,11,0,16,24,9,15", "sleep_t1"),
        ("sleep_n16_3,13,12,8,25,4,10,23,18,16,22,21,17,15,2,20", "sleep_t0"),
        ("sleep_n16_4,11,22,18,2,10,19,12,9,15,24,8,17,5,21,0", "sleep_t3"),
        ("sleep_n16_5,6,9,16,7,19,23,12,14,13,18,1,24,22,10,20", "sleep_t2"),
        ("sleep_n1_0", "sleep_t2"),
        ("sleep_n1_1", "sleep_t4"),
        ("sleep_n1_10", "sleep_t1"),
        ("sleep_n1_10", "sleep_t3"),
        ("sleep_n1_13", "sleep_t1"),
        ("sleep_n1_16", "sleep_t2"),
        ("sleep_n1_17", "sleep_t0"),
        ("sleep_n1_18", "sleep_t4"),
        ("sleep_n1_2", "sleep_t3"),
        ("sleep_n1_23", "sleep_t0"),
        ("sleep_n1_23", "sleep_t4"),
        ("sleep_n1_25", "sleep_t1"),
        ("sleep_n1_3", "sleep_t0"),
        ("sleep_n1_4", "sleep_t3"),
        ("sleep_n1_5", "sleep_t2"),
        ("sleep_n21_0,14,10,12,25,6,15,4,7,13,20,23,9,24,11,17,5,18,21,16,22", "sleep_t2"),
        ("sleep_n21_1,10,13,3,24,0,25,6,16,15,12,21,2,18,5,8,20,7,9,11,17", "sleep_t4"),
        ("sleep_n21_10,19,12,20,16,6,0,25,9,13,1,23,22,17,2,14,5,15,18,21,7", "sleep_t3"),
        ("sleep_n21_10,8,0,14,6,2,18,22,21,17,24,20,12,4,3,9,5,15,13,16,25", "sleep_t1"),
        ("sleep_n21_13,12,14,10,20,15,2,0,16,4,24,11,5,7,17,19,8,18,6,3,21", "sleep_t1"),
        ("sleep_n21_16,5,1,12,8,19,18,14,23,11,22,24,13,20,3,7,10,4,21,25,17", "sleep_t2"),
        ("sleep_n21_17,13,10,20,19,7,6,11,16,22,12,18,2,15,23,14,3,24,5,25,8", "sleep_t0"),
        ("sleep_n21_18,15,13,8,25,5,0,7,3,12,10,14,2,20,9,22,6,21,17,16,19", "sleep_t4"),
        ("sleep_n21_2,17,4,23,0,13,9,21,10,12,6,24,11,15,1,25,14,20,8,16,22", "sleep_t3"),
        ("sleep_n21_23,1,3,13,14,11,6,0,16,10,15,21,22,24,20,19,8,25,12,5,9", "sleep_t4"),
        ("sleep_n21_23,1,9,24,11,25,6,16,14,22,15,4,17,5,13,8,2,3,7,19,12", "sleep_t0"),
        ("sleep_n21_25,8,10,12,19,22,7,23,17,2,11,0,16,24,9,15,18,13,3,20,14", "sleep_t1"),
        ("sleep_n21_3,13,12,8,25,4,10,23,18,16,22,21,17,15,2,20,7,5,24,11,14", "sleep_t0"),
        ("sleep_n21_4,11,22,18,2,10,19,12,9,15,24,8,17,5,21,0,20,25,1,13,7", "sleep_t3"),
        ("sleep_n21_5,6,9,16,7,19,23,12,14,13,18,1,24,22,10,20,0,11,3,21,8", "sleep_t2"),
        ("sleep_n6_0,14,10,12,25,6", "sleep_t2"),
        ("sleep_n6_1,10,13,3,24,0", "sleep_t4"),
        ("sleep_n6_10,19,12,20,16,6", "sleep_t3"),
        ("sleep_n6_10,8,0,14,6,2", "sleep_t1"),
        ("sleep_n6_13,12,14,10,20,15", "sleep_t1"),
        ("sleep_n6_16,5,1,12,8,19", "sleep_t2"),
        ("sleep_n6_17,13,10,20,19,7", "sleep_t0"),
        ("sleep_n6_18,15,13,8,25,5", "sleep_t4"),
        ("sleep_n6_2,17,4,23,0,13", "sleep_t3"),
        ("sleep_n6_23,1,3,13,14,11", "sleep_t4"),
        ("sleep_n6_23,1,9,24,11,25", "sleep_t0"),
        ("sleep_n6_25,8,10,12,19,22", "sleep_t1"),
        ("sleep_n6_3,13,12,8,25,4", "sleep_t0"),
        ("sleep_n6_4,11,22,18,2,10", "sleep_t3"),
        ("sleep_n6_5,6,9,16,7,19", "sleep_t2"),
        ("ucihar_n13_10,15,24,18,25,7,1,11,13,26,30,28,17", "ucihar_t4"),
        ("ucihar_n13_12,5,25,2,14,16,10,30,18,7,17,6,23", "ucihar_t1"),
        ("ucihar_n13_16,10,27,22,6,28,13,15,1,12,14,5,11", "ucihar_t4"),
        ("ucihar_n13_16,14,6,8,15,29,10,4,23,22,28,2,27", "ucihar_t3"),
        ("ucihar_n13_20,16,12,24,23,8,7,14,13,17,11,26,21", "ucihar_t1"),
        ("ucihar_n13_20,7,25,8,17,18,19,29,4,26,13,2,15", "ucihar_t5"),
        ("ucihar_n13_21,20,18,16,24,17,8,3,4,22,6,14,13", "ucihar_t2"),
        ("ucihar_n13_21,9,4,27,12,22,6,15,10,23,1,7,2", "ucihar_t3"),
        ("ucihar_n13_23,1,15,20,7,27,26,13,11,28,5,21,16", "ucihar_t3"),
        ("ucihar_n13_23,26,11,7,18,28,5,22,8,12,4,16,1", "ucihar_t2"),
        ("ucihar_n13_26,23,8,30,20,14,9,12,29,3,10,13,15", "ucihar_t5"),
        ("ucihar_n13_26,4,19,20,24,29,11,25,21,14,8,16,30", "ucihar_t5"),
        ("ucihar_n13_27,23,18,6,21,24,20,5,3,9,12,19,29", "ucihar_t4"),
        ("ucihar_n13_4,20,1,13,26,18,24,11,15,5,17,6,8", "ucihar_t2"),
        ("ucihar_n13_4,23,14,9,21,17,18,28,30,11,13,25,16", "ucihar_t1"),
        ("ucihar_n19_10,15,24,18,25,7,1,11,13,26,30,28,17,3,20,6,16,19,21", "ucihar_t4"),
        ("ucihar_n19_12,5,25,2,14,16,10,30,18,7,17,6,23,26,15,24,27,29,9", "ucihar_t1"),
        ("ucihar_n19_16,10,27,22,6,28,13,15,1,12,14,5,11,23,8,9,19,21,24", "ucihar_t4"),
        ("ucihar_n19_16,14,6,8,15,29,10,4,23,22,28,2,27,20,24,9,11,13,5", "ucihar_t3"),
        ("ucihar_n19_20,16,12,24,23,8,7,14,13,17,11,26,21,30,18,28,3,15,4", "ucihar_t1"),
        ("ucihar_n19_20,7,25,8,17,18,19,29,4,26,13,2,15,21,27,1,3,28,16", "ucihar_t5"),
        ("ucihar_n19_21,20,18,16,24,17,8,3,4,22,6,14,13,28,9,7,15,11,26", "ucihar_t2"),
        ("ucihar_n19_21,9,4,27,12,22,6,15,10,23,1,7,2,29,17,5,20,18,16", "ucihar_t3"),
        ("ucihar_n19_23,1,15,20,7,27,26,13,11,28,5,21,16,25,14,24,22,6,19", "ucihar_t3"),
        ("ucihar_n19_23,26,11,7,18,28,5,22,8,12,4,16,1,21,3,17,30,29,13", "ucihar_t2"),
        ("ucihar_n19_26,23,8,30,20,14,9,12,29,3,10,13,15,2,17,6,4,25,27", "ucihar_t5"),
        ("ucihar_n19_26,4,19,20,24,29,11,25,21,14,8,16,30,9,13,7,23,6,12", "ucihar_t5"),
        ("ucihar_n19_27,23,18,6,21,24,20,5,3,9,12,19,29,13,26,11,16,10,22", "ucihar_t4"),
        ("ucihar_n19_4,20,1,13,26,18,24,11,15,5,17,6,8,27,29,23,21,3,12", "ucihar_t2"),
        ("ucihar_n19_4,23,14,9,21,17,18,28,30,11,13,25,16,19,3,5,22,8,6", "ucihar_t1"),
        ("ucihar_n1_10", "ucihar_t4"),
        ("ucihar_n1_12", "ucihar_t1"),
        ("ucihar_n1_16", "ucihar_t3"),
        ("ucihar_n1_16", "ucihar_t4"),
        ("ucihar_n1_17", "ucihar_t5"),
        ("ucihar_n1_20", "ucihar_t1"),
        ("ucihar_n1_20", "ucihar_t5"),
        ("ucihar_n1_21", "ucihar_t2"),
        ("ucihar_n1_21", "ucihar_t3"),
        ("ucihar_n1_23", "ucihar_t2"),
        ("ucihar_n1_23", "ucihar_t3"),
        ("ucihar_n1_26", "ucihar_t5"),
        ("ucihar_n1_27", "ucihar_t4"),
        ("ucihar_n1_4", "ucihar_t1"),
        ("ucihar_n1_4", "ucihar_t2"),
        ("ucihar_n25_10,15,24,18,25,7,1,11,13,26,30,28,17,3,20,6,16,19,21,8,9,27,12,14,2", "ucihar_t4"),
        ("ucihar_n25_12,5,25,2,14,16,10,30,18,7,17,6,23,26,15,24,27,29,9,3,4,8,20,13,21", "ucihar_t1"),
        ("ucihar_n25_16,10,27,22,6,28,13,15,1,12,14,5,11,23,8,9,19,21,24,3,25,29,7,26,18", "ucihar_t4"),
        ("ucihar_n25_16,14,6,8,15,29,10,4,23,22,28,2,27,20,24,9,11,13,5,25,17,12,21,1,18", "ucihar_t3"),
        ("ucihar_n25_20,16,12,24,23,8,7,14,13,17,11,26,21,30,18,28,3,15,4,19,29,6,27,9,10", "ucihar_t1"),
        ("ucihar_n25_20,7,25,8,17,18,19,29,4,26,13,2,15,21,27,1,3,28,16,11,22,30,6,14,10", "ucihar_t5"),
        ("ucihar_n25_21,20,18,16,24,17,8,3,4,22,6,14,13,28,9,7,15,11,26,12,10,30,27,5,25", "ucihar_t2"),
        ("ucihar_n25_21,9,4,27,12,22,6,15,10,23,1,7,2,29,17,5,20,18,16,13,14,11,28,30,24", "ucihar_t3"),
        ("ucihar_n25_23,1,15,20,7,27,26,13,11,28,5,21,16,25,14,24,22,6,19,8,17,10,30,18,29", "ucihar_t3"),
        ("ucihar_n25_23,26,11,7,18,28,5,22,8,12,4,16,1,21,3,17,30,29,13,14,24,15,20,10,27", "ucihar_t2"),
        ("ucihar_n25_26,23,8,30,20,14,9,12,29,3,10,13,15,2,17,6,4,25,27,1,18,22,21,11,24", "ucihar_t5"),
        ("ucihar_n25_26,4,19,20,24,29,11,25,21,14,8,16,30,9,13,7,23,6,12,1,27,28,15,22,3", "ucihar_t5"),
        ("ucihar_n25_27,23,18,6,21,24,20,5,3,9,12,19,29,13,26,11,16,10,22,1,25,30,2,14,8", "ucihar_t4"),
        ("ucihar_n25_4,20,1,13,26,18,24,11,15,5,17,6,8,27,29,23,21,3,12,28,30,22,10,14,16", "ucihar_t2"),
        ("ucihar_n25_4,23,14,9,21,17,18,28,30,11,13,25,16,19,3,5,22,8,6,29,12,15,24,7,26", "ucihar_t1"),
        ("ucihar_n7_10,15,24,18,25,7,1", "ucihar_t4"),
        ("ucihar_n7_12,5,25,2,14,16,10", "ucihar_t1"),
        ("ucihar_n7_16,10,27,22,6,28,13", "ucihar_t4"),
        ("ucihar_n7_16,14,6,8,15,29,10", "ucihar_t3"),
        ("ucihar_n7_20,16,12,24,23,8,7", "ucihar_t1"),
        ("ucihar_n7_20,7,25,8,17,18,19", "ucihar_t5"),
        ("ucihar_n7_21,20,18,16,24,17,8", "ucihar_t2"),
        ("ucihar_n7_21,9,4,27,12,22,6", "ucihar_t3"),
        ("ucihar_n7_23,1,15,20,7,27,26", "ucihar_t3"),
        ("ucihar_n7_23,26,11,7,18,28,5", "ucihar_t2"),
        ("ucihar_n7_26,23,8,30,20,14,9", "ucihar_t5"),
        ("ucihar_n7_26,4,19,20,24,29,11", "ucihar_t5"),
        ("ucihar_n7_27,23,18,6,21,24,20", "ucihar_t4"),
        ("ucihar_n7_4,20,1,13,26,18,24", "ucihar_t2"),
        ("ucihar_n7_4,23,14,9,21,17,18", "ucihar_t1"),
        ("uwave_n1_1", "uwave_t5"),
        ("uwave_n1_2", "uwave_t3"),
        ("uwave_n1_3", "uwave_t1"),
        ("uwave_n1_3", "uwave_t5"),
        ("uwave_n1_4", "uwave_t1"),
        ("uwave_n1_4", "uwave_t3"),
        ("uwave_n1_5", "uwave_t1"),
        ("uwave_n1_5", "uwave_t4"),
        ("uwave_n1_6", "uwave_t2"),
        ("uwave_n1_6", "uwave_t3"),
        ("uwave_n1_6", "uwave_t4"),
        ("uwave_n1_7", "uwave_t2"),
        ("uwave_n1_8", "uwave_t2"),
        ("uwave_n1_8", "uwave_t4"),
        ("uwave_n1_8", "uwave_t5"),
        ("uwave_n2_2,1", "uwave_t3"),
        ("uwave_n2_3,5", "uwave_t1"),
        ("uwave_n2_3,6", "uwave_t4"),
        ("uwave_n2_4,6", "uwave_t3"),
        ("uwave_n2_4,6", "uwave_t5"),
        ("uwave_n2_4,7", "uwave_t2"),
        ("uwave_n2_5,1", "uwave_t4"),
        ("uwave_n2_5,4", "uwave_t1"),
        ("uwave_n2_5,8", "uwave_t1"),
        ("uwave_n2_6,3", "uwave_t4"),
        ("uwave_n2_6,7", "uwave_t3"),
        ("uwave_n2_7,1", "uwave_t2"),
        ("uwave_n2_8,1", "uwave_t5"),
        ("uwave_n2_8,4", "uwave_t5"),
        ("uwave_n2_8,5", "uwave_t2"),
        ("uwave_n3_2,1,5", "uwave_t3"),
        ("uwave_n3_3,5,6", "uwave_t1"),
        ("uwave_n3_3,6,5", "uwave_t4"),
        ("uwave_n3_4,6,2", "uwave_t3"),
        ("uwave_n3_4,6,8", "uwave_t5"),
        ("uwave_n3_4,7,1", "uwave_t2"),
        ("uwave_n3_5,1,6", "uwave_t4"),
        ("uwave_n3_5,4,6", "uwave_t1"),
        ("uwave_n3_5,8,4", "uwave_t1"),
        ("uwave_n3_6,3,7", "uwave_t4"),
        ("uwave_n3_6,7,2", "uwave_t3"),
        ("uwave_n3_7,1,5", "uwave_t2"),
        ("uwave_n3_8,1,6", "uwave_t5"),
        ("uwave_n3_8,4,6", "uwave_t5"),
        ("uwave_n3_8,5,7", "uwave_t2"),
        ("uwave_n4_2,1,5,7", "uwave_t3"),
        ("uwave_n4_3,5,6,4", "uwave_t1"),
        ("uwave_n4_3,6,5,2", "uwave_t4"),
        ("uwave_n4_4,6,2,7", "uwave_t3"),
        ("uwave_n4_4,6,8,2", "uwave_t5"),
        ("uwave_n4_4,7,1,5", "uwave_t2"),
        ("uwave_n4_5,1,6,8", "uwave_t4"),
        ("uwave_n4_5,4,6,3", "uwave_t1"),
        ("uwave_n4_5,8,4,6", "uwave_t1"),
        ("uwave_n4_6,3,7,5", "uwave_t4"),
        ("uwave_n4_6,7,2,4", "uwave_t3"),
        ("uwave_n4_7,1,5,6", "uwave_t2"),
        ("uwave_n4_8,1,6,3", "uwave_t5"),
        ("uwave_n4_8,4,6,2", "uwave_t5"),
        ("uwave_n4_8,5,7,3", "uwave_t2"),
        ("uwave_n5_2,1,5,7,8", "uwave_t3"),
        ("uwave_n5_3,5,6,4,8", "uwave_t1"),
        ("uwave_n5_3,6,5,2,8", "uwave_t4"),
        ("uwave_n5_4,6,2,7,5", "uwave_t3"),
        ("uwave_n5_4,6,8,2,1", "uwave_t5"),
        ("uwave_n5_4,7,1,5,8", "uwave_t2"),
        ("uwave_n5_5,1,6,8,2", "uwave_t4"),
        ("uwave_n5_5,4,6,3,7", "uwave_t1"),
        ("uwave_n5_5,8,4,6,2", "uwave_t1"),
        ("uwave_n5_6,3,7,5,1", "uwave_t4"),
        ("uwave_n5_6,7,2,4,8", "uwave_t3"),
        ("uwave_n5_7,1,5,6,3", "uwave_t2"),
        ("uwave_n5_8,1,6,3,7", "uwave_t5"),
        ("uwave_n5_8,4,6,2,1", "uwave_t5"),
        ("uwave_n5_8,5,7,3,1", "uwave_t2"),
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
