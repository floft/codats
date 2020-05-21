#!/usr/bin/env python3
"""
Check the class balance for each dataset from the raw dataset (before .tfrecord)

Note: sets CUDA_VISIBLE_DEVICES= so that it doesn't use the GPU.

Run something like the following to save the result:
    ./class_balance.py | tee class_balance.txt

(Based on samples_per_target.py and datasets/class_balance.py [see git tag v2])
"""
import os
import numpy as np

from absl import app
from absl import flags

from datasets import datasets
from load_datasets import load_da

FLAGS = flags.FLAGS


def get_labels(dataset):
    """ Count training examples for all the sources datasets """
    ys = []

    for _, y in dataset:
        ys.append(y.numpy())

    return np.hstack(ys)


def calc_class_balance(labels, num_classes):
    """ Count number of labels from each class in the dataset

    (Copied from methods.py)
    """
    p_y = [0]*num_classes

    for class_num in range(0, num_classes):
        # Count instances of this class
        this_class_count = sum(labels == class_num)
        p_y[class_num] = this_class_count

    # Normalize to make P(y) sum to one like a proper probability
    # distribution
    p_y = p_y / sum(p_y)

    return p_y


def class_balance(dataset, num_classes):
    """ First get the labels as a numpy array, then calculate label proportions """
    return calc_class_balance(get_labels(dataset), num_classes)


def print_table(title, classes):
    """ Print tab-separated table of dataset and then each of the classes """
    print(title)
    for n, v in classes.items():
        print(n, *["{:.1f}".format(x*100) for x in v], sep="\t")
    print()


def print_class_balances(dataset_name, user_source_pairs):
    classes_train = {}
    classes_test = {}

    for user, source in user_source_pairs:
        train = class_balance(source.train_evaluation, source.num_classes)
        test = class_balance(source.test_evaluation, source.num_classes)

        name = dataset_name + "_" + str(user)
        classes_train[name] = train
        classes_test[name] = test

    print_table(dataset_name + " (train)", classes_train)
    print_table(dataset_name + " (test)", classes_test)


def main(argv):
    # Don't bother using the GPU for this
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    for dataset_name in datasets.list_datasets():
        user_source_pairs = []

        for user in datasets.get_dataset_users(dataset_name):
            # Note: test=False so we only look at the training samples, where
            # train=80% of training set, test=20% of training set, i.e. the
            # validation set
            sources, _ = load_da(dataset_name, str(user), "", test=False)

            # We load them one at a time
            assert len(sources) == 1
            source = sources[0]

            user_source_pairs.append((user, source))

        print_class_balances(dataset_name, user_source_pairs)


if __name__ == "__main__":
    app.run(main)
