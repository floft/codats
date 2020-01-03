#!/usr/bin/env python3
"""
Check the class balance for each dataset from the raw dataset (before .tfrecord)

Note: ../class_balance.py with test=True will provide this from the .tfrecord
files much faster than running this script.

Note: sets CUDA_VISIBLE_DEVICES= so that it doesn't use the GPU.

Run something like the following to save the result:
    ./class_balance.py | tee class_balance.txt
"""
import os

from absl import app

import datasets

from datasets import dataset_users


def calc_class_balance(labels, num_classes, percentage=True):
    """ Count number of labels from each class in the dataset """
    classes = [0]*num_classes

    for class_num in range(0, num_classes):
        # Count instances of this class -- note labels is a numpy array
        classes[class_num] = sum(labels == class_num)

        if percentage:
            classes[class_num] /= len(labels)

    return classes


def print_table(title, classes):
    """ Print tab-separated table of dataset and then each of the classes """
    print(title)
    for n, v in classes.items():
        print(n, *["{:.1f}".format(x*100) for x in v], sep="\t")
    print()


def print_class_balance(title, dataset_list):
    classes_train = {}
    classes_test = {}

    for dataset in dataset_list:
        source_dataset = datasets.load(dataset)
        #class_labels = source_dataset.class_labels
        num_classes = source_dataset.num_classes

        train = calc_class_balance(source_dataset.train_labels, num_classes)
        test = calc_class_balance(source_dataset.test_labels, num_classes)

        classes_train[dataset] = train
        classes_test[dataset] = test

    print_table(title + " (train)", classes_train)
    print_table(title + " (test)", classes_test)


def main(argv):
    # Don't bother using the GPU for this
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    for name, users in dataset_users.items():
        # Get class balances for each of the datasets
        datasets = [name+"_"+str(x) for x in users]

        # Print
        print_class_balance(name, datasets)


if __name__ == "__main__":
    app.run(main)
