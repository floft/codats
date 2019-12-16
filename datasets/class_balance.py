#!/usr/bin/env python3
"""
Check the class balance for each dataset

Note: sets CUDA_VISIBLE_DEVICES= so that it doesn't use the GPU.
"""
import os
import numpy as np

from absl import app

import datasets


def calc_class_balance(labels, num_classes, percentage=True):
    """ Count number of labels from each class in the dataset """
    classes = [0]*num_classes

    for class_num in range(0, num_classes):
        this_class = np.argmax(labels, axis=1) == class_num

        classes[class_num] = sum(this_class)

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

    # Get class balances for each of the datasets (separate since they differ
    # in the number of classes)
    print_class_balance("UCI HAR", ["ucihar_"+str(i) for i in range(1, 12+1)])
    print_class_balance("uWave", ["uwave_"+str(i) for i in range(1, 8+1)])
    print_class_balance("UT-Data-Complex", ["utdata_wrist", "utdata_pocket"])


if __name__ == "__main__":
    app.run(main)
