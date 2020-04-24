#!/usr/bin/env python3
"""
Count the number of training samples per dataset-target pair from tfrecord
files. This information is used in the vary-amount-of-target-data experiments.

./samples_per_target.py | tee samples_per_target.txt

Then add the resulting dictionary to pick_multi_source.py
"""
import collections

from absl import app
from absl import flags

from load_datasets import load_da
from datasets import datasets
from print_dictionary import print_dictionary

FLAGS = flags.FLAGS


def count_training_samples(sources):
    """ Count training examples for all the sources datasets """
    train_count = 0

    for source in sources:
        for _, y in source.train_evaluation:
            train_count += len(y)

    return train_count


def main(argv):
    counts = collections.defaultdict(dict)

    for dataset_name in datasets.list_datasets():
        # Note: test=False so we only look at the training samples, which is what
        # we will vary in the vary-amount-of-target-data experiments
        for user in datasets.get_dataset_users(dataset_name):
            sources, _ = load_da(dataset_name, str(user), "", test=False)
            train_count = count_training_samples(sources)
            counts[dataset_name][user] = train_count

    print_dictionary(counts, "dataset_target_training_sample_counts")


if __name__ == "__main__":
    app.run(main)
