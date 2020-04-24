#!/usr/bin/env python3
"""
As a sanity check, load the data from the source/target domains and display it

Run:

    python -m datasets.view_datasets <args>

Example:

    python -m datasets.view_datasets --source=ucihar_1 --target=ucihar_2

Note: sets CUDA_VISIBLE_DEVICES= so that it doesn't use the GPU.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from absl import app
from absl import flags

from datasets import datasets
from datasets.normalization import calc_normalization, apply_normalization

FLAGS = flags.FLAGS

flags.DEFINE_enum("source", None, datasets.names(), "What dataset to use as the source")
flags.DEFINE_enum("target", "", [""]+datasets.names(), "What dataset to use as the target")
flags.DEFINE_boolean("test", False, "Show test images instead of training images")
flags.DEFINE_integer("minexample", 0, "Start plotting with this example")
flags.DEFINE_integer("maxexample", 5, "Stop plotting with this example")

flags.mark_flag_as_required("source")


def display(name, data, feature_names):
    # Shape: examples, time steps, features
    num_examples, num_samples, num_features = data.shape

    fig, axes = plt.subplots(nrows=num_features, ncols=1,
        sharex=True, sharey=False)
    fig.suptitle(name)

    # Normalize
    if FLAGS.normalize != "none":
        data = apply_normalization(data, calc_normalization(data, FLAGS.normalize))

    for i in range(num_features):
        if num_features == 1:
            ax = axes
        else:
            ax = axes[i]

        # Set x to 0..num_samples. We care about the data values not the time
        # scale.
        x_list = np.tile(np.arange(0, num_samples),
            (FLAGS.maxexample-FLAGS.minexample, 1)).T
        values = data[FLAGS.minexample:FLAGS.maxexample, :, i].T

        if feature_names is not None:
            label = feature_names[i]
        else:
            label = "#"+str(i)

        ax.plot(x_list, values)
        ax.set_ylabel(label)
        ax.set_ylim(auto=True)


def main(argv):
    # Don't bother using the GPU for this
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Input data
    source_dataset, _ = datasets.load(FLAGS.source)

    if FLAGS.target != "":
        target_dataset, _ = datasets.load(FLAGS.target)
    else:
        target_dataset = None

    if not FLAGS.test:
        source_data = source_dataset.train_data
        target_data = target_dataset.train_data \
            if target_dataset is not None else None
    else:
        source_data = source_dataset.test_data
        target_data = target_dataset.test_data \
            if target_dataset is not None else None

    source_feature_names = source_dataset.feature_names
    target_feature_names = target_dataset.feature_names \
        if target_dataset is not None else None

    display("Source", source_data, source_feature_names)

    if target_dataset is not None:
        display("Target", target_data, target_feature_names)

    plt.show()


if __name__ == "__main__":
    app.run(main)
