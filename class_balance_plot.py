#!/usr/bin/env python3
"""
Plot WISDM AR class balance for paper Appendix (from .tfrecord files)

Note: sets CUDA_VISIBLE_DEVICES= so that it doesn't use the GPU.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from absl import app
from absl import flags

from datasets import datasets
from load_datasets import load_da
from file_utils import get_last_int

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


def compute_class_balances(dataset_name, user_source_pairs):
    balance_data = {}

    for user, source in user_source_pairs:
        train = class_balance(source.train_evaluation, source.num_classes)

        name = dataset_name + "_" + str(user)
        balance_data[name] = train

    return balance_data


def generate_plot(dataset_name, class_labels, class_balances,
        ymin=0, ymax=70, filename=None, horizontal=True, which=None,
        first_n=None):
    """
    Bar plot
    https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barchart.html
    https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barh.html
    """
    # Example:
    # class_labels = ["Men", "Women"]
    # groups = ["Person " + str(i) for i in range(5)]
    # values = [
    #     [20, 34, 30, 35, 27],
    #     [25, 32, 34, 20, 25],
    # ]

    keys = list(class_balances.keys())
    key_ints = [get_last_int(k) for k in keys]
    class_balances = list(class_balances.values())

    # Select subset of them if desired
    if which is not None:
        which = list(set(which))
        which.sort()
        # Since some datasets (uWave) are not 0-indexed, get the indices not
        # just assume it's 0, 1, 2, 3, etc... since sometimes it's 1, 2, 3, ...
        indices = [key_ints.index(w) for w in which]

        key_ints = [key_ints[i] for i in indices]
        class_balances = [class_balances[i] for i in indices]

        if first_n is not None:
            key_ints = key_ints[:first_n]
            class_balances = class_balances[:first_n]

    # Put in the format we want for this plot, see example above
    groups = ["Person " + str(k) for k in key_ints]
    values = []

    for i in range(len(class_labels)):
        values_for_class = []

        for b in class_balances:
            assert len(b) == len(class_labels)
            values_for_class.append(b[i]*100)  # Convert to %

        values.append(values_for_class)

    # Generate the plot
    rects = []

    num_groups = len(groups)
    num_rects = len(class_labels)

    if filename is not None:
        figsize = (1.3*num_groups, 8)

        if horizontal:
            figsize = (figsize[1], figsize[0])

        fig, ax = plt.subplots(figsize=figsize, dpi=200)
    else:
        fig, ax = plt.subplots()

    x = np.arange(num_groups)  # the label locations
    width = 0.70/num_rects  # the width of the bars
    margin = 0.02

    for i in range(num_rects):
        f = ax.barh if horizontal else ax.bar
        rects.append(f(
            x + i*width - width*num_rects/2 + i*margin, values[i], width,
            label=class_labels[i]))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    label = "Label Proportion (%)"
    title = "Label Proportions for " + dataset_name

    # Reduce padding
    pad_margin = 0.25*width
    xmin = min(x) - width*num_rects/2 - width - pad_margin
    xmax = max(x) + width*num_rects/2 + width + pad_margin

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(groups)
        ax.set_xlim([ymin, ymax])
        ax.set_ylim([xmin, xmax])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel(label)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_ylabel(label)

    if filename is None:
        ax.set_title(title)

    ax.legend()

    def formatnum(x):
        """ Remove zero at front of float, and round """
        s = "{:.1f}".format(x)
        # s = s.replace(".00", ".0")

        if s == "0.0":
            return "0"

        return s[1:] if s[0] == "0" else s

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            if horizontal:
                width = rect.get_width()
                ax.annotate(formatnum(width),
                    xy=(rect.get_x() + width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points", ha='left', va='center')
            else:
                height = rect.get_height()
                ax.annotate(formatnum(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points", ha='center', va='bottom')

    for rect in rects:
        autolabel(rect)

    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def main(argv):
    # Don't bother using the GPU for this
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ACM doesn't like Type 3 fonts
    # https://tex.stackexchange.com/q/18687
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)

    # We only want to plot this one dataset
    list_of_datasets = ["wisdm_ar", "ucihar", "uwave", "ucihhar"]
    dataset_names_nice = ["WISDM AR", "UCI HAR", "uWave", "UCI HHAR"]
    # Get only the ones used in the SS-DA experiments
    dataset_which = [
        [1, 3, 4, 2, 25, 7, 21, 2, 1, 0, 11, 15, 25, 29, 30, 31, 32, 7, 8],
        # [2, 7, 12, 12, 9, 14, 18, 6, 7, 17, 11, 13, 16, 18, 18, 19, 23, 24, 25],
        # [2, 3, 4, 2, 1, 2, 3, 1, 4, 7, 5, 6, 7, 8],
        # [1, 3, 4, 0, 1, 4, 5, 2, 3, 5, 3, 5, 6, 7, 8],
        # Just do for the first two adaptation problems in SS-DA experiments
        [2, 11, 7, 13],
        [2, 5, 3, 5],
        [1, 3, 3, 5],
    ]
    ymaxs = [70, 35, 16, 28]
    # We mostly care about WISDM AR and don't have enough space for all of them
    first_ns = [None, None, None, None]

    for i in range(len(list_of_datasets)):
        dataset_name = list_of_datasets[i]
        dataset_name_nice = dataset_names_nice[i]
        which = dataset_which[i]
        ymax = ymaxs[i]
        first_n = first_ns[i]

        # Get class balance for all users
        user_source_pairs = []

        for user in datasets.get_dataset_users(dataset_name):
            # Note: train_on_everything=True means the training dataset consists
            # of all train/valid/test data.
            sources, _ = load_da(dataset_name, str(user), "",
                train_on_everything=True)

            # We load them one at a time
            assert len(sources) == 1
            source = sources[0]

            user_source_pairs.append((user, source))

        balance_data = compute_class_balances(dataset_name, user_source_pairs)

        # Plot it
        class_labels = datasets.get_dataset(dataset_name).class_labels
        generate_plot(dataset_name_nice, class_labels, balance_data,
            filename="class_balance_" + dataset_name + ".pdf",
            which=which, ymax=ymax, first_n=first_n)


if __name__ == "__main__":
    app.run(main)
