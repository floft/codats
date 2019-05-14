#!/usr/bin/env python3
"""
As a sanity check, load the data from the source/target domains and display it

Note: sets CUDA_VISIBLE_DEVICES= so that it doesn't use the GPU.
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from absl import app
from absl import flags

import datasets

FLAGS = flags.FLAGS

flags.DEFINE_enum("source", None, datasets.names(), "What dataset to use as the source")
flags.DEFINE_enum("target", "", [""]+datasets.names(), "What dataset to use as the target")
flags.DEFINE_boolean("test", False, "Show test images instead of training images")

flags.mark_flag_as_required("source")


def display(name, images, max_number=16, office=False):
    fig = plt.figure(figsize=(4, 4))
    fig.suptitle(name)

    for i, image, in enumerate(images[:max_number]):
        plt.subplot(4, 4, i+1)
        channels = image.shape[2]

        if i == 0:
            print(name, "shape", image.shape, "min", image.min(),
                "max", image.max(), "mean", image.mean())

        if office:
            plt.imshow(image[:, :, 0])
        elif channels == 1:
            plt.imshow(image[:, :, 0] * 127.5 + 127.5, cmap='gray')
        elif channels == 3:
            plt.imshow(tf.cast(image[:, :, :] * 127.5 + 127.5, tf.int32))
        else:
            raise NotImplementedError("display() only supports gray or RGB")

        plt.axis('off')


def main(argv):
    # Don't bother using the GPU for this
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Input data
    if FLAGS.target != "":
        source_dataset, target_dataset = datasets.load_da(FLAGS.source, FLAGS.target)
    else:
        source_dataset = datasets.load(FLAGS.source)
        target_dataset = None

    if not FLAGS.test:
        source_data = source_dataset.train_images
        target_data = target_dataset.train_images \
            if target_dataset is not None else None
    else:
        source_data = source_dataset.test_images
        target_data = target_dataset.test_images \
            if target_dataset is not None else None

    display("Source", source_data, office="office_" in FLAGS.source)

    if target_dataset is not None:
        display("Target", target_data, office="office_" in FLAGS.target)

    plt.show()


if __name__ == "__main__":
    app.run(main)
