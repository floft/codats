"""
Datasets

Usage:
    import datasets

    print(datasets.names()) # mnist, usps, svhn, ...
    mnist = datasets.load("mnist")
    class_number = mnist.label_to_int("5")
    class_name = mnist.int_to_label(5)

    # Training
    train_iter = iter(mnist.train):
    labels = mnist.class_labels
    while True:
        next_batch = next(train_iter)

    # Evaluation
    for x, y in mnist.train_evaluation:
        ...
    for x, y in mnist.test_evaluation:
        ...
"""
import os
import tensorflow as tf

from absl import app
from absl import flags

from datasets import datasets
from datasets.tfrecord import tfrecord_filename

FLAGS = flags.FLAGS

flags.DEFINE_integer("train_batch", 128, "Batch size for training")
flags.DEFINE_integer("eval_batch", 4096, "Batch size for evaluation")
flags.DEFINE_integer("shuffle_buffer", 60000, "Dataset shuffle buffer size")
flags.DEFINE_integer("prefetch_buffer", 1, "Dataset prefetch buffer size (0 = autotune)")
flags.DEFINE_boolean("tune_num_parallel_calls", False, "Autotune num_parallel_calls")
flags.DEFINE_integer("eval_shuffle_seed", 0, "Evaluation shuffle seed for repeatability")
flags.DEFINE_integer("eval_max_examples", 0, "Max number of examples to evaluate for validation (default 0, i.e. all)")
flags.DEFINE_boolean("train_on_source_valid", True, "Train on source validation data for small training sets (and in this case, don't draw much from the number)")
flags.DEFINE_boolean("train_on_target_valid", False, "Train on target validation data for small training sets (i.e., Office-31)")


class Dataset:
    """ Load datasets from tfrecord files """
    def __init__(self, num_classes, class_labels,
            train_filenames, test_filenames,
            train_batch=None, eval_batch=None,
            shuffle_buffer=None, prefetch_buffer=None,
            eval_shuffle_seed=None, eval_max_examples=None,
            tune_num_parallel_calls=None):
        """
        Initialize dataset

        Must specify num_classes and class_labels (the names of the classes).
        Other arguments if None are defaults from command line flags.

        For example:
            Dataset(num_classes=2, class_labels=["class1", "class2"])
        """
        # Sanity checks
        assert num_classes == len(class_labels), \
            "num_classes != len(class_labels)"

        # Set parameters
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.train_batch = train_batch
        self.eval_batch = eval_batch
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.eval_shuffle_seed = eval_shuffle_seed
        self.eval_max_examples = eval_max_examples
        self.tune_num_parallel_calls = tune_num_parallel_calls

        # Set defaults if not specified
        if self.train_batch is None:
            self.train_batch = FLAGS.train_batch
        if self.eval_batch is None:
            self.eval_batch = FLAGS.eval_batch
        if self.shuffle_buffer is None:
            self.shuffle_buffer = FLAGS.shuffle_buffer
        if self.prefetch_buffer is None:
            self.prefetch_buffer = FLAGS.prefetch_buffer
        if self.eval_shuffle_seed is None:
            self.eval_shuffle_seed = FLAGS.eval_shuffle_seed
        if self.eval_max_examples is None:
            self.eval_max_examples = FLAGS.eval_max_examples
        if self.tune_num_parallel_calls is None:
            self.tune_num_parallel_calls = FLAGS.tune_num_parallel_calls

        # Load the dataset
        self.train, self.train_evaluation, self.test_evaluation = \
            self.load_dataset(train_filenames, test_filenames)

    def load_tfrecords(self, filenames, batch_size, count=False, evaluation=False):
        """
        Load data from .tfrecord files (requires less memory but more disk space)
        max_examples=0 -- no limit on the number of examples
        """
        if len(filenames) == 0:
            return None

        # Create a description of the features
        # See: https://www.tensorflow.org/tutorials/load_data/tf-records
        feature_description = {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_example_function(example_proto):
            """
            Parse the input tf.Example proto using the dictionary above.
            parse_single_example is without a batch, parse_example is with batches

            What's parsed returns byte strings, but really we want to get the
            tensors back that we encoded with tf.io.serialize_tensor() earlier,
            so also run tf.io.parse_tensor
            """
            parsed = tf.io.parse_single_example(serialized=example_proto,
                features=feature_description)

            x = tf.io.parse_tensor(parsed["x"], tf.float32)
            y = tf.io.parse_tensor(parsed["y"], tf.float32)

            return x, y

        # Interleave the tfrecord files
        files = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP').prefetch(100),
            cycle_length=len(filenames), block_length=1)

        # If desired, take the first max_examples examples
        if evaluation and self.eval_max_examples != 0:
            dataset = dataset.take(self.eval_max_examples)

        if count:  # only count, so no need to shuffle
            pass
        elif evaluation:  # don't repeat since we want to evaluate entire set
            dataset = dataset.shuffle(self.shuffle_buffer, seed=self.eval_shuffle_seed)
        else:  # repeat, shuffle, and batch
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(self.shuffle_buffer))

        # Whether to do autotuning of prefetch or num_parallel_calls
        prefetch_buffer = self.prefetch_buffer
        num_parallel_calls = None
        if self.tune_num_parallel_calls:
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        if self.prefetch_buffer == 0:
            prefetch_buffer = tf.data.experimental.AUTOTUNE

        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            _parse_example_function, batch_size,
            num_parallel_calls=num_parallel_calls))

        dataset = dataset.prefetch(prefetch_buffer)

        return dataset

    def load_dataset(self, train_filenames, test_filenames):
        """
        Load the X dataset as a tf.data.Dataset from train/test tfrecord filenames
        """
        train_dataset = self.load_tfrecords(train_filenames, self.train_batch)
        eval_train_dataset = self.load_tfrecords(train_filenames,
            self.eval_batch, evaluation=True)
        eval_test_dataset = self.load_tfrecords(test_filenames,
            self.eval_batch, evaluation=True)

        return train_dataset, eval_train_dataset, eval_test_dataset

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.class_labels.index(label_name)

    def int_to_label(self, label_index):
        """ e.g. Bathe to 0 """
        return self.class_labels[label_index]


def load_da(source_name, target_name, test=False, *args, **kwargs):
    """ Load two datasets (source and target) but perform necessary conversions
    to make them compatable for adaptation (i.e. same size, channels, etc.).
    Names must be in datasets.names().

    If test=True, then load real test set. Otherwise, load validation set as
    the "test" data (for use during training and hyperparameter tuning).
    """
    # Sanity checks
    assert source_name in datasets.datasets, \
        source_name + " not a supported dataset, only "+str(datasets.datasets)
    assert target_name in datasets.datasets, \
        target_name + " not a supported dataset, only "+str(datasets.datasets)

    # Get dataset information
    source_num_classes = datasets.datasets[source_name].num_classes
    source_class_labels = datasets.datasets[source_name].class_labels
    target_num_classes = datasets.datasets[target_name].num_classes
    target_class_labels = datasets.datasets[target_name].class_labels

    # Get dataset tfrecord filenames
    def _path(filename):
        """ Files are in datasets/ subdirectory. If the file exists, return it
        as an array since we may sometimes want more than one file for a
        dataset. If it doesn't exist, ignore it (some datasets don't have a test
        set for example)."""
        fn = os.path.join("datasets", filename)
        return [fn] if os.path.exists(fn) else []

    names = (source_name, target_name)
    source_train_filenames = _path(tfrecord_filename(*names, source_name, "train"))
    source_valid_filenames = _path(tfrecord_filename(*names, source_name, "valid"))
    source_test_filenames = _path(tfrecord_filename(*names, source_name, "test"))
    target_train_filenames = _path(tfrecord_filename(*names, target_name, "train"))
    target_valid_filenames = _path(tfrecord_filename(*names, target_name, "valid"))
    target_test_filenames = _path(tfrecord_filename(*names, target_name, "test"))

    # By default use validation data as the "test" data, unless test=True
    if not test:
        source_test_filenames = source_valid_filenames
        target_test_filenames = target_valid_filenames

        # However, also train on the source "valid" data since we don't actually
        # care about those numbers much and some datasets like Office are really
        # small.
        if FLAGS.train_on_source_valid:
            source_train_filenames += source_valid_filenames
            print("Warning: training on source \"valid\" data")

        # For very small datasets, e.g. Office-31, where there might only be a
        # few thousand target examples, then we might ought to use everything
        # for training (unlabeled still though; only validation uses labels for
        # testing, but not during training).
        if FLAGS.train_on_target_valid:
            target_train_filenames += target_valid_filenames
            print("Warning: training on unlabeled target \"valid\" data")

    # If test=True, then make "train" consist of both training and validation
    # data to match the original dataset.
    else:
        source_train_filenames += source_valid_filenames
        target_train_filenames += target_valid_filenames

    # Create all the train, test, evaluation, ... tf.data.Dataset objects within
    # a Dataset() class that stores them
    source_dataset = Dataset(source_num_classes, source_class_labels,
        source_train_filenames, source_test_filenames,
        *args, **kwargs)
    target_dataset = Dataset(target_num_classes, target_class_labels,
        target_train_filenames, target_test_filenames,
        *args, **kwargs)

    return source_dataset, target_dataset


def names():
    """
    Returns list of all the available datasets to load with
        load_da(source, target, ...)
    """
    return datasets.names()


def main(argv):
    print("Available datasets:", names())

    # Example showing that the sizes and number of channels are matched
    source, target = load_da("mnist", "usps")

    print("Source:", source.train)
    print("Target:", target.train)

    for x, y in source.train:
        print("Source x shape:", x.shape)
        print("Source y shape:", y.shape)
        break

    for x, y in target.train:
        print("Target x shape:", x.shape)
        print("Target y shape:", y.shape)
        break


if __name__ == "__main__":
    app.run(main)
