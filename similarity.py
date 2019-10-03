#!/usr/bin/env python3
"""
Check how similar two domain adaptation problems are using dynamic time warping
"""
import os
import ot
import pathlib
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

import models
import load_datasets

from models import DomainAdaptationModel
from checkpoints import CheckpointManager
from gpu_memory import set_gpu_memory

FLAGS = flags.FLAGS

# Copy from main.py
methods = [
    # Domain adaptation
    "none", "random", "cyclegan", "forecast", "cyclegan_dann", "cycada",
    "dann_shu", "dann_grl", "deepjdot", "pseudo", "instance", "rdann", "vrada",

    # Domain generalization
    "dann_grl_dg", "sleep_dg",
]

flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
# Specific for evaluation
flags.DEFINE_float("gpumem", 8140, "GPU memory to let TensorFlow use, in MiB (divided among jobs)")
flags.DEFINE_boolean("last", False, "Use last model rather than one with best validation set performance")
flags.DEFINE_integer("sample_size", 1000, "How many pairs to sample on which distance is computed")
flags.DEFINE_integer("repeat", 3, "How times to repeat the sampling for computing the standard deviation")


def get_model_args():
    """
    Returns information about the model for which the log/model dirs were given

    Returns: log_dir, model_dir, source, target, model_name, method_name
    """
    log_dir = pathlib.Path(FLAGS.logdir)

    items = str(log_dir.stem).split("-")
    assert len(items) >= 4, \
        "name should be one of source-target-model-method{,-num}"

    source, target, model_name, method_name = items[:4]
    assert method_name in methods, "unknown method "+method_name

    model_dir = FLAGS.modeldir
    assert os.path.exists(model_dir), "Model does not exist "+str(model_dir)

    assert model_name in models.names(), "Unknown model "+str(model_name)
    assert source in load_datasets.names(), "Unknown source "+str(source)
    assert target in [""]+load_datasets.names(), "Unknown target "+str(target)

    return str(log_dir), model_dir, source, target, model_name, method_name


def load_model_and_data(log_dir, model_dir, source, target, model_name, method_name):
    # Only support these at the moment
    assert model_name in ["fcn"]
    assert method_name in ["dann_grl"]
    assert target != ""

    # Load datasets
    source_datasets, target_dataset = load_datasets.load_da(source, target,
        test=True)

    # Evaluation datasets if we have the dataset
    source_dataset_train = source_dataset.train_evaluation
    target_dataset_train = target_dataset.train_evaluation
    source_dataset_test = source_dataset.test_evaluation
    target_dataset_test = target_dataset.test_evaluation

    # Information about domains
    num_classes = source_dataset.num_classes
    num_domains = source_dataset.num_domains

    # Build our model
    model = DomainAdaptationModel(num_classes, num_domains, model_name, 1, 1)

    # TODO maybe support mapping and other options at some point,
    # see main_eval.py

    # Load model from checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir)

    if FLAGS.last:
        checkpoint_manager.restore_latest()
    else:
        checkpoint_manager.restore_best()

    assert checkpoint_manager.found, "Could not find model to restore"

    return model, source_dataset_train, target_dataset_train, \
        source_dataset_test, target_dataset_test


def get_samples(dataset, number=None, random_seed=None):
    """ Get max of "number" samples (just the x value) from dataset,
    randomly shuffled and stratified by domain """
    # Get data
    xs = []
    domains = []

    for x, y, domain in dataset:
        xs.append(x)
        domains.append(domain)

    # Since the data is in batches, we need to stack the results
    xs = np.vstack(xs).astype(np.float32)
    domains = np.hstack(domains).astype(np.float32)

    # If we want more than the number of samples we have, then we can't "split".
    # Just use them all. Otherwise, shuffle split but stratify on domain, get
    # at most "number" of them
    if number < len(xs):
        _, xs = train_test_split(xs, test_size=number, stratify=domains,
            random_state=random_seed)

    return xs


@tf.function
def run_model(model, x, domain):
    """ Compile running the model """
    _, _, embedding = model(x, training=False, domain=domain)
    return embedding


def get_embeddings(model, samples, domain):
    """ Run samples through the model to compute the embedding """
    embeddings = []

    dataset = tf.data.Dataset.from_tensor_slices(samples)
    dataset = dataset.batch(FLAGS.eval_batch)
    dataset = dataset.prefetch(FLAGS.prefetch_buffer)

    for x in dataset:
        embeddings.append(run_model(model, x, domain))

    # Since the data is in batches, we need to stack the results
    embeddings = np.vstack(embeddings).astype(np.float32)

    return embeddings


def compute_distance(model, source_dataset, target_dataset):
    """ Compute Wasserstein distance between source and target distributions """
    # Get x data
    source_samples = get_samples(source_dataset, number=FLAGS.sample_size)
    target_samples = get_samples(target_dataset, number=FLAGS.sample_size)

    # Get embeddings of samples
    source_embeddings = get_embeddings(model, source_samples, "source")
    target_embeddings = get_embeddings(model, target_samples, "target")

    # Ground cost matrix, using squared Euclidean distance (in feature space)
    # Maybe: ot.gpu.dist
    C = cdist(source_embeddings, target_embeddings, metric="sqeuclidean")

    # Wasserstein distance
    emd = ot.emd2(ot.unif(source_embeddings.shape[0]),
        ot.unif(target_embeddings.shape[0]), C)

    return emd


def get_distances(model, source_dataset, target_dataset):
    """ Compute the distances but with mean and standard deviation """
    distances = []

    for i in range(FLAGS.repeat):
        distances.append(compute_distance(model, source_dataset, target_dataset))

    distances = np.array(distances, dtype=np.float32)

    return distances


def main(argv):
    set_gpu_memory(FLAGS.gpumem)
    model_args = get_model_args()
    model, source_train_data, target_train_data, \
        source_test_data, target_test_data = load_model_and_data(*model_args)

    print("LogDir;ModelDir;Source;Target;Model;Method;DistanceMeanTrain;DistanceStdevTrain;DistanceMeanTest;DistanceStdevTest")
    print(*model_args, sep=";", end=";")

    # Compute distance - mean +/- std
    train_distances = get_distances(model, source_train_data, target_train_data)
    test_distances = get_distances(model, source_test_data, target_test_data)

    print(train_distances.mean(), train_distances.std(ddof=0),
        test_distances.mean(), test_distances.std(ddof=0), sep=";")


if __name__ == "__main__":
    app.run(main)
