"""
Functions for handling loading/saving files
"""
import os
import re
import yaml
import pathlib
import numpy as np

from absl import flags


FLAGS = flags.FLAGS


def get_last_int(s, only_one=False):
    """
    Get last integer in a string

    If only_one==True, then assert there's only one number
    """
    regex = re.compile(r'\d+')
    numbers = [int(x) for x in regex.findall(s)]

    if only_one:
        assert len(numbers) == 1, \
            "get_last_int() should not match more than one integer"

    if len(numbers) == 0:
        return None

    return numbers[-1]


def last_modified_number(dir_name, glob):
    """
    Looks in dir_name at all files matching glob and takes number
    from the one last modified
    """
    files = pathlib.Path(dir_name).glob(glob)
    files = sorted(files, key=lambda cp: cp.stat().st_mtime)

    if len(files) > 0:
        # Get number from filename
        return get_last_int(str(files[-1]), only_one=True)

    return None


def last_modified(dir_name, glob):
    """
    Looks in dir_name at all files matching glob and returns the file last
    modified
    """
    files = pathlib.Path(dir_name).glob(glob)
    files = sorted(files, key=lambda cp: cp.stat().st_mtime)

    if len(files) > 0:
        return str(files[-1])

    return None


def get_best_valid_accuracy(log_dir, filename="best_valid_accuracy.txt"):
    """
    Read in the best validation accuracy from the best_valid_accuracy.txt file
    in the log_dir, if it exists. If it doesn't, return None.
    """
    filename = os.path.join(log_dir, filename)

    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                try:
                    return float(line)
                except ValueError:
                    pass

    return None


def write_best_valid_accuracy(log_dir, accuracy,
        filename="best_valid_accuracy.txt"):
    """ Write the best validation accuracy to a file """
    filename = os.path.join(log_dir, filename)

    with open(filename, "w") as f:
        f.write(str(accuracy))


def get_finished(log_dir):
    """ Does the file indicating completion exist? """
    filename = os.path.join(log_dir, "finished.txt")
    return os.path.exists(filename)


def get_log_subdirs(log_dir):
    """ Get all the subdirectories of the log_dir """
    dirs = []

    if os.path.exists(log_dir):
        for d in os.listdir(log_dir):
            subdir = os.path.join(log_dir, d)

            if os.path.isdir(subdir):
                dirs.append(subdir)

    return dirs


def get_average_valid_accuracy(log_dir):
    """ Get average of the best_valid_accuracy.txt files for all subdirs """
    dirs = get_log_subdirs(log_dir)
    accs = []

    for d in dirs:
        acc = get_best_valid_accuracy(d)

        if acc is not None:
            accs.append(acc)

    return np.array(accs).mean()


def get_num_finished(log_dir):
    """
    Count the number of files indicating completion in subdirectories
    of the specified log directory
    """
    dirs = get_log_subdirs(log_dir)

    # Count the number that have finished.txt in them
    num_finished = 0

    for d in dirs:
        if get_finished(d):
            num_finished += 1

    return num_finished


def write_finished(log_dir):
    """ Write the file indicating completion """
    filename = os.path.join(log_dir, "finished.txt")

    with open(filename, "w") as f:
        f.write("\n")


def get_config(log_dir):
    """ Get config file containing dataset name, sources, target, etc. """
    filename = os.path.join(log_dir, "config.yaml")

    with open(filename) as f:
        # See: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config


def write_config_from_args(log_dir):
    """ Save config file containing dataset name, sources, target, etc. """
    filename = os.path.join(log_dir, "config.yaml")

    # Log everything -- get list from ./main.py --helpfull
    config = {
        # main.py
        "dataset": FLAGS.dataset,
        "debug": FLAGS.debug,
        "debugnum": FLAGS.debugnum,
        "gpumem": FLAGS.gpumem,
        "log_plots_steps": FLAGS.log_plots_steps,
        "log_train_steps": FLAGS.log_train_steps,
        "log_val_steps": FLAGS.log_val_steps,
        "logdir": FLAGS.logdir,
        "method": FLAGS.method,
        "model_steps": FLAGS.model_steps,
        "modeldir": FLAGS.modeldir,
        "sources": [int(x) for x in FLAGS.sources.split(",")],
        "steps": FLAGS.steps,
        "subdir": FLAGS.subdir,
        "target": int(FLAGS.target) if FLAGS.target != "" else None,
        "test": FLAGS.test,
        "uid": FLAGS.uid,

        # checkpoints.py
        "best_checkpoints": FLAGS.best_checkpoints,
        "latest_checkpoints": FLAGS.latest_checkpoints,

        # dataset.py
        "normalize": FLAGS.normalize,

        # load_datasets.py
        "batch_division": FLAGS.batch_division,
        "cache": FLAGS.cache,
        "eval_batch": FLAGS.eval_batch,
        "eval_max_examples": FLAGS.eval_max_examples,
        "eval_shuffle_seed": FLAGS.eval_shuffle_seed,
        "feature_subset": FLAGS.feature_subset,
        "max_target_examples": FLAGS.max_target_examples,
        "prefetch_buffer": FLAGS.prefetch_buffer,
        "shuffle_buffer": FLAGS.shuffle_buffer,
        "train_batch": FLAGS.train_batch,
        "train_max_examples": FLAGS.train_max_examples,
        "trim_time_steps": FLAGS.trim_time_steps,
        "tune_num_parallel_calls": FLAGS.tune_num_parallel_calls,

        # methods.py
        "lr": FLAGS.lr,
        "lr_domain_mult": FLAGS.lr_domain_mult,
        "model": FLAGS.model,

        # plots.py
        "max_plot_embedding": FLAGS.max_plot_embedding,
    }

    with open(filename, "w") as f:
        yaml.dump(config, f)
