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

flags.DEFINE_boolean("ignore_old_config", False, "Ignore checking that subsequent training of a model uses the same config as previously")


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


def get_best_valid(log_dir, filename="best_valid_accuracy.txt"):
    """
    Read in the best validation accuracy/mse/etc. from the
    best_valid_accuracy.txt file in the log_dir, if it exists. If it doesn't,
    return None.
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


def write_best_valid(log_dir, value,
        filename="best_valid_accuracy.txt"):
    """ Write the best validation accuracy/mse/etc. to a file """
    filename = os.path.join(log_dir, filename)

    with open(filename, "w") as f:
        f.write(str(value))


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


def get_average_valid(log_dir):
    """ Get average of the best_valid_accuracy.txt files for all subdirs """
    dirs = get_log_subdirs(log_dir)
    values = []

    for d in dirs:
        value = get_best_valid(d)

        if value is not None:
            values.append(value)

    return np.array(values).mean()


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

    if not os.path.exists(filename):
        return None

    with open(filename) as f:
        # See: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        return yaml.load(f, Loader=yaml.SafeLoader)


def write_config_from_args(log_dir):
    """ Save config file containing all flags """
    config_filename = os.path.join(log_dir, "config.yaml")
    flag_filename = os.path.join(log_dir, "config.flags")

    # Rather than maintaining an ever-changing list of the flags from this
    # code, just get all of the ones that aren't absl or tensorflow related.
    config = {}

    # Note: we could use __flags_by_module directly, but there appears to be a
    # function flags_by_module_dict() that returns this, which hopefully is
    # less likely to change/disappear in the future.
    for module, module_flags in FLAGS.flags_by_module_dict().items():
        if "absl" not in module and "tensorflow" not in module:
            for module_flag in module_flags:
                name = module_flag.name
                # Note: this shouldn't change ever since they say it's always
                # FLAGS.name to access the flag.
                assert name not in config, \
                    "duplicate flag name should not be possible"
                config[name] = getattr(FLAGS, name)

    # If we're continuing training from a previous run (e.g. training was
    # preempted), check that the old config matches the new config. Otherwise,
    # we may accidentally have a uid conflict and mess up the trained model.
    # It's better to clearly error so we can fix the conflict.
    #
    # Note: we check for the three edit cases: insertion, deletion, edit
    old_config = get_config(log_dir)

    if old_config is not None and not FLAGS.ignore_old_config:
        for key, old_value in old_config.items():
            # Skip the flag that tells us whether or not to ignore these
            # differences. We want to be able to go back and forth on that.
            if key == "ignore_old_config":
                continue

            # Check old key is in the new config
            assert key in config, \
                "mismatch in old/new config for key \"" + str(key) + "\": " \
                + "missing key in new config"

            # Check old/new values are the same
            new_value = config[key]
            assert new_value == old_value, \
                "mismatch in old/new config for key \"" + str(key) + "\": " \
                + "old = " + str(old_value) + ", " \
                + "new = " + str(new_value)

        # Check new keys are in old config
        for key, new_value in config.items():
            assert key in old_config, \
                "mismatch in old/new config for key \"" + str(key) + "\": " \
                + "missing key in old config"

    # Write the config file
    with open(config_filename, "w") as f:
        yaml.dump(config, f)

    # Also write out the flags into a file that can easily be loaded with
    # --flagfile=config.flags
    with open(flag_filename, "w") as f:
        f.write(FLAGS.flags_into_string())
