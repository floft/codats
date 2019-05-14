"""
Functions for handling loading/saving files
"""
import os
import re
import pathlib
import numpy as np


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


def get_best_target_valid_accuracy(log_dir):
    """ Best accuracy based on target classifier """
    return get_best_valid_accuracy(log_dir, "best_target_valid_accuracy.txt")


def write_best_valid_accuracy(log_dir, accuracy,
        filename="best_valid_accuracy.txt"):
    """ Write the best validation accuracy to a file """
    filename = os.path.join(log_dir, filename)

    with open(filename, "w") as f:
        f.write(str(accuracy))


def write_best_target_valid_accuracy(log_dir, accuracy):
    """ Write best accuracy based on target classifier """
    return write_best_valid_accuracy(log_dir, accuracy,
        "best_target_valid_accuracy.txt")


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
