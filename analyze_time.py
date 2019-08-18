#!/usr/bin/env python3
"""
Calculate average training time for each method on a particular dataset,
for comparing how fast each is
"""
import numpy as np

from analysis import nice_method_names

# TODO none includes both lower and upper at the moment...


def calculate_duration(duration):
    """ From the string hh:mm:ss calculate number of minutes """
    h, m, s = duration.split(":")
    h = int(h)
    m = int(m)
    s = int(s)
    return h*60 + m + s/60


def calculate_training_time(filename):
    """ Open file, create dictionary indexed by method name with list if number
    of minutes each method took """

    training_times = {name: [] for name in nice_method_names.keys()}

    with open(filename) as f:
        for line in f:
            method, duration = line.strip().split(" ")
            training_times[method].append(calculate_duration(duration))

    for name, durations in training_times.items():
        if len(durations) > 0:
            durations = np.array(durations, dtype=np.float32)

            nice_name = nice_method_names[name]

            # Both use the "none" method, and times are about the same
            if nice_name == "Lower Bound":
                nice_name = "Lower/Upper Bound"

            # ddof=0 is the numpy default, ddof=1 is Pandas' default
            print(nice_name + " ("+str(len(durations))+"): ",
                durations.mean(), "+/-", durations.std(ddof=0), "min.")


if __name__ == "__main__":
    calculate_training_time("training_time.txt")
