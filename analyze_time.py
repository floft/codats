#!/usr/bin/env python3
"""
Calculate average training time for each method on a particular dataset,
for comparing how fast each is

Run timing test:
    sbatch -J timing kamiak_timing_real.srun timing1

Then download results (SLURM output logs):
    . kamiak_config.sh
    mkdir slurm_logs
    scp kamiak:"$remotedir"/slurm_logs/timing_10770594_*.out slurm_logs/

Finally, run this script passing in the job number:
    ./analyze_time.py --jobnum=10770594 --latex
"""
import pathlib
import numpy as np

from absl import app
from absl import flags

from analysis import nice_method_names

FLAGS = flags.FLAGS

flags.DEFINE_string("jobnum", None, "Will look for slurm_logs/timing_<jobnum>_*.out")
flags.DEFINE_boolean("latex", False, "Print LaTex code for paper")
flags.mark_flag_as_required("jobnum")


def parse_file(filename):
    method_name = None
    source_name = None
    times = []

    with open(filename) as f:
        for line in f:
            line = line.strip()

            # Skip warning lines
            if "Warning" in line:
                continue

            parts = line.split(",")

            # Times, skip the first one
            if len(parts) == 2:
                if int(parts[0]) > 1:
                    times.append(float(parts[1]))
            elif len(parts) == 6:
                method_name = parts[1]
                source_name = parts[2]

    return method_name, source_name, times


def format_stats(data):
    """ Pretty print mean +/- stdev """
    if FLAGS.latex:
        sep = "$\\pm$"
    else:
        sep = "+/-"

    data = np.array(data, dtype=np.float32)
    # ddof=0 is the numpy default, ddof=1 is Pandas' default
    return "{:.3f}".format(data.mean()) + " " + sep + " " + \
        "{:.3f}".format(data.std(ddof=0))


def main(argv):
    files = pathlib.Path("slurm_logs").glob("timing_"+FLAGS.jobnum+"_*.out")

    timings = {}

    for filename in files:
        method, dataset, times = parse_file(filename)

        if method not in timings:
            timings[method] = {}
        if dataset not in timings[method]:
            timings[method][dataset] = []

        timings[method][dataset] += times

    if FLAGS.latex:
        methods = [
            "none", "random", "rdann", "vrada", "dann_grl", "dann", "cycada",
            "deepjdot", "upper",
        ]
        for method in methods:
            columns = [
                format_stats(timings[method]["ucihar_1"]),
                format_stats(timings[method]["utdata_wrist"]),
                format_stats(timings[method]["uwave_1"]),
            ]
            name = nice_method_names[method]

            if "CoDATS" in name:
                name = "\\textit{"+name+"}"
                name = name.replace(" + ", "+")

            print(name, *columns, sep=" & ", end="")
            print(" \\\\")
    else:
        for method, datasets in timings.items():
            print(nice_method_names[method])

            for dataset, times in datasets.items():
                print(dataset.ljust(15), format_stats(times))

            print()


if __name__ == "__main__":
    app.run(main)