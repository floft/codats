#!/usr/bin/env python3
"""
Calculate average training time for each method on a particular dataset,
for comparing how fast each is

Run timing test:
    sbatch -J time1 kamiak_timing_ssda.srun timing1

Then download results (SLURM output logs):
    . kamiak_config.sh
    mkdir slurm_logs
    scp kamiak:"$remotedir"/slurm_logs/timing_15670129_*.out slurm_logs/
    scp kamiak:"$remotedir"/slurm_logs/timing_15670177_*.out slurm_logs/
    scp kamiak:"$remotedir"/slurm_logs/timing_15738526_*.out slurm_logs/
    ...

Finally, run this script passing in the job number:
    ./analyze_time.py --jobnum=15670129,15670177,15738526 --latex
"""
import pathlib
import numpy as np

from absl import app
from absl import flags

from analysis import nice_method_names

FLAGS = flags.FLAGS

flags.DEFINE_string("jobnum", None, "Will look for slurm_logs/timing_<jobnum>_*.out; can be comma separated")
flags.DEFINE_boolean("latex", False, "Print LaTex code for paper")
flags.mark_flag_as_required("jobnum")


def parse_file(filename):
    #adaptation = None
    dataset = None
    times = []

    with open(filename) as f:
        for line in f:
            line = line.strip()

            # Skip warning lines
            if "Warning" in line or "Debugging attempt" in line:
                continue

            # It's the line describing the dataset, sources, target, etc.
            if ";" in line:
                parts = line.split(";")
                assert len(parts) == 8
                _, method, dataset, sources, target, _, _, _ = parts
                #adaptation = dataset + " " + sources + " --> " + target
            else:
                parts = line.split(",")
                assert len(parts) == 2
                times.append(float(parts[1]))

    return method, dataset, times


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
    jobnums = FLAGS.jobnum.split(",")
    files = []

    for num in jobnums:
        files += list(pathlib.Path("slurm_logs").glob("timing_"+num+"_*.out"))

    timings = {}

    for filename in files:
        method, dataset, times = parse_file(filename)

        if method not in timings:
            timings[method] = {}
        if dataset not in timings[method]:
            timings[method][dataset] = []

        timings[method][dataset] += times

    if FLAGS.latex:
        methods = [  # note "dann" is CoDATS
            "none", "rdann", "vrada", "dann", "daws",
        ]
        for method in methods:
            columns = [
                format_stats(timings[method]["ucihar"]),
                format_stats(timings[method]["ucihhar"]),
                format_stats(timings[method]["uwave"]),
            ]
            name = nice_method_names[method]

            if "CoDATS" in name or "DA-WS" in name:
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
