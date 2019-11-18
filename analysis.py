#!/usr/bin/env python3
"""
Analyze the results
"""
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from absl import app
from absl import flags
# from scipy import stats

from pool import run_job_pool
from file_utils import get_config

FLAGS = flags.FLAGS

flags.DEFINE_integer("jobs", 0, "Number of jobs to use for processing files (0 == number of cores)")


# Use nice names for the plot
nice_method_names = {
    # No adaptation or training on target
    "none": "None",  # (no adaptation)
    "upper": "Target Only",  # (train on target)

    # Multi-source domain adaptation
    "dann": "MS-DA-DANN",
    "dann_gs": "GS-DA",
    "dann_smooth": "MS-DA-Smooth",

    # Domain generalization
    "dann_dg": "DG-DANN",
    "sleep_dg": "DG-Sleep",
    "aflac_dg": "DG-AFLAC",
    "ciddg_dg": "DG-CIDDG",
}

method_lines = {
    # Approximate bounds
    "none": "-.",
    "upper": "-.",

    # MS-DA solid
    "dann": "-",
    "dann_gs": "-",
    "dann_smooth": "-",

    # DG dashed
    "dann_dg": "--",
    "sleep_dg": "--",
    "aflac_dg": "--",
    "ciddg_dg": "--",
}


def get_tuning_files(dir_name, prefix):
    """ Get all the hyperparameter evaluation result files """
    files = []
    matching = pathlib.Path(dir_name).glob(prefix+"*.txt")

    for m in matching:
        name = m.stem.replace(prefix, "")
        file = str(m)
        files.append((name, file))

    return files


def beginning_match(match, line):
    """ Does the first x=len(match) chars of line match the match string """
    return line[:len(match)] == match


def parse_file(filename):
    """
    Get all of the data from the file

    Several parts:
        - Best validation accuracy per target/fold at a particular step
        - target/fold train/test A/B accuracy
        - averages of train/test A/B accuracies
    """
    in_validation = False
    in_traintest = False
    in_averages = False

    validation = []
    traintest = []
    averages = []

    valid_header = "Log Dir;Dataset;Sources;Target;Model;Method;Best Step;Accuracy at Step"
    traintest_header = "Log Dir;Dataset;Sources;Target;Model;Method;Train A;Test A;Train B;Test B"
    averages_header = "Dataset;Avg;Std"

    # Config information
    log_dir = None
    dataset_name = None
    sources = None
    num_domains = None
    target = None
    model = None
    method = None

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if line == "Virtual devices must be set at program startup":
                pass
            elif line == "Error occurred -- exiting":
                print("Found:", line, "in", filename, file=sys.stderr)
                exit(1)
            elif beginning_match("Source batch size: ", line) or \
                    beginning_match("Target batch size: ", line):
                # Forgot that I print out {Source,Target} batch size: ...
                pass
            elif beginning_match(valid_header, line):
                in_validation = True
                in_traintest = False
                in_averages = False
            elif beginning_match(traintest_header, line):
                in_validation = False
                in_traintest = True
                in_averages = False
            elif beginning_match(averages_header, line):
                in_validation = False
                in_traintest = False
                in_averages = True
            elif len(line) > 0:
                values = line.split(";")

                # For example, if we ran evaluation before we had any models to
                # evaluate, we'd get no data.
                if values[0] == "No data.":
                    return None

                if in_validation:
                    # If there was no model yet (e.g. if a method errors before
                    # starting training)
                    if values[6] == "None":
                        print("Warning: no best model for", filename, file=sys.stderr)
                        return None

                    validation.append((values[0], values[1], values[2],
                        values[3], values[4], values[5],
                        int(values[6]), float(values[7])))

                    # Doesn't really matter which one... just get one of them
                    log_dir = values[0]

                    # Get data, make sure it's not conflicting with previous
                    # values
                    assert dataset_name is None or values[1] == dataset_name
                    dataset_name = values[1]

                    assert sources is None or values[2] == sources
                    sources = values[2]

                    num_domains = len(sources.split(","))

                    assert target is None or values[3] == target
                    target = values[3]

                    if target != "":
                        # Probably 1, but still will work if we ever support
                        # multiple targets
                        num_domains += len(target.split(","))

                    assert model is None or values[4] == model
                    model = values[4]

                    assert method is None or values[5] == method
                    method = values[5]
                elif in_traintest:
                    traintest.append((values[0], values[1], values[2],
                        values[3], values[4], values[5],
                        float(values[6]), float(values[7]),
                        float(values[8]), float(values[9])))
                elif in_averages:
                    averages.append((values[0], float(values[1]), float(values[2])))
            else:
                # Empty lines ends a section
                in_validation = False
                in_traintest = False
                in_averages = False

    validation = pd.DataFrame(data=validation, columns=valid_header.split(";"))
    traintest = pd.DataFrame(data=traintest, columns=traintest_header.split(";"))
    averages = pd.DataFrame(data=averages, columns=averages_header.split(";"))

    # method="upper" doesn't actually exist since it uses "none", but our upper
    # bound is method="none" without any target domains, so set appropriately.
    if method == "none" and target == "":
        method = "upper"

    # Create params
    assert num_domains is not None, \
        "could not find config information in file (probably no runs): " \
        + filename

    # If the config file exists, add it to the parameters. We assume all the
    # runs have the same-ish config (you average over them, so those parts
    # may be different)
    config = get_config(log_dir)

    params = {
        "dataset": dataset_name,
        "sources": sources,
        "num_domains": num_domains,
        "target": target,
        "model": model,
        "method": method,
        "config": config,
    }

    return validation, traintest, averages, params


def compute_mean_std(df, name, filename):
    data = df[name]

    # I expect there to only be 3 or 5 of each... if not, warn
    length = len(data)

    if length != 1 and length != 3 and length != 5:
        print("Warning: number of runs ", length, "(not 3 or 5) for", filename,
            file=sys.stderr)

    # ddof=0 is the numpy default, ddof=1 is Pandas' default
    return data.mean(), data.std(ddof=0)


def compute_val_stats(df, filename):
    return compute_mean_std(df, "Accuracy at Step", filename)


def compute_eval_stats(df, filename):
    names = ["Train A", "Test A", "Train B", "Test B"]
    data = [[name]+list(compute_mean_std(df, name, filename)) for name in names]
    return pd.DataFrame(data=data, columns=["Dataset", "Avg", "Std"])


def _all_stats(name, file, recompute_averages):
    parse_result = parse_file(file)

    if parse_result is None:
        print("Warning: no data, skipping", file, file=sys.stderr)
        return

    validation, traintest, averages, params = parse_result

    if recompute_averages:
        averages = compute_eval_stats(traintest, name)

    validavg = compute_val_stats(validation, name)

    return {
        "name": name,
        "parameters": params,
        "file": file,
        "validation": validation,
        "traintest": traintest,
        "averages": averages,
        "validavg": validavg,
    }


def all_stats(files, recompute_averages=True, sort=False):
    """ Process all files, but since we may have many, many thousands, do it
    with multiple cores by default """
    if FLAGS.jobs == 1:
        results = []

        for name, file in files:
            results.append(_all_stats(name, file, recompute_averages))
    else:
        commands = []

        for name, file in files:
            commands.append((name, file, recompute_averages))

        jobs = FLAGS.jobs if FLAGS.jobs != 0 else None
        results = run_job_pool(_all_stats, commands, cores=jobs)

    # Sort by name
    if sort:
        results.sort(key=lambda x: x["name"])

    return results


def gen_jitter(length, amount=0.04):
    """ "Dodge" the points slightly on the x axis, so that they don't overlap """
    x = []
    value = -(amount/length)/2

    for i in range(length):
        x.append(value)
        value += amount

    return np.array(x, dtype=np.float32)


def export_legend(legend, dir_name=".", filename="key.pdf", expand=[-5, -5, 5, 5]):
    """ See: https://stackoverflow.com/a/47749903 """
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(dir_name, filename), dpi="figure", bbox_inches=bbox)


def make_replacements(s, replacements):
    """ Make a bunch of replacements in a string """
    if s is None:
        return s

    for before, after in replacements:
        s = s.replace(before, after)

    return s


def pretty_dataset_name(dataset):
    """ Make dataset name look good for plots """
    replacements = [
        ("ucihar", "HAR"),
        ("ucihhar", "HHAR"),
        ("ucihm", "HM"),
        ("uwave", "uWave"),
        ("utdata", "UT-Data"),
        ("sleep", "Sleep"),
        ("wisdm_ar", "WISDM AR"),
        ("wisdm_at", "WISDM AT"),
    ]

    return make_replacements(dataset, replacements)


def get_results(results, average=False):
    """ Get results - get the test on target mean and standard deviation values,
    indexed by,
    if average=False: ms_results[dataset_name + " " + target][method][n]
    if average=True:  ms_results[dataset_name][method][n]
    """
    ms_results = {}

    for result in results:
        params = result["parameters"]
        avgs = result["averages"]
        method = params["method"]
        n = params["num_domains"]
        dataset_name = pretty_dataset_name(params["dataset"])

        # Indexed by target, i.e. separate plot per dataset-target. Otherwise,
        # indexed by dataset, i.e. one plot per dataset (averaged over multiple
        # targets).
        if not average:
            # For the upper bound, we don't have a target, but set the source
            # as the target
            if method == "upper":
                target = dataset_name + " " + params["sources"]
            else:
                target = dataset_name + " " + params["target"]
            dataset_name = target

        mean = avgs[avgs["Dataset"] == "Test B"]["Avg"].values[0]
        std = avgs[avgs["Dataset"] == "Test B"]["Std"].values[0]

        # For upper bound, we set the source to the target
        if method == "upper":
            mean = avgs[avgs["Dataset"] == "Test A"]["Avg"].values[0]
            std = avgs[avgs["Dataset"] == "Test A"]["Std"].values[0]

        #print(target, method, seqlen, mean, std, sep=";")

        if dataset_name not in ms_results:
            ms_results[dataset_name] = {}
        if method not in ms_results[dataset_name]:
            ms_results[dataset_name][method] = {}
        if n not in ms_results[dataset_name][method]:
            ms_results[dataset_name][method][n] = []
        ms_results[dataset_name][method][n].append((n, mean, std))

    return ms_results


def average_over_n(ms_results):
    """
    Average over multiple runs (values of n, the number of source domains)
    - Recompute mean/stdev for those that have multiple entries
    - Get rid of the n-specific dictionary
    """
    for dataset, values in ms_results.items():
        for method, n_values in values.items():
            new_values = []

            for n, ms_values in n_values.items():
                if len(ms_values) > 1:
                    ms_values = np.array(ms_values, dtype=np.float32)
                    # All the 0th elements should be the same n
                    # Then recompute the mean/stdev from the accuracy values in 1th column
                    new_values.append((int(ms_values[0, 0]),
                        ms_values[:, 1].mean(), ms_values[:, 1].std(ddof=0)))
                elif len(ms_values) == 1:
                    # Leave as is if there's only one
                    #assert new_values == [], "upper bound has multiple runs?"
                    ms_values = np.array(ms_values, dtype=np.float32)
                    new_values.append((int(ms_values[0, 0]),
                        ms_values[0, 1], ms_values[0, 2]))
                else:
                    raise NotImplementedError("must be several or one run")

            # Sort on n
            new_values.sort(key=lambda x: x[0])

            ms_results[dataset][method] = new_values

    # Make numpy array
    for dataset, values in ms_results.items():
        for method, ms_values in values.items():
            ms_results[dataset][method] = \
                np.array(ms_values, dtype=np.float32)

    return ms_results


def generate_plots(ms_results, prefix, save_plot=True, show_title=False,
        legend_separate=True, suffix="pdf", dir_name="results"):
    # See: https://matplotlib.org/3.1.1/api/markers_api.html
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "D", "P", "X", "h",
        "1", "2", "3", "4", "+", "x"]

    sorted_dataset_names = list(ms_results.keys())
    sorted_dataset_names.sort()

    for dataset_name in sorted_dataset_names:
        dataset_values = ms_results[dataset_name]
        methods = list(dataset_values.keys())
        data = list(dataset_values.values())

        # Find min/max x values for scaling the jittering appropriately
        max_x = -np.inf
        min_x = np.inf
        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0]
            max_x = max(max(x), max_x)
            min_x = min(min(x), min_x)
        x_range = max_x - min_x

        # "dodge" points so they don't overlap
        jitter = gen_jitter(len(data), amount=0.01*x_range)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4.1), dpi=100)

        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0] + jitter[i]
            y = method_data[:, 1]*100
            std = method_data[:, 2]*100
            method_name = nice_method_names[methods[i]]
            p = plt.errorbar(x, y, yerr=std, label=method_name, fmt=markers[i]+method_lines[methods[i]], alpha=0.8)

            # Make a horizontal line at the upper bound since it doesn't matter
            # what "n" is for this method (ignores the sources, only trains
            # on target)
            if methods[i] == "upper":
                # xmin=1 since the upper bound is 1 source in a sense
                assert method_lines[methods[i]] == "-.", \
                    "change linestyles in hlines to match that of method_lines[\"upper\"]"
                ax.hlines(y=y, xmin=1, xmax=max_x, colors=p[0].get_color(),
                    linestyles="dashdot")

        if show_title:
            plt.title("Adaptation and Generalization Methods on "+dataset_name)

        ax.set_xlabel("Number of source domains")
        ax.set_ylabel("Target Domain Accuracy (%)")

        if legend_separate:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            export_legend(legend, dir_name, filename=prefix+"_key."+suffix)
            legend.remove()
        else:
            # Put legend outside the graph http://stackoverflow.com/a/4701285
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if save_plot:
            plt.savefig(os.path.join(dir_name,
                prefix+"_"+dataset_name+"."+suffix), bbox_inches='tight')
            plt.close()

    if not save_plot:
        plt.show()


def plot_multisource(dataset, variant, variant_match=None, save_plot=True,
        show_title=False, legend_separate=True, suffix="pdf"):
    """ Generate plots of target accuracy vs. number of source domains """
    if variant_match is None:
        variant_match = variant

    files = get_tuning_files("results", prefix="results_"+dataset+"_"+variant_match+"-")
    results = all_stats(files)
    ms_results = average_over_n(get_results(results, average=False))
    ms_averages = average_over_n(get_results(results, average=True))
    generate_plots(ms_results, "multisource_"+variant, save_plot, show_title,
        legend_separate, suffix)
    generate_plots(ms_averages, "multisource_average_"+variant, save_plot,
        show_title, legend_separate, suffix)


def main(argv):
    plot_multisource("vary_n_best_source", "best_source",
        save_plot=True, show_title=True,
        legend_separate=True, suffix="pdf")

    # We pass variant=best_target, but match * variant since for the upper bound
    # there isn't a "target" (since target is passed as source), but we all the
    # others we evaluate only with best_target, so we can match all to get the
    # best_source only for the upper bound.
    plot_multisource("vary_n_best_target", "best_target", "*",
        save_plot=True, show_title=True,
        legend_separate=True, suffix="pdf")


if __name__ == "__main__":
    app.run(main)
