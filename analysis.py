#!/usr/bin/env python3
"""
Analyze the results
"""
import os
import sys
import yaml
import pathlib
import collections
import numpy as np
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from scipy import stats
from matplotlib.ticker import MaxNLocator

from pool import run_job_pool

FLAGS = flags.FLAGS

flags.DEFINE_integer("jobs", 0, "Number of jobs to use for processing files (0 == number of cores)")
flags.DEFINE_bool("paper", False, "Whether to make paper-version plots (e.g. .pdf not .png), outputs to result_plots_paper")
flags.DEFINE_enum("metric", "accuracy", ["accuracy"], "Which metric to plot")  # TODO add F1 score


# Use nice names for the plot
nice_method_names = {
    # No adaptation or training on target
    "none": "No Adaptation",
    "upper": "Train on Target",

    # Domain adaptation
    "rdann": "R-DANN",
    "vrada": "VRADA",
    "dann": "CoDATS",

    # Domain adaptation with weak supervision
    "daws": "CoDATS-WS",

    # Multi-source domain adaptation
    "dann_gs": "GS-DA",
    "dann_smooth": "MS-DA-Smooth",

    # Domain generalization
    "dann_dg": "DG-DANN",
    "sleep_dg": "DG-Sleep",
    "aflac_dg": "DG-AFLAC",
    "ciddg_dg": "DG-CIDDG",

    # Heterogeneous DA
    "dann_hda": "CoDATS-H",
    "dann_pad": "Pad-DANN-H",
    "dann_drop": "Drop-DANN-H",
    "none_pad": "Pad-None-H",
    "none_drop": "Drop-None-H",
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

nice_metric_names = {
    "accuracy": "Accuracy (%)",
    "f1score_macro": "F1 Score (Macro)",
}

dataset_replacements = [
    ("watch_noother", "LABNAME AR"),
    ("ucihar", "HAR"),
    ("ucihhar", "HHAR"),
    ("ucihm", "HM"),
    ("uwave", "uWave"),
    ("utdata", "UT-Data"),
    ("sleep", "Sleep"),
    ("wisdm_ar", "WISDM AR"),
    ("wisdm_at", "WISDM AT"),
]


def get_tuning_files(dir_name, prefix):
    """ Get all the hyperparameter evaluation result files """
    files = []
    matching = pathlib.Path(dir_name).glob(prefix+"*.yaml")

    for m in matching:
        name = m.stem.replace(prefix, "")
        file = str(m)
        files.append((name, file))

    return files


def compute_average(name, data, metric, domain, train_or_valid):
    results = []

    for d in data:
        # Make sure this value exists in the evaluation results .yaml file
        assert "results" in d, \
            "No results in: " + str(d) + " for " + name
        name_of_value = metric+"_task/"+domain+"/"+train_or_valid
        assert name_of_value in d["results"], \
            "No metric value " + name_of_value + " in: " + str(d["results"]) \
            + " for " + name

        result = d["results"][name_of_value]
        results.append(result)

    # There should be 1 or 3 of each; if not, warn
    length = len(results)

    if length != 1 and length != 3:
        print("Warning: number of runs ", length, "(not 1 or 3) for", name,
            file=sys.stderr)

    # ddof=0 is the numpy default, ddof=1 is Pandas' default
    results = np.array(results, dtype=np.float32)
    return results.mean(), results.std(ddof=0), results


def get_method(method, target):
    """
    method="upper" doesn't actually exist since it uses method="none", but
    our upper bound is method="none" without any target domains, so set
    appropriately.
    """
    if method == "none" and target == "":
        method = "upper"

    return method


def _all_stats(name, filename, source_feature_subset, target_feature_subset):
    with open(filename) as f:
        # See: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Get some of the config
    dataset = None
    method = None
    sources = None
    target = None

    if len(data) == 0:
        print("Warning: no data in file", filename)
        return {}

    for d in data:
        config = d["config"]

        assert dataset is None or config["dataset"] == dataset, \
            "runs disagree on dataset: " \
            + config["dataset"] + " vs. " + str(dataset)
        dataset = config["dataset"]

        assert sources is None or config["sources"] == sources, \
            "runs disagree on sources: " \
            + config["sources"] + " vs. " + str(sources)
        sources = config["sources"]

        assert target is None or config["target"] == target, \
            "runs disagree on target: " \
            + config["target"] + " vs. " + str(target)
        target = config["target"]

        new_method = get_method(config["method"], target)
        assert method is None or new_method == method, \
            "runs disagree on method: " + new_method + " vs. " + str(method)
        method = new_method

        # Skip if not the right source/target features
        current_source_feature_subset = config["source_feature_subset"]
        current_target_feature_subset = config["target_feature_subset"]

        if source_feature_subset is not None \
                and source_feature_subset != current_source_feature_subset:
            return {}

        if target_feature_subset is not None \
                and target_feature_subset != current_target_feature_subset:
            return {}

    results = {
        "name": name,
        "dataset": dataset,
        "method": method,
        "sources": sources,
        "target": target,
        # Full data if we need it
        # "data": data,  # not used...
    }

    # For upper bound, there's no target, so instead use the "source" value
    # as the "target" value
    if method == "upper":
        source_or_target = "source"
    else:
        source_or_target = "target"

    # results["results_source_train"] = compute_average(name, data, FLAGS.metric, "source", "training")
    # results["results_source_test"] = compute_average(name, data, FLAGS.metric, "source", "validation")
    # results["results_target_train"] = compute_average(name, data, FLAGS.metric, "target", "training")
    results["results_target_test"] = compute_average(name, data, FLAGS.metric, source_or_target, "validation")

    return results


def all_stats(files, source_feature_subset, target_feature_subset):
    """ Process all files, but since we may have many, many thousands, do it
    with multiple cores by default """
    if FLAGS.jobs == 1:
        results = []

        for name, filename in files:
            results.append(_all_stats(name, filename,
                source_feature_subset, target_feature_subset))
    else:
        commands = []

        for name, filename in files:
            commands.append((name, filename, source_feature_subset,
                target_feature_subset))

        jobs = FLAGS.jobs if FLAGS.jobs != 0 else None
        results = run_job_pool(_all_stats, commands, cores=jobs)

    # Remove empty dictionaries (the "no data" cases)
    results = [r for r in results if r != {}]

    # Sort by name
    results.sort(key=lambda x: x["name"])

    return results


def get_results(dataset, variant, variant_match, source_feature_subset,
        target_feature_subset):
    """ Get the right result files and load them """
    if variant_match is None:
        variant_match = variant

    files = get_tuning_files("results",
        prefix="results_"+dataset+"_"+variant_match+"-")
    results = all_stats(files, source_feature_subset, target_feature_subset)

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


def pretty_dataset_name(dataset_name):
    """ Make dataset name look good for plots """
    return make_replacements(dataset_name, dataset_replacements)


def average_over_n(results):
    """ Average over multiple runs (values of n, the number of source domains)

    - Recompute mean/stdev for those that have multiple entries
    - Get rid of the n-specific dictionary

    i.e. we go from:
        results[dataset_name][method][n] = [
            (n, mean, std), ...
        ]
    to
        averaged_results[dataset_name][method] = [
            (n, mean, std), ...
        ]
    """
    # averaged_results[dataset_name][method] = []
    averaged_results = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for dataset_name, v1 in results.items():
        for method_name, v2 in v1.items():
            new_values = []

            for n, values in v2.items():
                # Average over the multiple values here and recompute
                # the standard deviation
                if len(values) > 1:
                    values = np.array(values, dtype=np.float32)
                    # All the 0th elements should be the same n
                    # Then recompute the mean/stdev from the accuracy values
                    # in 1th column
                    new_values.append((values[0, 0], values[:, 1].mean(),
                        values[:, 1].std(ddof=0)))
                elif len(values) == 1:
                    # Leave as is if there's only one
                    values = np.array(values, dtype=np.float32)
                    new_values.append((values[0, 0], values[0, 1],
                        values[0, 2]))
                else:
                    raise NotImplementedError("must be several or one run")

            # Sort on n
            new_values.sort(key=lambda x: x[0])

            averaged_results[dataset_name][method_name] = \
                np.array(new_values, dtype=np.float32)

    return averaged_results


def process_results(results, average_over_users, ssda):
    """ Get results - get the test mean/std results indexed by:

        if not average, not ssda (i.e. msda):
            results[dataset_name + " " + target][method]
        if not average, ssda:
            results[(dataset_name, source(s), target)][method]
        if average, not ssda (i.e. msda):
            results[dataset_name][method]
        if average, ssda:
            results[dataset_name][method]

    Note: for example, dataset_name="ucihar", sources="1", target="2", and
    method="dann".
    """
    # results[dataset_name][method][n] = []
    # Note: at the end we average over the "n" dictionary
    processed_results = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
    )

    for result in results:
        dataset_name = result["dataset"]
        dataset_name = pretty_dataset_name(dataset_name)

        # For single-source domain adaptation, we create a table for each
        # source -> target pair, so we need index by that.
        if ssda and not average_over_users:
            dataset_name = (dataset_name, result["sources"], result["target"])
        elif not average_over_users:
            # For upper bound, we set target = source, so to name it use what
            # we used as the "source" during training.
            if result["method"] == "upper":
                dataset_name += " " + result["sources"]
            else:
                dataset_name += " " + result["target"]

        method_name = result["method"]
        n = len(result["sources"].split(","))  # number of source domains
        # We care about the target domain (note for the upper bound, we
        # replaced the "target" value with "source" in _all_stats())
        mean, std, all_values = result["results_target_test"]

        processed_results[dataset_name][method_name][n].append(
            (n, mean, std))

        # Keep sorted by n
        processed_results[dataset_name][method_name][n].sort(key=lambda x: x[0])

    # Get rid of the n dictionary and average over the multiple values (should
    # only be >1 if average_over_users==True)
    processed_results = average_over_n(processed_results)

    return processed_results


def dictionary_sorted_keys(d):
    keys = list(d.keys())
    keys.sort()
    return keys


def generate_plots(results, prefix, save_plot=True, show_title=False,
        legend_separate=True, suffix="pdf", dir_name="result_plots",
        error_bars=True, figsize=(5, 3), skip=[], yrange=None,
        integer_axis=False, ncol=1, jitter_amount=0.01,
        x_is_percentage=False, y_is_percentage=True):
    # See: https://matplotlib.org/3.1.1/api/markers_api.html
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "D", "P", "X", "h",
       "1", "2", "3", "4", "+", "x", "d", "H", "|", "_"] * 2
    hollow = [False] * len(markers)
    # markers = ["o", "x", "*", "d"]
    # hollow = [True, False, False, False]

    # Do this sorted by name for a consistent ordering
    for dataset_name in dictionary_sorted_keys(results):
        dataset_values = results[dataset_name]
        methods = dictionary_sorted_keys(dataset_values)

        # Get data in order of the sorted methods
        data = [dataset_values[m] for m in methods]

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
        jitter = gen_jitter(len(data), amount=jitter_amount*x_range)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)

        if yrange is not None:
            ax.set_ylim(yrange)

        # Only integers on x axis
        # https://stackoverflow.com/a/38096332
        if integer_axis:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0] + jitter[i]
            y = method_data[:, 1]
            std = method_data[:, 2]

            if x_is_percentage:
                x *= 100
            if y_is_percentage:
                y *= 100
                std *= 100

            if methods[i] in skip:
                continue

            if methods[i] in nice_method_names:
                method_name = nice_method_names[methods[i]]
            else:
                method_name = methods[i]

            if methods[i] in method_lines:
                line_type = method_lines[methods[i]]
            else:
                line_type = "-"

            if hollow[i]:
                mfc = "None"
            else:
                mfc = None

            if error_bars:
                p = plt.errorbar(x, y, yerr=std, label=method_name,
                    fmt=markers[i]+line_type, alpha=0.8, markerfacecolor=mfc)
            else:
                p = plt.plot(x, y, markers[i]+line_type, label=method_name,
                    alpha=0.8, markerfacecolor=mfc)

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
            plt.title("Dataset: " + dataset_name)

        ax.set_xlabel("Number of source domains")
        ax.set_ylabel("Target Domain " + nice_metric_names[FLAGS.metric])

        if legend_separate:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol)
            export_legend(legend, dir_name, filename=prefix+"_key."+suffix)
            legend.remove()
        else:
            # Put legend outside the graph http://stackoverflow.com/a/4701285
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol)

        if save_plot:
            save_dataset_name = dataset_name.replace(" ", "_")
            filename = prefix + "_" + save_dataset_name + "_" \
                + FLAGS.metric + "."+suffix
            plt.savefig(os.path.join(dir_name, filename),
                bbox_inches='tight')
            plt.close()

    if not save_plot:
        plt.show()


def make_plots(run_suffix, variant, variant_match=None, save_plot=True,
        show_title=False, legend_separate=True, ncol=4, suffix="pdf",
        skip=[], figsize=(5, 3), dir_name="result_plots",
        jitter_amount=0.005, source_feature_subset=None,
        target_feature_subset=None):
    """ Load files, process, save plots """
    results = get_results(run_suffix, variant, variant_match,
        source_feature_subset, target_feature_subset)
    averages = process_results(results, average_over_users=True, ssda=False)
    generate_plots(averages, run_suffix, save_plot,
        show_title, legend_separate, suffix, ncol=ncol, skip=skip,
        figsize=figsize, dir_name=dir_name, jitter_amount=jitter_amount)


def replace_highest_bold(values):
    """ Replace highest DDD.D $\pm$ DDD.D with \textbf{...} """
    max_index = []
    max_value = None

    for i, v in enumerate(values):
        if len(v) > 0:
            parts = v.split(" $\pm$ ")

            if len(parts) == 1 or len(parts) == 2:
                if "underline{" in parts[0]:
                    parts[0] = parts[0].replace("\\underline{", "")
                    parts[1] = parts[1].replace("?", "")

                float_value = float(parts[0])

                if max_value is None or float_value > max_value:
                    max_value = float_value
                    max_index = [i]
                elif float_value == max_value:
                    max_index.append(i)

    if max_index is not None:
        new_values = []

        for i, v in enumerate(values):
            if i in max_index:
                new_values.append("\\textbf{"+v+"}")
            else:
                new_values.append(v)

        return new_values
    else:
        return values


def compute_significance(results, significance_level=0.05, average=False,
        with_vrada=False, with_codats=False):
    """ Calculate significance:

    For each CoDATS method, is the mean significantly different than the
    mean of both of the RNN methods (separately compare with R-DANN and VRADA
    and it's significant if p<0.05 for both)

    Note: only works for real data
    """
    # datasets[dataset_name][method] = ...
    datasets = collections.defaultdict(dict)

    for result in results:
        method = result["method"]
        mean, std, all_values = result["results_target_test"]
        target_accuracies = all_values

        # Skip
        if method == "upper" or method == "lower" or method == "random":
            continue

        dataset_name = pretty_dataset_name(result["dataset"])

        if average:
            dataset = dataset_name
        else:
            dataset = (dataset_name, result["sources"], result["target"])

        method = nice_method_names[method]

        if method in datasets[dataset]:
            datasets[dataset][method] = np.hstack((
                datasets[dataset][method], target_accuracies))
        else:
            datasets[dataset][method] = target_accuracies

    significantly_better = {}

    for dataset, values in datasets.items():
        codats = None
        daws = None

        if with_codats:
            if "CoDATS" in values:
                if "CoDATS-WS" in values:
                    daws = \
                        stats.ttest_rel(values["CoDATS"], values["CoDATS-WS"]).pvalue < significance_level
                else:
                    print("Warning: no CoDATS so no significance")
            else:
                print("Warning: no R-DANN/VRADA so no significance", file=sys.stderr)

        elif with_vrada:
            if "R-DANN" in values and "VRADA" in values:
                if "CoDATS" in values:
                    codats = \
                        stats.ttest_rel(values["R-DANN"], values["CoDATS"]).pvalue < significance_level and \
                        stats.ttest_rel(values["VRADA"], values["CoDATS"]).pvalue < significance_level
                    # codats = \
                    #     stats.ttest_rel(values["VRADA"], values["CoDATS"]).pvalue < significance_level
                else:
                    print("Warning: no CoDATS so no significance")

                if "CoDATS-WS" in values:
                    daws = \
                        stats.ttest_rel(values["R-DANN"], values["CoDATS-WS"]).pvalue < significance_level and \
                        stats.ttest_rel(values["VRADA"], values["CoDATS-WS"]).pvalue < significance_level
                else:
                    print("Warning: no CoDATS-WS so no significance")
            else:
                print("Warning: no R-DANN/VRADA so no significance", file=sys.stderr)

        significantly_better[dataset] = {
            "CoDATS": codats,
            "CoDATS-WS": daws,
        }

    return significantly_better


def write_table(output_filename, table, replace_bold=None):
    """ Write Latex table to file, bold highest row if (row_start, row_end)
    inclusive """
    with open(output_filename, "w") as f:
        for row in table:
            # \hline's
            if len(row) == 1:
                f.write(row[0]+"\n")
                continue

            # Bold between columns if desired
            if replace_bold is not None:
                try:
                    row_start, row_end = replace_bold
                    row[row_start:row_end+1] = replace_highest_bold(row[row_start:row_end+1])
                except ValueError:
                    # If it's the header... ignore the error
                    pass

            for i, column in enumerate(row):
                f.write(column+" ")

                if i == len(row)-1:
                    f.write("\\\\\n")
                else:
                    f.write("& ")


def output_latex_ss_results(results, output_filename):
    """ There's >350 values to fill in... I'm not going to manually type that
    in LaTex, especially when I'll have to do it more than once. This is not
    clean code per se. """
    significantly_better = compute_significance(results, with_vrada=True)
    significantly_better_avg = compute_significance(results, average=True, with_vrada=True)

    # index tuple: (dataset, source, target)
    datasets = process_results(results, average_over_users=False, ssda=True)
    # index tuple: dataset
    averaged = process_results(results, average_over_users=True, ssda=True)

    #
    # Per-adaptation-problem results
    #
    indexed_by_target = {}
    keys = list(datasets.keys())
    keys.sort(key=lambda x: [x[0], x[2], x[1]])  # sort dataset, target, source

    for key in keys:
        dataset, source, target = key
        data = datasets[key]
        indexed_by_target[key] = {}

        for method, values in data.items():
            method = nice_method_names[method]

            # It's a list of a list, but we want the inner list,
            # e.g. it's something like [[2, 1, 0]]
            assert len(values) == 1, "values not length one: " + str(values) \
                + " for method " + method
            values = values[0]

            # From process_results(), [1] is mean and [2] is stdev
            indexed_by_target[key][method] = "{:.1f} $\\pm$ {:.1f}".format(values[1]*100, values[2]*100)

            # Check for significance
            if key in significantly_better and \
                    method in significantly_better[key] and \
                    significantly_better[key][method]:
                indexed_by_target[key][method] = \
                    "\\underline{" + indexed_by_target[key][method] + "}"

    #
    # Averaged over each dataset results
    #
    indexed_by_target_avg = {}
    keys_avg = list(averaged.keys())
    keys_avg.sort()

    for dataset in keys_avg:
        data = averaged[dataset]
        indexed_by_target_avg[dataset] = {}

        for method, values in data.items():
            method = nice_method_names[method]

            # It's a list of a list, but we want the inner list,
            # e.g. it's something like [[2, 1, 0]]
            assert len(values) == 1, "values not length one: " + str(values) \
                + " for method " + method
            values = values[0]

            # From process_results(), [1] is mean and [2] is stdev
            indexed_by_target_avg[dataset][method] = "{:.1f} $\\pm$ {:.1f}".format(values[1]*100, values[2]*100)

            # Check for significance
            if dataset in significantly_better_avg and \
                    method in significantly_better_avg[dataset] and \
                    significantly_better_avg[dataset][method]:
                indexed_by_target_avg[dataset][method] = \
                    "\\underline{" + indexed_by_target_avg[dataset][method] + "}"

    #
    # Create Latex table
    #
    columns = ["No Adaptation", "R-DANN", "VRADA", "CoDATS", "CoDATS-WS", "Train on Target"]

    # Create table
    table = []

    for i, key in enumerate(keys):
        # # The \hline's
        # if len(row) == 1:
        #     table.append([row[0]])
        #     continue

        # Upper bounds are included separately below
        if key[2] == "":
            continue

        dataset, sources, target = key
        adaptation = dataset + " " + sources + " $\\rightarrow$ " + target
        thisrow = [adaptation]

        for column in columns:
            if column == "Train on Target":
                # get upper bound when the source was this one's target
                val = indexed_by_target[(dataset, target, "")][column]
            else:
                if column in indexed_by_target[key]:
                    val = indexed_by_target[key][column]
                else:
                    # Not done yet (e.g. CyCADA gave errors at some point)
                    val = ""

            thisrow.append(val)

        table.append(thisrow)

        # Average after each dataset's rows, when last row or next row is
        # a different dataset
        if i == len(keys)-1 or dataset != keys[i+1][0]:
            table.append(["\\hdashline"])
            thisrow = [dataset + " Average"]

            for column in columns:
                if column in indexed_by_target_avg[dataset]:
                    val = indexed_by_target_avg[dataset][column]
                else:
                    # Not done yet (e.g. CyCADA gave errors at some point)
                    val = ""
                thisrow.append(val)
            table.append(thisrow)

            # Not if last line
            if i != len(keys)-1:
                table.append(["\\hline"])

    # Skip problem name and upper bound
    write_table(output_filename, table, replace_bold=(1, 5))


def table_singlesource(run_suffix, variant, variant_match=None,
        output="table.tex", source_feature_subset=None,
        target_feature_subset=None):
    """ Similar to make_plots(), but for the SS-DA table instead """
    results = get_results(run_suffix, variant, variant_match,
        source_feature_subset, target_feature_subset)
    output_latex_ss_results(results, output)


def output_latex_ms_results(results, output_filename):
    """ There's >350 values to fill in... I'm not going to manually type that
    in LaTex, especially when I'll have to do it more than once. This is not
    clean code per se. """
    averages = process_results(results, average_over_users=True, ssda=False)
    significantly_better = compute_significance(results, average=True, with_codats=True)

    # We're only looking at WISDM AR at the moment
    dataset = "WISDM AR"

    dataset_results = averages[dataset]
    indexed = {}
    keys = None

    for i, (method, values) in enumerate(dataset_results.items()):
        method_results = []
        method = nice_method_names[method]
        method_keys = []  # keys for this method

        for row in range(len(values)):
            n = values[row, 0]
            mean = values[row, 1]
            std = values[row, 2]

            # Create keys (n=1, n=2, ...) on first method, but not upper bound
            # since it only has one key (n=1)
            if method != "Train on Target":
                method_keys.append(int(n))

            val = "{:.1f} $\\pm$ {:.1f}".format(mean*100, std*100)

            # Check for significance
            if dataset in significantly_better and \
                    method in significantly_better[dataset] and \
                    significantly_better[dataset][method]:
                val = \
                    "\\underline{" + val + "}"

            method_results.append(val)

        assert not keys or keys == method_keys, \
            "n values must be the same for each method"
        keys = method_keys
        indexed[method] = method_results

    #
    # Create Latex table
    #
    columns = ["No Adaptation", "CoDATS", "CoDATS-WS"]
    fancy_columns = ["No Adaptation", "\\textit{CoDATS}", "\\textit{CoDATS-WS}"]

    # Create table
    table = []
    table.append(["\\toprule"])
    table.append(["Number of Sources"] + fancy_columns)
    table.append(["\\midrule"])

    for i, key in enumerate(keys):
        thisrow = ["$n = "+str(key)+"$"]

        for method in columns:
            val = indexed[method][i]
            thisrow.append(val)

        table.append(thisrow)

    table.append(["\\bottomrule"])

    # Print table, but bold the highest in each row exluding method name
    write_table(output_filename, table, replace_bold=(1, 3))


def table_multisource(run_suffix, variant, variant_match=None,
        output="table.tex", source_feature_subset=None,
        target_feature_subset=None):
    """ Similar to make_plots(), but for the MS-DA table instead """
    results = get_results(run_suffix, variant, variant_match,
        source_feature_subset, target_feature_subset)
    output_latex_ms_results(results, output)


def main(argv):
    outdir = "result_plots"
    for_paper = FLAGS.paper
    skip = []

    if for_paper:
        outdir += "_paper"
        show_title = False
        legend_separate = True
        ncol = 5
        suffix = "pdf"
        figsize = (5, 3)
        jitter_amount = 0.005
    else:
        show_title = True
        legend_separate = False
        ncol = 1
        suffix = "png"
        figsize = (30, 18)
        jitter_amount = 0.005

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # ACM doesn't like Type 3 fonts
    # https://tex.stackexchange.com/q/18687
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)

    # Multi-source plots
    #
    # We pass variant=best_target, but match * variant since for the upper bound
    # there isn't a "target" (since target is passed as source), but we all the
    # others we evaluate only with best_target, so we can match all to get the
    # best_source only for the upper bound.
    make_plots("msda1", "best_target", "*",
        show_title=show_title, legend_separate=legend_separate, ncol=ncol,
        suffix=suffix, skip=skip, figsize=figsize, dir_name=outdir,
        jitter_amount=jitter_amount)

    table_multisource("msda1", "best_target", "*", output="table_msda.tex")

    # Single-source table
    table_singlesource("ssda1", "best_target", "*", output="table.tex")


if __name__ == "__main__":
    app.run(main)
