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
from scipy import stats
from matplotlib.ticker import MaxNLocator

from pool import run_job_pool
from file_utils import get_config

FLAGS = flags.FLAGS

flags.DEFINE_integer("jobs", 0, "Number of jobs to use for processing files (0 == number of cores)")


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
    "daws": "DA-WS",

    # Multi-source domain adaptation
    #"dann": "MS-DA-DANN",
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

method_types = {
    "none": "none",
    "upper": "upper",

    "dann": "MS-DA",
    "dann_gs": "MS-DA",
    "dann_smooth": "MS-DA",

    "daws": "DA-WS",

    "dann_dg": "DG",
    "sleep_dg": "DG",
    "aflac_dg": "DG",
    "ciddg_dg": "DG",
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

    valid_header = "Log Dir;Dataset;Sources;Target;Method;Best Step;Accuracy at Step"
    traintest_header = "Log Dir;Dataset;Sources;Target;Method;Train A;Test A;Train B;Test B"
    averages_header = "Dataset;Avg;Std"

    # Config information
    log_dir = None
    dataset_name = None
    sources = None
    num_domains = None
    target = None
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
                    if values[5] == "None":
                        print("Warning: no best model for", filename, file=sys.stderr)
                        return None

                    validation.append((values[0], values[1], values[2],
                        values[3], values[4], int(values[5]), float(values[6])))

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

                    assert method is None or values[4] == method
                    method = values[4]
                elif in_traintest:
                    traintest.append((values[0], values[1], values[2],
                        values[3], values[4],
                        float(values[5]), float(values[6]),
                        float(values[7]), float(values[8])))
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
        #("watch_noother", "CASAS AR"),
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

    return make_replacements(dataset, replacements)


def get_results(results, average=False, method_average=False,
        target_amount=False, ssda=False):
    """ Get results - get the test on target mean and standard deviation values,
    indexed by,

    if average*=False: ms_results[dataset_name + " " + target][method][n]
    if average=True:  ms_results[dataset_name][method][n]
    if method_average=True: ms_results[...][method_type (MS-DA, DG, ...)][n]

    if ssda=True: ms_results[(dataset_name, source, target)][method]
    if ssda=True and average=True: ms_results[dataset_name][method]
    """
    ms_results = {}

    for result in results:
        params = result["parameters"]
        avgs = result["averages"]
        method = params["method"]
        assert params["config"] is not None, "no config for "+str(result)

        if target_amount:
            n = params["config"]["max_target_examples"]
        else:
            n = len(params["config"]["sources"])

        dataset_name = pretty_dataset_name(params["dataset"])

        # Not using our dataset in this paper, skip WISDM-AT for now -- does
        # very poorly without MS-DA
        #if "LABNAME" in dataset_name or "WISDM AT" in dataset_name:
        #    continue

        # For ssda, the index will be the raw tuple
        if ssda and not average:
            # Not that it really matters, but error anyway...
            assert "," not in params["sources"], \
                "ssda=True only should be used for SS-DA with one source"
            dataset_name = (dataset_name, params["sources"], params["target"])
        # Indexed by target, i.e. separate plot per dataset-target. Otherwise,
        # indexed by dataset, i.e. one plot per dataset (averaged over multiple
        # targets).
        elif not average:
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

        # Group by types, average over all methods of that type
        if method_average:
            method = method_types[method]

        #print(target, method, seqlen, mean, std, sep=";")

        if dataset_name not in ms_results:
            ms_results[dataset_name] = {}

        # For ssda=True, we don't need any "n" value
        if ssda:
            if method not in ms_results[dataset_name]:
                ms_results[dataset_name][method] = []
            ms_results[dataset_name][method].append((mean, std))
        else:
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


def average_over_method(ss_results):
    """ For SS-DA methods, average over the runs for each method, which
    typically will be one (standard deviation already computed from the 3 runs)
    or if averaging over all targets, then maybe 10 or so """
    for dataset, values in ss_results.items():
        for method, ss_values in values.items():
            if len(ss_values) > 1:
                ss_values = np.array(ss_values, dtype=np.float32)
                ss_results[dataset][method] = (ss_values[:, 0].mean(), ss_values[:, 0].std(ddof=0))
            else:
                ss_results[dataset][method] = ss_values[0]

    # Make numpy array
    for dataset, values in ss_results.items():
        for method, ss_values in values.items():
            ss_results[dataset][method] = \
                np.array(ss_values, dtype=np.float32)

    return ss_results


def generate_plots(ms_results, prefix, save_plot=True, show_title=False,
        legend_separate=True, suffix="pdf", dir_name="result_plots",
        error_bars=True, figsize=(5, 3), xlabel="Number of source domains",
        skip=[], yrange=None, ncol=1):
    # See: https://matplotlib.org/3.1.1/api/markers_api.html
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "D", "P", "X", "h",
        "1", "2", "3", "4", "+", "x"]

    # Sort datasets by name
    dataset_names = list(ms_results.keys())
    dataset_names.sort()

    for dataset_name in dataset_names:
        # Sort methods by name
        dataset_values = ms_results[dataset_name]
        methods = list(dataset_values.keys())
        methods.sort()

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
        jitter = gen_jitter(len(data), amount=0.01*x_range)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)

        if yrange is not None:
            ax.set_ylim(yrange)

        # Only integers on x axis
        # https://stackoverflow.com/a/38096332
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0] + jitter[i]
            y = method_data[:, 1]*100
            std = method_data[:, 2]*100

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

            if error_bars:
                p = plt.errorbar(x, y, yerr=std, label=method_name, fmt=markers[i]+line_type, alpha=0.8)
            else:
                p = plt.plot(x, y, markers[i]+line_type, label=method_name, alpha=0.8)

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

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Target Domain Accuracy (%)")

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

    ms_results = average_over_n(get_results(results))
    ms_averages = average_over_n(get_results(results, average=True))
    ms_method_averages = average_over_n(get_results(results, average=True, method_average=True))

    generate_plots(ms_results, "multisource_"+variant, save_plot, show_title,
        legend_separate, suffix, ncol=4)
    generate_plots(ms_averages, "multisource_average_"+variant, save_plot,
        show_title, legend_separate, suffix, ncol=4)
    generate_plots(ms_method_averages, "multisource_methodaverage_"+variant, save_plot,
        show_title, legend_separate, suffix, error_bars=False,
        yrange=[35, 105], ncol=2)


def plot_varyamount(dataset, variant, variant_match=None, save_plot=True,
        show_title=False, legend_separate=True, suffix="pdf"):
    if variant_match is None:
        variant_match = variant

    files = get_tuning_files("results", prefix="results_"+dataset+"_"+variant_match+"-")
    results = all_stats(files)

    ms_results = average_over_n(get_results(results, target_amount=True))
    ms_averages = average_over_n(get_results(results, average=True,
        target_amount=True))
    ms_method_averages = average_over_n(get_results(results, average=True,
        method_average=True, target_amount=True))

    xlabel = "Number of unlabeled target instances for training"

    generate_plots(ms_results, "varyamount_"+variant, save_plot, show_title,
        legend_separate, suffix, xlabel=xlabel, ncol=2)
    generate_plots(ms_averages, "varyamount_average_"+variant, save_plot,
        show_title, legend_separate, suffix, xlabel=xlabel, ncol=2)
    generate_plots(ms_method_averages, "varyamount_methodaverage_"+variant, save_plot,
        show_title, legend_separate, suffix, error_bars=True,
        xlabel=xlabel, skip=["none", "upper"],
        yrange=[35, 105], ncol=2)


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
    # Indexed by target_accuracy[dataset][method]
    datasets = {}

    for result in results:
        params = result["parameters"]
        traintest = result["traintest"]
        method = params["method"]
        target_accuracy = traintest["Test B"].values

        # Skip
        if method == "upper" or method == "lower" or method == "random":
            continue

        dataset_name = pretty_dataset_name(params["dataset"])

        if average:
            dataset = dataset_name
        else:
            dataset = (dataset_name, params["sources"], params["target"])

        method = nice_method_names[method]

        if dataset not in datasets:
            datasets[dataset] = {}
        if method not in datasets[dataset]:
            datasets[dataset][method] = []

        datasets[dataset][method] = target_accuracy

    significantly_better = {}

    for dataset, values in datasets.items():
        codats = None
        daws = None

        if with_codats:
            if "CoDATS" in values:
                if "DA-WS" in values:
                    daws = \
                        stats.ttest_rel(values["CoDATS"], values["DA-WS"]).pvalue < significance_level
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

                if "DA-WS" in values:
                    daws = \
                        stats.ttest_rel(values["R-DANN"], values["DA-WS"]).pvalue < significance_level and \
                        stats.ttest_rel(values["VRADA"], values["DA-WS"]).pvalue < significance_level
                else:
                    print("Warning: no DA-WS so no significance")
            else:
                print("Warning: no R-DANN/VRADA so no significance", file=sys.stderr)

        significantly_better[dataset] = {
            "CoDATS": codats,
            "DA-WS": daws,
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
    datasets = average_over_method(get_results(results, ssda=True))  # index tuple: (dataset, source, target)
    averaged = average_over_method(get_results(results, ssda=True, average=True))  # index tuple: dataset

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

            indexed_by_target[key][method] = "{:.1f} $\\pm$ {:.1f}".format(values[0]*100, values[1]*100)

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
            indexed_by_target_avg[dataset][method] = "{:.1f} $\\pm$ {:.1f}".format(values[0]*100, values[1]*100)

            # Check for significance
            if dataset in significantly_better_avg and \
                    method in significantly_better_avg[dataset] and \
                    significantly_better_avg[dataset][method]:
                indexed_by_target_avg[dataset][method] = \
                    "\\underline{" + indexed_by_target_avg[dataset][method] + "}"

    #
    # Create Latex table
    #
    columns = ["No Adaptation", "R-DANN", "VRADA", "CoDATS", "DA-WS", "Train on Target"]

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


def table_singlesource(dataset, variant, variant_match=None, output="table.tex"):
    files = get_tuning_files("results", prefix="results_"+dataset+"_"+variant_match+"-")
    results = all_stats(files, sort=True)
    output_latex_ss_results(results, output)


def output_latex_ms_results(results, output_filename):
    """ There's >350 values to fill in... I'm not going to manually type that
    in LaTex, especially when I'll have to do it more than once. This is not
    clean code per se. """
    ms_averages = average_over_n(get_results(results, average=True))
    significantly_better = compute_significance(results, average=True, with_codats=True)

    # We're only looking at WISDM AR at the moment
    dataset = "WISDM AR"

    dataset_results = ms_averages[dataset]
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
    columns = ["No Adaptation", "CoDATS", "DA-WS"]
    fancy_columns = ["No Adaptation", "\\textit{CoDATS}", "\\textit{DA-WS}"]

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


def table_multisource(dataset, variant, variant_match=None, output="table.tex"):
    files = get_tuning_files("results", prefix="results_"+dataset+"_"+variant_match+"-")
    results = all_stats(files, sort=True)
    output_latex_ms_results(results, output)


def main(argv):
    outdir = "result_plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Multi-source plots
    #
    # We pass variant=best_target, but match * variant since for the upper bound
    # there isn't a "target" (since target is passed as source), but we all the
    # others we evaluate only with best_target, so we can match all to get the
    # best_source only for the upper bound.
    plot_multisource("msda1", "best_target", "*",
        save_plot=True, show_title=False,
        legend_separate=True, suffix="pdf")

    table_multisource("msda1", "best_target", "*", output="table_msda.tex")

    # Single-source table
    table_singlesource("ssda1", "best_target", "*", output="table.tex")


if __name__ == "__main__":
    app.run(main)
