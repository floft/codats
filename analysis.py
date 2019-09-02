#!/usr/bin/env python3
"""
Analyze the results
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from scipy import stats


FLAGS = flags.FLAGS

flags.DEFINE_string("ignore", "", "List of models to ignore, comma separated")


# Use nice names for the plot
nice_method_names = {
    "none": "Lower Bound",  # (no adaptation)
    "upper": "Upper Bound",  # (train on target)
    "dann": "CoDATS + DANN-Shu",
    "dann_grl": "CoDATS + DANN-GRL",
    "cyclegan": "CoDATS + CycleGAN",
    "cyclegan_dann": "CoDATS + CycleGAN + DANN",
    "cycada": "CoDATS + CyCADA",
    "deepjdot": "CoDATS + DeepJDOT",
    "rdann": "R-DANN",
    "vrada": "VRADA",
    "random": "Many Reinit",
}


def get_tuning_files(dir_name, prefix="results_runwalk01_"):
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

    valid_header = "Log Dir,Source,Target,Model,Method,Best Step,Accuracy at Step"
    traintest_header = "Log Dir,Source,Target,Model,Method,Train A,Test A,Train B,Test B,Target Train A,Target Test A,Target Train B,Target Test B"
    averages_header = "Dataset,Avg,Std"

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if line == "Virtual devices must be set at program startup":
                pass
            elif line == "Error occured -- exiting":
                print("Found:", line, "in", filename, file=sys.stderr)
                exit(1)
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
                values = line.split(",")

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
                elif in_traintest:
                    traintest.append((values[0], values[1], values[2],
                        values[3], values[4], float(values[5]),
                        float(values[6]), float(values[7]), float(values[8]),
                        float(values[9]), float(values[10]), float(values[11]),
                        float(values[12])))
                elif in_averages:
                    averages.append((values[0], float(values[1]), float(values[2])))
            else:
                # Empty lines ends a section
                in_validation = False
                in_traintest = False
                in_averages = False

    validation = pd.DataFrame(data=validation, columns=valid_header.split(","))
    traintest = pd.DataFrame(data=traintest, columns=traintest_header.split(","))
    averages = pd.DataFrame(data=averages, columns=averages_header.split(","))

    return validation, traintest, averages


def compute_mean_std(df, name, ignore_label_flipping, filename):
    data = df[name]

    if ignore_label_flipping:
        # To ignore label flipping (assuming this is binary classification), we
        # flip anything less than 50% to over 50%
        data[data < 0.5] = 1 - data[data < 0.5]

    # I expect there to only be 3 or 5 of each... if not, warn
    length = len(data)

    if length != 3 and length != 5:
        print("Warning: number of runs ", length, "(not 3 or 5) for", filename,
            file=sys.stderr)

    # ddof=0 is the numpy default, ddof=1 is Pandas' default
    return data.mean(), data.std(ddof=0)


def compute_val_stats(df, ignore_label_flipping, filename):
    return compute_mean_std(df, "Accuracy at Step", ignore_label_flipping, filename)


def compute_eval_stats(df, filename, has_target_clasifier=False, ignore_label_flipping=False):
    names = ["Train A", "Test A", "Train B", "Test B"]
    if has_target_clasifier:
        names += ["Target Train A", "Target Test A", "Target Train B", "Target Test B"]
    data = [[name]+list(compute_mean_std(df, name, ignore_label_flipping, filename)) for name in names]
    return pd.DataFrame(data=data, columns=["Dataset", "Avg", "Std"])


def parse_name_synthetic_old(name):
    # Get values
    values = name.split("-")

    dataset = values[0]
    method = values[1]
    adaptation = int(values[2].replace("b", ""))

    return {
        "dataset": dataset,
        "method": method,
        "adaptation": adaptation,
    }


def parse_name_synthetic(name):
    # Get values
    values = name.split("-")

    method = values[0]
    dataset = values[1]
    adaptation = int(values[2].split("_")[-1].replace("b", ""))

    return {
        "dataset": dataset,
        "method": method,
        "adaptation": adaptation,
    }


def parse_name_real(name):
    # Get values
    values = name.split("-")

    method = values[0]
    source = values[1]
    target = values[2]

    return {
        "method": method,
        "source": source,
        "target": target,
    }


def all_stats(files, recompute_averages=True, sort_on_test=False,
        sort_on_b=False, sort_by_name=False, has_target_clasifier=False,
        ignore_label_flipping=False, real_data=False):
    stats = []

    for name, file in files:
        parse_result = parse_file(file)

        if parse_result is None:
            print("Warning: no data, skipping", file, file=sys.stderr)
            continue

        validation, traintest, averages = parse_result

        if recompute_averages:
            averages = compute_eval_stats(traintest, name,
                has_target_clasifier, ignore_label_flipping)

        validavg = compute_val_stats(validation, ignore_label_flipping, name)

        if real_data:
            params = parse_name_real(name)
        else:
            params = parse_name_synthetic(name)

        stats.append({
            "name": name,
            "parameters": params,
            "file": file,
            "validation": validation,
            "traintest": traintest,
            "averages": averages,
            "validavg": validavg,
        })

    if sort_by_name:
        stats.sort(key=lambda x: x["name"])
    elif sort_on_test:
        # Sort by test accuracy (i.e. cheating)
        stats.sort(key=lambda x: x["averages"][x["averages"]["Dataset"] == "Test A"]["Avg"].values[0])
    elif sort_on_b:
        # Sort by test accuracy on domain B (i.e. cheating)
        stats.sort(key=lambda x: x["averages"][x["averages"]["Dataset"] == "Test B"]["Avg"].values[0])
    else:
        # Sort by validation accuracy
        stats.sort(key=lambda x: x["validavg"][0])

    return stats


def output_csv(results):
    """ Output most relevant part of the evaluation results to CSV file """
    print("Dataset,Method,Adaptation,Train A,Test A, Train B, Test B")
    for result in results:
        params = result["parameters"]
        avgs = result["averages"]
        print(params["dataset"], params["method"], params["adaptation"],
            avgs[avgs["Dataset"] == "Train A"]["Avg"].values[0],
            avgs[avgs["Dataset"] == "Test A"]["Avg"].values[0],
            avgs[avgs["Dataset"] == "Train B"]["Avg"].values[0],
            avgs[avgs["Dataset"] == "Test B"]["Avg"].values[0], sep=",")


def gen_jitter(length, amount=0.04):
    """ "Dodge" the points slightly on the x axis, so that they don't overlap """
    x = []
    value = -(amount/length)/2

    for i in range(length):
        x.append(value)
        value += amount

    return np.array(x, dtype=np.float32)


def compute_significance(results, significance_level=0.05):
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

        if method in FLAGS.ignore.split(","):
            print("Skipping", method, file=sys.stderr)
            continue

        dataset = params["source"] + " --> " + params["target"]
        method = nice_method_names[method]

        if dataset not in datasets:
            datasets[dataset] = {}
        if method not in datasets[dataset]:
            datasets[dataset][method] = []

        datasets[dataset][method] = target_accuracy

    significantly_better = {}

    for dataset, values in datasets.items():
        dannshu = None
        danngrl = None
        cycada = None
        deepjdot = None

        if "R-DANN" in values and "VRADA" in values:
            if "CoDATS + DANN-Shu" in values:
                dannshu = \
                    stats.ttest_rel(values["R-DANN"], values["CoDATS + DANN-Shu"]).pvalue < significance_level and \
                    stats.ttest_rel(values["VRADA"], values["CoDATS + DANN-Shu"]).pvalue < significance_level

            if "CoDATS + DANN-GRL" in values:
                danngrl = \
                    stats.ttest_rel(values["R-DANN"], values["CoDATS + DANN-GRL"]).pvalue < significance_level and \
                    stats.ttest_rel(values["VRADA"], values["CoDATS + DANN-GRL"]).pvalue < significance_level

            if "CoDATS + CyCADA" in values:
                cycada = \
                    stats.ttest_rel(values["R-DANN"], values["CoDATS + CyCADA"]).pvalue < significance_level and \
                    stats.ttest_rel(values["VRADA"], values["CoDATS + CyCADA"]).pvalue < significance_level

            if "CoDATS + DeepJDOT" in values:
                deepjdot = \
                    stats.ttest_rel(values["R-DANN"], values["CoDATS + DeepJDOT"]).pvalue < significance_level and \
                    stats.ttest_rel(values["VRADA"], values["CoDATS + DeepJDOT"]).pvalue < significance_level

        else:
            print("Warning: no R-DANN/VRADA so no significance", file=sys.stderr)

        print(dataset, "- DANN-Shu", dannshu, "DANN-GRL", danngrl, "CyCADA", cycada, "DeepJDOT", deepjdot)

        significantly_better[dataset] = {
            "CoDATS + DANN-Shu": dannshu,
            "CoDATS + DANN-GRL": danngrl,
            "CoDATS + CyCADA": cycada,
            "CoDATS + DeepJDOT": deepjdot,
        }

    return significantly_better


def process_results(results, real_data=False):
    datasets = {}

    for result in results:
        params = result["parameters"]
        avgs = result["averages"]
        method = params["method"]
        source_accuracy = avgs[avgs["Dataset"] == "Test A"]["Avg"].values[0]
        source_accuracy_std = avgs[avgs["Dataset"] == "Test A"]["Std"].values[0]
        target_accuracy = avgs[avgs["Dataset"] == "Test B"]["Avg"].values[0]
        target_accuracy_std = avgs[avgs["Dataset"] == "Test B"]["Std"].values[0]

        # For upper bound, we set the source to the target
        if method == "upper":
            target_accuracy = source_accuracy
            target_accuracy_std = source_accuracy_std

        if real_data:
            if method == "upper":
                dataset = params["source"]
            else:
                dataset = params["source"] + " --> " + params["target"]
        else:
            dataset = params["dataset"]
            adaptation = params["adaptation"]

        # Skip if desired
        if method in FLAGS.ignore.split(","):
            print("Skipping", method, file=sys.stderr)
            continue

        method = nice_method_names[method]

        if dataset not in datasets:
            datasets[dataset] = {}
        if method not in datasets[dataset]:
            datasets[dataset][method] = []

        if real_data:
            # Not .append() since there's only one of each method/dataset on
            # the real data
            datasets[dataset][method] = [target_accuracy, target_accuracy_std]
        else:
            # (x,y,error) - x axis is adaptation problem {0..5} and y axes is accuracy
            datasets[dataset][method].append([adaptation, target_accuracy,
                target_accuracy_std])
            # Sort on adaptation problem, so we get 0, 1, 2, 3, 4, 5
            datasets[dataset][method].sort(key=lambda x: x[0])

    return datasets


def export_legend(legend, filename="key.pdf", expand=[-5, -5, 5, 5]):
    """ See: https://stackoverflow.com/a/47749903 """
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def plot_synthetic_results(results, save_plot=False, save_prefix="plot_",
        title_suffix="", show_title=False, legend_separate=True, suffix="pdf"):
    """ Generate a plot for each dataset comparing how well the various
    adaptation methods handle varying amounts of domain shift """
    datasets = process_results(results)

    # See: https://matplotlib.org/3.1.1/api/markers_api.html
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "D", "P", "X", "h",
        "1", "2", "3", "4", "+", "x"]

    for dataset_name, dataset_values in datasets.items():
        methods = list(dataset_values.keys())
        data = list(dataset_values.values())
        jitter = gen_jitter(len(data))  # "dodge" points so they don't overlap

        fig, ax = plt.subplots(1, 1, figsize=(10, 4.1), dpi=100)

        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0] + jitter[i]
            y = method_data[:, 1]*100
            std = method_data[:, 2]*100
            plt.errorbar(x, y, yerr=std, label=methods[i], fmt=markers[i]+"--", alpha=0.8)

        if show_title:
            plt.title("Various Adaptation Methods on Dataset "+dataset_name+title_suffix)

        ax.set_xlabel("Domain Shift Amount")
        ax.set_ylabel("Target Domain Accuracy (%)")

        if legend_separate:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            export_legend(legend)
            legend.remove()
        else:
            # Put legend outside the graph http://stackoverflow.com/a/4701285
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if save_plot:
            plt.savefig(save_prefix+dataset_name+"."+suffix, bbox_inches='tight')

    if save_plot:
        plt.close()
    else:
        plt.show()


def to_list(datasets):
    """ Convert two-level dictionary to list """
    output = []
    for dataset, data in datasets.items():
        for method, values in data.items():
            output.append((dataset, method, values[0]*100, values[1]*100))
    return output


def print_real_results(results, title=None, filename=None):
    """ Print table comparing different methods on real datasets """
    # Don't truncate
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", -1)
    pd.set_option("display.width", None)

    datasets = process_results(results, real_data=True)
    df = pd.DataFrame(to_list(datasets),
        columns=["Adaptation", "Method", "Mean", "Std"])

    if title is not None:
        print(title)
    print(df)

    if filename is not None:
        df.to_csv(filename)


def make_replacements(s, replacements):
    """ Make a bunch of replacements in a string """
    if s is None:
        return s

    for before, after in replacements:
        s = s.replace(before, after)

    return s


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


def pretty_source_target_names(source, target):
    replacements = [
        ("ucihar_", "HAR "),
        ("uwave_", "uWave "),
        ("utdata_wrist", "Wrist"),
        ("utdata_pocket", "Pocket"),
    ]

    source = make_replacements(source, replacements)
    target = make_replacements(target, replacements)

    return source, target


def print_latex_results(results):
    """ There's >350 values to fill in... I'm not going to manually type that
    in LaTex, especially when I'll have to do it more than once. This is not
    clean code per se. """
    significantly_better = compute_significance(results)
    datasets = process_results(results, real_data=True)
    indexed_by_target = {}

    for dataset, data in datasets.items():
        values = dataset.split(" --> ")

        if len(values) == 2:
            source, target = values
        elif len(values) == 1:
            source = None
            target = values[0]

        source, target = pretty_source_target_names(source, target)
        adaptation = (source, target)
        indexed_by_target[adaptation] = {}

        for method, values in data.items():
            indexed_by_target[adaptation][method] = "{:.1f} $\\pm$ {:.1f}".format(values[0]*100, values[1]*100)

            # Check for significance
            if dataset in significantly_better and \
                    method in significantly_better[dataset] and \
                    significantly_better[dataset][method]:
                indexed_by_target[adaptation][method] = \
                    "\\underline{" + indexed_by_target[adaptation][method] + "}"

    columns = ["Lower Bound", "Many Reinit", "R-DANN", "VRADA", "CoDATS + DANN-GRL",
        "CoDATS + DANN-Shu", "CoDATS + CyCADA", "CoDATS + DeepJDOT", "Upper Bound"]
    rows = [
        ("HAR 1", "HAR 2"), ("HAR 2", "HAR 1"),
        ("HAR 3", "HAR 4"), ("HAR 4", "HAR 3"),
        ("HAR 5", "HAR 6"), ("HAR 6", "HAR 5"),
        ("HAR 7", "HAR 8"), ("HAR 8", "HAR 7"),
        ("HAR 9", "HAR 10"), ("HAR 10", "HAR 9"),
        ("HAR 11", "HAR 12"), ("HAR 12", "HAR 11"),
        ("\\hline",),
        ("uWave 1", "uWave 2"), ("uWave 2", "uWave 1"),
        ("uWave 3", "uWave 4"), ("uWave 4", "uWave 3"),
        ("uWave 5", "uWave 6"), ("uWave 6", "uWave 5"),
        ("uWave 7", "uWave 8"), ("uWave 8", "uWave 7"),
        ("\\hline",),
        ("Wrist", "Pocket"), ("Pocket", "Wrist"),
    ]

    # Create table
    table = []

    for row in rows:
        # The \hline's
        if len(row) == 1:
            table.append([row[0]])
            continue

        thisrow = [row[0], row[1]]

        for column in columns:
            if column == "Upper Bound":
                val = indexed_by_target[(None, row[1])][column]
            else:
                if column in indexed_by_target[row]:
                    val = indexed_by_target[row][column]
                else:
                    # Not done yet (e.g. CyCADA gave errors at some point)
                    val = ""

            thisrow.append(val)

        table.append(thisrow)

    # Print table, but bold the highest in each row excluding the last
    for row in table:
        row[2:9+1] = replace_highest_bold(row[2:9+1])

        # \hline's
        if len(row) == 1:
            print(row[0])
            continue

        for i, column in enumerate(row):
            print(column, end=" ")

            if i == len(row)-1:
                print("\\\\")
            else:
                print("&", end=" ")


def plot_seqlen(datasets, variant, save_plot=True, show_title=False,
        legend_separate=True, save_prefix="seqlen_", suffix="pdf"):
    """ Similar to plot_synthetic_results but for varying sequence lengths
    on the real datasets """
    seqlen_results = {}
    dataset_filenames = {}

    # Get all the results for sequence lengths 10, 20, 30, ..., 100
    for dataset in datasets:
        files = get_tuning_files(".", prefix="results_"+dataset+"_"+variant+"-")
        results = all_stats(files, sort_by_name=True, real_data=True)

        seqlen = int(dataset.replace("seqlen", ""))

        for result in results:
            params = result["parameters"]
            avgs = result["averages"]
            method = params["method"]
            source, target = pretty_source_target_names(params["source"], params["target"])
            dataset_name = source + " --> " + target
            mean = avgs[avgs["Dataset"] == "Test B"]["Avg"].values[0]
            std = avgs[avgs["Dataset"] == "Test B"]["Std"].values[0]

            # Filenames shouldn't have --> in them for Latex
            dataset_filenames[dataset_name] = params["source"]

            # For upper bound, we set the source to the target
            if method == "upper":
                mean = avgs[avgs["Dataset"] == "Test A"]["Avg"].values[0]
                std = avgs[avgs["Dataset"] == "Test A"]["Std"].values[0]

            #print(dataset_name, method, seqlen, mean, std, sep=",")

            if dataset_name not in seqlen_results:
                seqlen_results[dataset_name] = {}
            if method not in seqlen_results[dataset_name]:
                seqlen_results[dataset_name][method] = []
            seqlen_results[dataset_name][method].append((seqlen, mean, std))

    # Make numpy array
    for dataset, values in seqlen_results.items():
        for method, seqlen_values in values.items():
            seqlen_results[dataset][method] = \
                np.array(seqlen_results[dataset][method], dtype=np.float32)

    # See: https://matplotlib.org/3.1.1/api/markers_api.html
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "D", "P", "X", "h",
        "1", "2", "3", "4", "+", "x"]

    for dataset_name, dataset_values in seqlen_results.items():
        methods = list(dataset_values.keys())
        data = list(dataset_values.values())
        jitter = gen_jitter(len(data))  # "dodge" points so they don't overlap

        fig, ax = plt.subplots(1, 1, figsize=(10, 4.1), dpi=100)

        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0] + jitter[i]
            y = method_data[:, 1]*100
            std = method_data[:, 2]*100
            method_name = nice_method_names[methods[i]]
            plt.errorbar(x, y, yerr=std, label=method_name, fmt=markers[i]+"--", alpha=0.8)

        if show_title:
            plt.title("Adaptation Methods with Varying Sequence Lengths on "+dataset_name)

        ax.set_xlabel("Trimmed Sequence Length")
        ax.set_ylabel("Target Domain Accuracy (%)")

        if legend_separate:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            export_legend(legend, filename=save_prefix+"key."+suffix)
            legend.remove()
        else:
            # Put legend outside the graph http://stackoverflow.com/a/4701285
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if save_plot:
            dataset_filename = dataset_filenames[dataset_name]
            plt.savefig(save_prefix+dataset_filename+"."+suffix, bbox_inches='tight')

    if save_plot:
        plt.close()
    else:
        plt.show()


def main(argv):
    # Ignoring label flipping won't work on best model since if it flips the
    # labels, it'll pick (and actually save during training) the wrong "best"
    # model
    #files = get_tuning_files(".", prefix="results_runwalk01_best-")
    #results = all_stats(files, sort_by_name=True)
    #plot_synthetic_results(results, save_plot=True, save_prefix="plot_runwalk01_best_", title_suffix=" (best)")

    #variants = ["best", "last"]
    variants = ["best"]

    #
    # Synthetic data
    #
    datasets = [
        # "runwalk01",
        # "runwalk2",
        # "runwalk3",
        # "runwalk4",
        # "losses-grl-bi",
        # "losses-gan-bi",
        # "losses-lsgan-bi",
        # "losses-wgan-bi",
        # "losses-gan-nobi",
        # "losses-lsgan-nobi",
        # "losses-wgan-nobi",
        # "runwalk5",
        # "runwalk6",
        # "rand1",
        # "rand2",
        # "runwalk7",
        # "rotate1",
        # "comb1",
        # "synthetic1",
    ]

    for dataset in datasets:
        for variant in variants:
            files = get_tuning_files(".", prefix="results_"+dataset+"_"+variant+"-")
            results = all_stats(files, sort_by_name=True)
            plot_synthetic_results(results, save_plot=True,
                save_prefix="plot_"+dataset+"_"+variant+"_",
                title_suffix=" ("+variant+")")

    #
    # Real data
    #
    datasets = [
        # "real_utdata1",
        # "real_utdata_rand1",
        # "realdata1",
        # "ucihar1",
        # "ucihar2",
        # "uwave2",
        # "real1",
        # "real2",
    ]

    for dataset in datasets:
        print("Dataset:", dataset)

        for variant in variants:
            files = get_tuning_files(".", prefix="results_"+dataset+"_"+variant+"-")
            results = all_stats(files, sort_by_name=True, real_data=True)
            #print_real_results(results, title="Real Dataset Adaptation ("+variant+")",
            #    filename="analysis_"+dataset+"_"+variant+".csv")
            # Just make the .tex code
            print_latex_results(results)
            print()

    #
    # Varying sequence lengths
    #
    variant = "best"
    datasets = ["seqlen"+str(i) for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
    plot_seqlen(datasets, variant)


if __name__ == "__main__":
    app.run(main)
