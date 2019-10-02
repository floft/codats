#!/usr/bin/env python3
"""
Analyze the results
"""
import re
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
flags.DEFINE_boolean("similarity", False, "Similarity on x axis rather than number of source domains")
flags.DEFINE_string("similarity_filename", "similarity.txt", "File containing similarity")


# Use nice names for the plot
nice_method_names = {
    # No adaptation or training on target
    "none": "None",  # (no adaptation)
    "upper": "Target Only",  # (train on target)

    # Multi-source domain adaptation
    "dann_grl": "MS-DA-DANN",
    "dann_grl_gs": "GS-DA",
    "dann_smooth": "MS-DA-Smooth",

    # Domain generalization
    "dann_grl_dg": "DG-DANN",
    "sleep_dg": "DG-Sleep",
    "aflac_dg": "DG-AFLAC",
    "ciddg_dg": "DG-CIDDG",

    #"none": "Lower Bound",  # (no adaptation)
    #"upper": "Upper Bound",  # (train on target)
    "dann_shu": "CoDATS + DANN-Shu",
    "cyclegan": "CoDATS + CycleGAN",
    "cyclegan_dann": "CoDATS + CycleGAN + DANN",
    "cycada": "CoDATS + CyCADA",
    "deepjdot": "CoDATS + DeepJDOT",
    "rdann": "R-DANN",
    "vrada": "VRADA",
    "random": "Many Reinit",
}


def get_similarity():
    """ Returns dictionary indexed by target then source and lists the
    train mean +/- stdev then test mean +/- stdev similarity """
    similarity = {}

    with open(FLAGS.similarity_filename) as f:
        for line in f:
            parts = line.split(";")
            assert len(parts) == 10
            _, _, source, target, _, _, \
                distance_train_mean, distance_train_std, \
                distance_test_mean, distance_test_std = parts

            if target not in similarity:
                similarity[target] = {}

            similarity[target][source] = (
                distance_train_mean, distance_train_std,
                distance_test_mean, distance_test_std
            )

    return similarity


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


def smart_split(line):
    """ If the source/target names have commas in them, we can't simply split
    on commas, but we do know the format of the filenames so can match them
    separately

    TODO get rid of this and just use ; as the separator in main_eval.py
    """
    m = re.search(r"^([^/]+/[^-]*-[^-]*-[^-]*-[^-]*-[^-,]*),(.*)$", line)

    # If none, no match, so just split on commas like normal
    if m is None:
        return line.split(",")
    else:
        filename = m.group(1)
        rest = m.group(2)

        # Remove the source from the rest since it may have commas in it
        m = re.search(r"^[^/]+/([^-]*)-[^-]*-[^-]*-[^-]*-[^-,]*$", filename)
        assert m is not None, "couldn't find source in: " + filename
        source = m.group(1)

        # Get rid of source name and ignore it when splitting
        rest = rest.replace(source, "source").split(",")[1:]

        return [filename, source] + rest


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
                values = smart_split(line)

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


def compute_eval_stats(df, filename, has_target_classifier=False):
    names = ["Train A", "Test A", "Train B", "Test B"]
    if has_target_classifier:
        names += ["Target Train A", "Target Test A", "Target Train B", "Target Test B"]
    data = [[name]+list(compute_mean_std(df, name, filename)) for name in names]
    return pd.DataFrame(data=data, columns=["Dataset", "Avg", "Std"])


def parse_name_real(name):
    # Get values
    values = name.split("-")

    method = values[0]
    source = values[1]
    target = values[2]

    # number of source domains, 0 for upper bound with only target
    if source == "":
        num_source = 0
    else:
        # Find the n# in the source domain name
        m = re.search(r"n([0-9]+)", source)
        assert m is not None, "could not find n# in "+source
        num_source = m.group(1)

    return {
        "method": method,
        "source": source,
        "target": target,
        "n": num_source,
    }


def all_stats(files, recompute_averages=True, has_target_classifier=False):
    results = []

    for name, file in files:
        parse_result = parse_file(file)

        if parse_result is None:
            print("Warning: no data, skipping", file, file=sys.stderr)
            continue

        validation, traintest, averages = parse_result

        if recompute_averages:
            averages = compute_eval_stats(traintest, name, has_target_classifier)

        validavg = compute_val_stats(validation, name)
        params = parse_name_real(name)

        results.append({
            "name": name,
            "parameters": params,
            "file": file,
            "validation": validation,
            "traintest": traintest,
            "averages": averages,
            "validavg": validavg,
        })

    # Sort by name
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


def export_legend(legend, filename="key.pdf", expand=[-5, -5, 5, 5]):
    """ See: https://stackoverflow.com/a/47749903 """
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def make_replacements(s, replacements):
    """ Make a bunch of replacements in a string """
    if s is None:
        return s

    for before, after in replacements:
        s = s.replace(before, after)

    return s


def pretty_source_target_names(source, target):
    # Remove the list of which source domains it used exactly. We only care
    # about the number.
    m = re.search(r"^(.*_n[0-9]+)_.*$", source)
    if m is not None:
        source = m.group(1)

    # Make it look good
    replacements = [
        ("ucihar_", "HAR "),
        ("ucihhar_", "HHAR "),
        ("ucihm_", "HM "),
        ("uwave_", "uWave "),
        ("utdata_wrist", "Wrist"),
        ("utdata_pocket", "Pocket"),
    ]

    source = make_replacements(source, replacements)
    target = make_replacements(target, replacements)

    return source, target


def plot_multisource(dataset, variant="best", save_plot=True, show_title=False,
        legend_separate=True, suffix="pdf"):
    """ Generate plots of target accuracy vs. number of source domains """
    ms_results = {}

    files = get_tuning_files(".", prefix="results_"+dataset+"_"+variant+"-")
    results = all_stats(files)

    if FLAGS.similarity:
        prefix = "similarity"
        similarity = get_similarity()
    else:
        prefix = "multisource"
        similarity = None

    for result in results:
        params = result["parameters"]
        avgs = result["averages"]
        method = params["method"]
        n = params["n"]
        source, target = pretty_source_target_names(params["source"], params["target"])
        #dataset_name = source + " --> " + target
        dataset_name = target
        mean = avgs[avgs["Dataset"] == "Test B"]["Avg"].values[0]
        std = avgs[avgs["Dataset"] == "Test B"]["Std"].values[0]

        if FLAGS.similarity:
            # The raw values, not pretty versions
            if params["target"] in similarity and \
                    params["source"] in similarity[params["target"]]:
                result_similarity = similarity[params["target"]][params["source"]]

                # Use the mean training value
                result_similarity = result_similarity[0]
            else:
                print("Warning: skipping", params["source"], "to",
                    params["target"], "if not target, then probably update",
                    FLAGS.similarity_filename)
                continue
        else:
            result_similarity = 0.0

        # For upper bound, we set the source to the target
        if method == "upper":
            mean = avgs[avgs["Dataset"] == "Test A"]["Avg"].values[0]
            std = avgs[avgs["Dataset"] == "Test A"]["Std"].values[0]

        #print(dataset_name, method, seqlen, mean, std, sep=",")

        if dataset_name not in ms_results:
            ms_results[dataset_name] = {}
        if method not in ms_results[dataset_name]:
            ms_results[dataset_name][method] = {}
        if n not in ms_results[dataset_name][method]:
            ms_results[dataset_name][method][n] = []
        ms_results[dataset_name][method][n].append((n, result_similarity, mean, std))

    # Recompute mean/stdev for those that have multiple entries
    # Get rid of the n-specific dictionary
    for dataset, values in ms_results.items():
        for method, n_values in values.items():
            new_values = []

            for n, ms_values in n_values.items():
                if len(ms_values) > 1:
                    ms_values = np.array(ms_values, dtype=np.float32)
                    # All the 0th elements should be the same n
                    # Then recompute the mean/stdev from the accuracy values in 1th column
                    new_values.append((int(ms_values[0, 0]), float(ms_values[0, 1]), ms_values[:, 2].mean(), ms_values[:, 2].std(ddof=0)))
                elif len(ms_values) == 1:
                    # Leave as is if there's only one
                    #assert new_values == [], "upper bound has multiple runs?"
                    ms_values = np.array(ms_values, dtype=np.float32)
                    new_values.append((int(ms_values[0, 0]), float(ms_values[0, 1]), ms_values[0, 2], ms_values[0, 3]))
                else:
                    raise NotImplementedError("must be several or one run")

            # Sort on n or similarity
            if FLAGS.similarity:
                new_values.sort(key=lambda x: x[1])
            else:
                new_values.sort(key=lambda x: x[0])

            ms_results[dataset][method] = new_values

    # Make numpy array
    for dataset, values in ms_results.items():
        for method, ms_values in values.items():
            ms_results[dataset][method] = \
                np.array(ms_values, dtype=np.float32)

    # See: https://matplotlib.org/3.1.1/api/markers_api.html
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "D", "P", "X", "h",
        "1", "2", "3", "4", "+", "x"]

    for dataset_name, dataset_values in ms_results.items():
        methods = list(dataset_values.keys())
        data = list(dataset_values.values())
        jitter = gen_jitter(len(data))  # "dodge" points so they don't overlap

        fig, ax = plt.subplots(1, 1, figsize=(10, 4.1), dpi=100)

        for i in range(len(data)):
            method_data = np.array(data[i])

            if FLAGS.similarity:
                x = method_data[:, 1]  # don't jitter since x means something
            else:
                x = method_data[:, 0] + jitter[i]

            y = method_data[:, 2]*100
            std = method_data[:, 3]*100
            method_name = nice_method_names[methods[i]]
            plt.errorbar(x, y, yerr=std, label=method_name, fmt=markers[i]+"--", alpha=0.8)

        if show_title:
            plt.title("Adaptation and Generalization Methods on "+dataset_name)

        if FLAGS.similarity:
            xaxis = "Feature-level Wasserstein distance between source(s) and target"
        else:
            xaxis = "Number of source domains"

        ax.set_xlabel(xaxis)
        ax.set_ylabel("Target Domain Accuracy (%)")

        if legend_separate:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            export_legend(legend, filename=prefix+"_key."+suffix)
            legend.remove()
        else:
            # Put legend outside the graph http://stackoverflow.com/a/4701285
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if save_plot:
            plt.savefig(prefix+"_"+dataset_name+"."+suffix, bbox_inches='tight')
            plt.close()

    if not save_plot:
        plt.show()


def main(argv):
    datasets = [
        "test1",
    ]

    for dataset in datasets:
        plot_multisource(dataset, save_plot=True, show_title=True,
            legend_separate=False, suffix="png")


if __name__ == "__main__":
    app.run(main)
