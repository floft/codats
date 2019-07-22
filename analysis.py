#!/usr/bin/env python3
"""
Analyze the results
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

            if beginning_match(valid_header, line):
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


def compute_mean_std(df, name, ignore_label_flipping):
    data = df[name]

    if ignore_label_flipping:
        # To ignore label flipping (assuming this is binary classification), we
        # flip anything less than 50% to over 50%
        data[data < 0.5] = 1 - data[data < 0.5]

    # I expect there to only be 3 or 5 of each... if not, warn
    length = len(data)

    if length != 3 and length != 5:
        print("Warning: number of runs ", length, "(not 3 or 5)")

    # ddof=0 is the numpy default, ddof=1 is Pandas' default
    return data.mean(), data.std(ddof=0)


def compute_val_stats(df, ignore_label_flipping):
    return compute_mean_std(df, "Accuracy at Step", ignore_label_flipping)


def compute_eval_stats(df, has_target_clasifier=False, ignore_label_flipping=False):
    names = ["Train A", "Test A", "Train B", "Test B"]
    if has_target_clasifier:
        names += ["Target Train A", "Target Test A", "Target Train B", "Target Test B"]
    data = [[name]+list(compute_mean_std(df, name, ignore_label_flipping)) for name in names]
    return pd.DataFrame(data=data, columns=["Dataset", "Avg", "Std"])


def parse_name(name):
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


def all_stats(files, recompute_averages=True, sort_on_test=False,
        sort_on_b=False, sort_by_name=False, has_target_clasifier=False,
        ignore_label_flipping=False):
    stats = []

    for name, file in files:
        parse_result = parse_file(file)

        if parse_result is None:
            print("Warning: skipping", file)
            continue

        validation, traintest, averages = parse_result

        if recompute_averages:
            averages = compute_eval_stats(traintest,
                has_target_clasifier, ignore_label_flipping)

        validavg = compute_val_stats(validation, ignore_label_flipping)

        stats.append({
            "name": name,
            "parameters": parse_name(name),
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


def plot_results(results, save_plot=False, save_prefix="plot_", title_suffix=""):
    """ Generate a plot for each dataset comparing how well the various
    adaptation methods handle varying amounts of domain shift """
    datasets = {}

    for result in results:
        params = result["parameters"]
        avgs = result["averages"]
        dataset = params["dataset"]
        method = params["method"]
        adaptation = params["adaptation"]
        source_accuracy = avgs[avgs["Dataset"] == "Test A"]["Avg"].values[0]
        source_accuracy_std = avgs[avgs["Dataset"] == "Test A"]["Std"].values[0]
        target_accuracy = avgs[avgs["Dataset"] == "Test B"]["Avg"].values[0]
        target_accuracy_std = avgs[avgs["Dataset"] == "Test B"]["Std"].values[0]

        # For upper bound, we set the source to the target
        if method == "upper":
            target_accuracy = source_accuracy
            target_accuracy_std = source_accuracy_std

        # Use nice names for the plot
        nice_method_names = {
            "none": "Lower Bound (no adaptation)",
            "upper": "Upper Bound (train on target)",
            "dann": "DANN",
            "cyclegan": "CycleGAN",
            "cyclegan_dann": "CycleGAN+DANN",
            "cycada": "CyCADA",
            "deepjdot": "DeepJDOT",
            "rdann": "R-DANN",
            "vrada": "VRADA",
        }

        method = nice_method_names[method]

        if dataset not in datasets:
            datasets[dataset] = {}
        if method not in datasets[dataset]:
            datasets[dataset][method] = []

        # (x,y,error) - x axis is adaptation problem {0..5} and y axes is accuracy
        datasets[dataset][method].append([adaptation, target_accuracy,
            target_accuracy_std])
        # Sort on adaptation problem, so we get 0, 1, 2, 3, 4, 5
        datasets[dataset][method].sort(key=lambda x: x[0])

    for dataset_name, dataset_values in datasets.items():
        methods = list(dataset_values.keys())
        data = list(dataset_values.values())
        jitter = gen_jitter(len(data))  # "dodge" points so they don't overlap

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=100)

        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0] + jitter[i]
            y = method_data[:, 1]*100
            std = method_data[:, 2]*100
            plt.errorbar(x, y, yerr=std, label=methods[i], fmt="o--", alpha=0.8)

        plt.title("Various Adaptation Methods on Dataset "+dataset_name+title_suffix)
        ax.set_xlabel("Domain Shift (0 = no shift, 5 = the most shift)")
        ax.set_ylabel("Target Domain Accuracy (%)")

        # Put legend outside the graph http://stackoverflow.com/a/4701285
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if save_plot:
            plt.savefig(save_prefix+dataset_name+".png", bbox_inches='tight')

    if not save_plot:
        plt.show()


if __name__ == "__main__":
    # Ignoring label flipping won't work on best model since if it flips the
    # labels, it'll pick (and actually save during training) the wrong "best"
    # model
    #files = get_tuning_files(".", prefix="results_runwalk01_best-")
    #results = all_stats(files, sort_by_name=True, ignore_label_flipping=False)
    #plot_results(results, save_plot=True, save_prefix="plot_runwalk01_best_", title_suffix=" (best)")

    for dataset in ["runwalk01", "runwalk2"]:
        files = get_tuning_files(".", prefix="results_"+dataset+"_last-")

        results = all_stats(files, sort_by_name=True, ignore_label_flipping=False)
        plot_results(results, save_plot=True,
            save_prefix="plot_"+dataset+"_last_normal_",
            title_suffix=" (last, as is)")

        results = all_stats(files, sort_by_name=True, ignore_label_flipping=True)
        plot_results(results, save_plot=True,
            save_prefix="plot_"+dataset+"_last_ignore_",
            title_suffix=" (last, ignore flipping)")
