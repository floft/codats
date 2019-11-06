#!/usr/bin/env python3
"""
From the results of kamiak_{train,eval}_tune.srun pick the best hyperparameters
for each dataset-method pair

Outputs/prints a hyperparameter dictionary to put in pick_multi_source.py,
specifying which hyperparameters to pass during training.
"""
from absl import app
from absl import flags

from analysis import get_tuning_files, all_stats

FLAGS = flags.FLAGS

flags.DEFINE_enum("selection", "best_source", ["best_source", "best_target"], "Which model to select")

# Which parameters were changed during tuning, must be saved/available in the
# config.yaml file
parameter_list = [
    "batch_division",
    "train_batch",
    "lr",
]


def params_to_str(params):
    args = []

    for i, param in enumerate(params):
        args.append("--"+parameter_list[i]+"="+str(param))

    return " ".join(args)


def main(argv):
    dataset = "tune2"
    variant = FLAGS.selection
    files = get_tuning_files("results", prefix="results_"+dataset+"_"+variant+"-")
    tuning_results = all_stats(files)

    # Group by [dataset][method][hyperparams] since we want to select the best
    # hyperparameters for each dataset-method pair.
    indexed = {}

    for result in tuning_results:
        config = result["parameters"]["config"]
        dataset = config["dataset"]
        method = config["method"]
        hyperparams = tuple([config[p] for p in parameter_list])

        if dataset not in indexed:
            indexed[dataset] = {}
        if method not in indexed[dataset]:
            indexed[dataset][method] = {}
        if hyperparams not in indexed[dataset][method]:
            indexed[dataset][method][hyperparams] = []
        indexed[dataset][method][hyperparams].append(result)

    # Average over the runs for each set of parameters, then pick the one with
    # the highest source or target accuracy. Also, output the target accuracy
    # we would get using these parameters (note: on target valid set, not test
    # set). Hopefully it doesn't change much from the hyperparameters picked
    # with source vs. target accuracy.
    print("Dataset;Method;BestAccuracy;TargetAccuracy;BestHyperParameters")
    final_params = {}

    for dataset in indexed.keys():
        for method in indexed[dataset].keys():
            best_acc = 0
            best_target_acc = []
            best_params = []

            # Average over runs with each set of hyperparameters
            for hyperparams in indexed[dataset][method].keys():
                accuracies = []
                target_accuracies = []

                for result in indexed[dataset][method][hyperparams]:
                    avgs = result["averages"]
                    # Note: "Test" is actually the validation data since in
                    # kamiak_eval_tune.srun we pass --notest to main_eval.py
                    source_acc = avgs[avgs["Dataset"] == "Test A"]["Avg"].values[0]
                    target_acc = avgs[avgs["Dataset"] == "Test B"]["Avg"].values[0]

                    if variant == "best_source":
                        accuracies.append(source_acc)
                    else:
                        accuracies.append(target_acc)

                    target_accuracies.append(target_acc)

                if len(accuracies) > 0:
                    accuracy = sum(accuracies)/len(accuracies)
                    target_accuracy = sum(target_accuracies)/len(target_accuracies)
                else:
                    accuracy = 0
                    target_accuracy = 0
                    print("Warning: no runs found for", dataset, method, hyperparams)

                # Update best parameters if this is better, if it's the same,
                # then add to the list of good parameters
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_target_acc = [target_accuracy]
                    best_params = [hyperparams]
                elif accuracy == best_acc:
                    best_params.append(hyperparams)
                    best_target_acc.append(target_accuracy)

            # Print out the best we found
            best_target_acc = sum(best_target_acc)/len(best_target_acc)
            print(dataset, method, best_acc, best_target_acc, *best_params, sep=";")

            # Save final parameters to put into pick_multi_source.py
            if dataset not in final_params:
                final_params[dataset] = {}
            assert method not in final_params[dataset]
            final_params[dataset][method] = best_params

    # Output dictionary for final parameters for pick_multi_source.py
    print("hyperparameters = {")
    for dataset in final_params.keys():
        print("    \""+dataset+"\": {")
        for method in final_params[dataset].keys():
            best_params = final_params[dataset][method]
            print("        \""+method+"\": \""+params_to_str(best_params[0])+"\",")

            # Alternative parameters for equivalent accuracy
            for i in range(1, len(best_params)):
                print("        #\""+method+"\": \""+params_to_str(best_params[i])+"\",")
        print("    },")
    print("}")


if __name__ == "__main__":
    app.run(main)
