#!/usr/bin/env python3
"""
Evaluate models

This takes a model trained by main.py and evaluates it on both:
    train - this is the normal "train" set for these datasets
    valid/test - when combined, this is the same as the normal "test" set for
        these datasets

It'll output the {source,target}-{train,test} accuracies for comparison with
other methods.
"""
import os
import pathlib
import multiprocessing
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

import models
import methods
import file_utils
import load_datasets

from pool import run_job_pool
from metrics import Metrics
from checkpoints import CheckpointManager
from gpu_memory import set_gpu_memory


FLAGS = flags.FLAGS

# Same as in main.py
flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
# Specific for evaluation
flags.DEFINE_float("gpumem", 8140, "GPU memory to let TensorFlow use, in MiB (divided among jobs)")
flags.DEFINE_string("match", "*-*-*", "String matching to determine which logs/models to process")
flags.DEFINE_integer("jobs", 4, "Number of TensorFlow jobs to run at once")
flags.DEFINE_integer("gpus", 1, "Split jobs between GPUs -- overrides jobs (1 == run multiple jobs on first GPU)")
flags.DEFINE_boolean("last", False, "Use last model rather than one with best validation set performance")


def get_gpus():
    """
    Get the list of GPU ID's that SLURM is giving us
    """
    return [int(x) for x in os.getenv("SLURM_JOB_GPUS", "").split(",")]


def get_pool_id():
    """
    Get unique ID for this process in the job pool. It'll range from
    1 to max_jobs. See: https://stackoverflow.com/a/10192611/2698494

    Will return a number in [0,max_jobs)
    """
    current = multiprocessing.current_process()
    return current._identity[0]-1


def get_models_to_evaluate():
    """
    Returns the models to evaluate based on what is in logdir and modeldir
    specified as command line arguments. The matching pattern is specified by
    the match argument.

    Returns: [(log_dir, model_dir, dataset_name, sources, target,
        model_name, method_name), ...]
    """
    files = pathlib.Path(FLAGS.logdir).glob(FLAGS.match)
    models_to_evaluate = []

    for log_dir in files:
        # Get information from config files
        config = file_utils.get_config(log_dir)

        dataset_name = config["dataset"]
        # We want the string versions of the sources and target, not integers
        sources = ",".join([str(x) for x in config["sources"]])
        target = str(config["target"])
        model_name = config["model"]
        method_name = config["method"]
        assert method_name in methods.names(), "Unknown method "+method_name

        model_dir = os.path.join(FLAGS.modeldir, log_dir.stem)
        assert os.path.exists(model_dir), "Model does not exist "+str(model_dir)
        assert model_name in models.names(), "Unknown model "+str(model_name)

        models_to_evaluate.append((str(log_dir), model_dir, dataset_name,
            sources, target, model_name, method_name))

    return models_to_evaluate


def print_results(results):
    """ Print out the accuracies on {Train,Test}{A,B} on each target-fold pair
    followed by the averages and standard deviations of these. """
    source_train = []
    source_test = []
    target_train = []
    target_test = []

    print("Log Dir;Dataset;Sources;Target;Model;Method;Train A;Test A;Train B;Test B")
    for log_dir, dataset_name, sources, target, model, method, \
            s_train, t_train, s_test, t_test in results:
        if s_train is not None and s_test is not None:
            # If we don't have a target domain, just output zero
            if t_train is None:
                t_train = 0
            if t_test is None:
                t_test = 0

            print(log_dir + ";" + dataset_name + ";" + sources + ";"
                + target + ";" + model + ";" + method + ";"
                + str(s_train) + ";" + str(s_test) + ";"
                + str(t_train) + ";" + str(t_test))

            # Task classifier
            source_train.append(s_train)
            source_test.append(s_test)
            target_train.append(t_train)
            target_test.append(t_test)

    # Task classifier
    source_train = np.array(source_train)
    source_test = np.array(source_test)
    target_train = np.array(target_train)
    target_test = np.array(target_test)

    if len(source_train) > 0 and len(source_test) > 0 \
            and len(target_train) > 0 and len(target_test) > 0:
        print()
        print()
        print("Dataset;Avg;Std")
        # Task classifier
        print("Train A;" + str(source_train.mean()) + ";" + str(source_train.std()))
        print("Test A;" + str(source_test.mean()) + ";" + str(source_test.std()))
        print("Train B;" + str(target_train.mean()) + ";" + str(target_train.std()))
        print("Test B;" + str(target_test.mean()) + ";" + str(target_test.std()))

        print()
        print()
        print("Averages over", len(source_train), "runs")
        # Task classifier
        print("Train A \t Avg:", source_train.mean(), "\t Std:", source_train.std())
        print("Test A  \t Avg:", source_test.mean(), "\t Std:", source_test.std())
        print("Train B \t Avg:", target_train.mean(), "\t Std:", target_train.std())
        print("Test B  \t Avg:", target_test.mean(), "\t Std:", target_test.std())
    else:
        print("No data.")


def process_model(log_dir, model_dir, dataset_name, sources, target, model_name,
        method_name, gpumem, multi_gpu):
    """ Evaluate a model on the train/test data and compute the results """
    # We need to do this in the process since otherwise TF can't access cuDNN
    # for some reason. But, we only need to do this the first time we create the
    # process. It'll error on any subsequent calls (since the pool re-uses
    # process).
    try:
        set_gpu_memory(FLAGS.gpumem)
    except RuntimeError:
        pass  # Ignore: "RuntimeError: GPU options must be set at program startup"

    # Get what GPU to run this on, otherwise it'll default to whatever the
    # first one is
    if multi_gpu:
        # Get all GPUs SLURM gave to us and what process in the pool this is
        available_gpus = get_gpus()
        pool_id = get_pool_id()

        # Pick which one based on pool id
        gpu = available_gpus[pool_id]

        # Only let TensorFlow see this GPU. I tried tf.device, but somehow
        # each process still put some stuff into memory on every GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Load datasets
    source_datasets, target_dataset = load_datasets.load_da(dataset_name,
        sources, target, test=True)

    # Load the method, model, etc.
    # Note: {global,num}_step are for training, so it doesn't matter what
    # we set them to here
    method = methods.load(method_name,
        source_datasets=source_datasets,
        target_dataset=target_dataset,
        global_step=1, total_steps=1)

    # Load model from checkpoint
    checkpoint = tf.train.Checkpoint(**method.checkpoint_variables)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir)

    if FLAGS.last:
        checkpoint_manager.restore_latest()
        max_accuracy_step = checkpoint_manager.latest_step()
        max_accuracy = 0  # We don't really care...
    else:
        checkpoint_manager.restore_best()
        max_accuracy_step = checkpoint_manager.best_step()
        max_accuracy = checkpoint_manager.best_validation

    # Print which step we're loading the model for
    print(log_dir + ";" + dataset_name + ";" + sources + ";" + target + ";"
        + method_name + ";" + model_name + ";"
        + str(max_accuracy_step) + ";" + str(max_accuracy))

    # If not found, give up
    if not checkpoint_manager.found:
        return log_dir, dataset_name, sources, target, model_name, method_name, \
            None, None, None, None

    # Metrics
    has_target_domain = target_dataset is not None
    metrics = Metrics(log_dir, method, source_datasets, target_dataset,
        has_target_domain)

    # Evaluate on both datasets
    metrics.train_eval()
    metrics.test(evaluation=True)

    # Get results
    results = metrics.results()
    s_train = results["accuracy_task/source/training"]
    s_test = results["accuracy_task/source/validation"]

    if target_dataset is not None:
        t_train = results["accuracy_task/target/training"]
        t_test = results["accuracy_task/target/validation"]
    else:
        t_train = None
        t_test = None

    return log_dir, dataset_name, sources, target, model_name, method_name, \
        s_train, t_train, s_test, t_test


def main(argv):
    # If single GPU, then split memory between jobs. But, if multiple GPUs,
    # each GPU has its own memory, so don't divide it up.
    #
    # If multiple GPUs, the jobs are split by GPU not by the "jobs" argument, so
    # ignore it and just set jobs to the GPU count.
    if FLAGS.gpus == 1:
        jobs = FLAGS.jobs
        gpumem = FLAGS.gpumem / jobs
        multi_gpu = False
    else:
        jobs = FLAGS.gpus
        gpumem = FLAGS.gpumem
        multi_gpu = True

    # Find models in the model/log directories
    models_to_evaluate = get_models_to_evaluate()

    # Run in parallel
    commands = []

    for model_params in models_to_evaluate:
        commands.append((*model_params, gpumem, multi_gpu))

    # Also prints which models we load
    print("Log Dir;Dataset;Sources;Target;Model;Method;Best Step;Accuracy at Step")
    results = run_job_pool(process_model, commands, cores=jobs)

    # Print results, averages, etc.
    print_results(results)


if __name__ == "__main__":
    app.run(main)
