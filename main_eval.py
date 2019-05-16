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
import load_datasets

from pool import run_job_pool
from models import DomainAdaptationModel
from metrics import Metrics
from checkpoints import CheckpointManager
from gpu_memory import set_gpu_memory


FLAGS = flags.FLAGS

methods = ["none", "dann", "pseudo", "instance"]

# Same as in main.py
flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
flags.DEFINE_boolean("target_classifier", True, "Use separate target classifier in ATT or Pseudo[-labeling] methods")
flags.DEFINE_boolean("best_target", True, "If target_classifier, then pick best model based on target classifier accuracy (not task classifier accuracy)")
# Specific for evaluation
flags.DEFINE_float("gpumem", 0.8, "Percentage of GPU memory to let TensorFlow use (divided among jobs)")
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

    Returns: [(log_dir, model_dir, source, target, model_name, method_name), ...]
    """
    files = pathlib.Path(FLAGS.logdir).glob(FLAGS.match)
    models_to_evaluate = []

    for log_dir in files:
        items = str(log_dir.stem).split("-")
        assert len(items) >= 3 or len(items) <= 5, \
            "name should be one of source-target-model-{-method{-num,},-num,}"

        method_name = "none"

        if len(items) == 3:
            source, target, model_name = items
        elif len(items) == 4 or len(items) == 5:
            source, target, model_name, keyword = items[:4]

            if keyword in methods:
                method_name = keyword
            else:
                pass  # probably a debug number, which we don't care about

        model_dir = os.path.join(FLAGS.modeldir, log_dir.stem)
        assert os.path.exists(model_dir), "Model does not exist "+str(model_dir)

        assert model_name in models.names(), "Unknown model "+str(model_name)
        assert source in load_datasets.names(), "Unknown source "+str(source)
        assert target in [""]+load_datasets.names(), "Unknown target "+str(target)

        models_to_evaluate.append((str(log_dir), model_dir, source, target,
            model_name, method_name))

    return models_to_evaluate


def print_results(results):
    """ Print out the accuracies on {Train,Test}{A,B} on each target-fold pair
    followed by the averages and standard deviations of these. """
    source_train = []
    source_test = []
    target_train = []
    target_test = []
    target_source_train = []
    target_source_test = []
    target_target_train = []
    target_target_test = []

    print("Log Dir,Source,Target,Model,Method,"
        "Train A,Test A,Train B,Test B,Target Train A,"
        "Target Test A,Target Train B,Target Test B")
    for log_dir, source, target, model, method, \
            s_train, t_train, s_test, t_test, \
            target_s_train, target_s_test, target_t_train, target_t_test \
            in results:
        if s_train is not None and s_test is not None:
            # If we don't have a target domain, just output zero
            if t_train is None:
                t_train = 0
            if t_test is None:
                t_test = 0

            # If we don't have a target classifier, just output zero
            if target_s_train is None:
                target_s_train = 0
            if target_s_test is None:
                target_s_test = 0
            if target_t_train is None:
                target_t_train = 0
            if target_t_test is None:
                target_t_test = 0

            print(log_dir + "," + source + "," + target + ","
                + model + "," + method + ","
                + str(s_train) + "," + str(s_test) + ","
                + str(t_train) + "," + str(t_test) + ","
                + str(target_s_train) + "," + str(target_s_test) + ","
                + str(target_t_train) + "," + str(target_t_test))

            # Task classifier
            source_train.append(s_train)
            source_test.append(s_test)
            target_train.append(t_train)
            target_test.append(t_test)
            # Target classifier
            target_source_train.append(target_s_train)
            target_source_test.append(target_s_test)
            target_target_train.append(target_t_train)
            target_target_test.append(target_t_test)

    # Task classifier
    source_train = np.array(source_train)
    source_test = np.array(source_test)
    target_train = np.array(target_train)
    target_test = np.array(target_test)
    # Target classifier
    target_source_train = np.array(target_source_train)
    target_source_test = np.array(target_source_test)
    target_target_train = np.array(target_target_train)
    target_target_test = np.array(target_target_test)

    if len(source_train) > 0 and len(source_test) > 0 \
            and len(target_train) > 0 and len(target_test) > 0:
        print()
        print()
        print("Dataset,Avg,Std")
        # Task classifier
        print("Train A," + str(source_train.mean()) + "," + str(source_train.std()))
        print("Test A," + str(source_test.mean()) + "," + str(source_test.std()))
        print("Train B," + str(target_train.mean()) + "," + str(target_train.std()))
        print("Test B," + str(target_test.mean()) + "," + str(target_test.std()))
        # Target classifier
        print("Target Train A," + str(target_source_train.mean()) + "," + str(target_source_train.std()))
        print("Target Test A," + str(target_source_test.mean()) + "," + str(target_source_test.std()))
        print("Target Train B," + str(target_target_train.mean()) + "," + str(target_target_train.std()))
        print("Target Test B," + str(target_target_test.mean()) + "," + str(target_target_test.std()))

        print()
        print()
        print("Averages over", len(source_train), "runs (each home is 3-fold CV)")
        # Task classifier
        print("Train A \t Avg:", source_train.mean(), "\t Std:", source_train.std())
        print("Test A  \t Avg:", source_test.mean(), "\t Std:", source_test.std())
        print("Train B \t Avg:", target_train.mean(), "\t Std:", target_train.std())
        print("Test B  \t Avg:", target_test.mean(), "\t Std:", target_test.std())
        # Target classifier
        print("Target Train A \t Avg:", target_source_train.mean(), "\t Std:", target_source_train.std())
        print("Target Test A  \t Avg:", target_source_test.mean(), "\t Std:", target_source_test.std())
        print("Target Train B \t Avg:", target_target_train.mean(), "\t Std:", target_target_train.std())
        print("Target Test B  \t Avg:", target_target_test.mean(), "\t Std:", target_target_test.std())
    else:
        print("No data.")


def process_model(log_dir, model_dir, source, target, model_name, method_name,
        gpumem, multi_gpu):
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
    if target != "":
        source_dataset, target_dataset = load_datasets.load_da(source, target,
            test=True)
        assert source_dataset.num_classes == target_dataset.num_classes, \
            "Adapting from source to target with different classes not supported"
    else:
        raise NotImplementedError("currently don't support only source")
        source_dataset = load_datasets.load(source, test=True)
        target_dataset = None

    # Evaluation datasets if we have the dataset
    source_dataset_train = source_dataset.train_evaluation
    target_dataset_train = target_dataset.train_evaluation \
        if target_dataset is not None else None
    source_dataset_test = source_dataset.test_evaluation
    target_dataset_test = target_dataset.test_evaluation \
        if target_dataset is not None else None

    # Information about domains
    num_classes = source_dataset.num_classes

    # Build our model
    # Note: {global,num}_step are for training, so it doesn't matter what
    # we set them to here
    global_step = 1
    num_steps = 1
    model = DomainAdaptationModel(num_classes, model_name,
        global_step, num_steps)

    # Does this method use a target classifier?
    has_target_classifier = method_name in ["pseudo", "instance"] \
        and FLAGS.target_classifier

    # Load model from checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir,
        target=has_target_classifier)

    if FLAGS.last:
        checkpoint_manager.restore_latest()
        max_accuracy_step = checkpoint_manager.latest_step()
        max_accuracy = 0  # We don't really care...
    else:
        checkpoint_manager.restore_best(FLAGS.best_target)
        max_accuracy_step = checkpoint_manager.best_step(FLAGS.best_target)

        if has_target_classifier and FLAGS.best_target:
            max_accuracy = checkpoint_manager.best_target_validation
        else:
            max_accuracy = checkpoint_manager.best_validation

    # Print which step we're loading the model for
    print(log_dir + "," + source + "," + target + ","
        + method_name + "," + model_name + ","
        + str(max_accuracy_step) + "," + str(max_accuracy))

    # If not found, give up
    if not checkpoint_manager.found:
        return source, target, model_name, method_name, \
            None, None, None, None, \
            None, None, None, None

    # Metrics
    have_target_domain = target_dataset is not None
    metrics = Metrics(log_dir, source_dataset,
            None, None, have_target_domain,
            target_classifier=has_target_classifier,
            enable_compile=False)

    # Evaluate on both datasets
    metrics.train(model, source_dataset_train, target_dataset_train, evaluation=True)
    metrics.test(model, source_dataset_test, target_dataset_test, evaluation=True)

    # Get results
    results = metrics.results()
    s_train = results["accuracy_task/source/training"]
    s_test = results["accuracy_task/source/validation"]

    target_s_train = None
    target_s_test = None
    target_t_train = None
    target_t_test = None

    if has_target_classifier:
        target_s_train = results["accuracy_target/source/training"]
        target_s_test = results["accuracy_target/source/validation"]

    if target_dataset is not None:
        t_train = results["accuracy_task/target/training"]
        t_test = results["accuracy_task/target/validation"]

        if has_target_classifier:
            target_t_train = results["accuracy_target/target/training"]
            target_t_test = results["accuracy_target/target/validation"]
    else:
        t_train = None
        t_test = None

    return log_dir, source, target, model_name, method_name, \
        s_train, t_train, s_test, t_test, \
        target_s_train, target_s_test, target_t_train, target_t_test


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
    print("Log Dir,Source,Target,Model,Method,Best Step,Accuracy at Step")
    results = run_job_pool(process_model, commands, cores=jobs)

    # Print results, averages, etc.
    print_results(results)


if __name__ == "__main__":
    app.run(main)
