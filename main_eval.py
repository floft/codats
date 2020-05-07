#!/usr/bin/env python3
"""
Evaluate models

This takes a model trained by main.py and evaluates it on train/valid and test sets.
"""
import os
import yaml
import pathlib
import multiprocessing
import tensorflow as tf

from absl import app
from absl import flags

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
flags.DEFINE_string("output_file", None, "Output filename to write the yaml file with the results")
flags.DEFINE_float("gpumem", 8140, "GPU memory to let TensorFlow use, in MiB (0 for all, divided among jobs)")
flags.DEFINE_string("match", "*-*-*", "String matching to determine which logs/models to process")
flags.DEFINE_integer("jobs", 4, "Number of TensorFlow jobs to run at once")
flags.DEFINE_integer("gpus", 1, "Split jobs between GPUs -- overrides jobs (1 == run multiple jobs on first GPU)")
flags.DEFINE_enum("selection", "best_source", ["last", "best_source", "best_target"], "Which model to select")
flags.DEFINE_boolean("test", True, "Whether to evaluate on the true test set or if --notest, then the validation set")

flags.mark_flag_as_required("output_file")


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


def setup_gpu_for_process(gpumem, multi_gpu):
    """ Handle setting GPU memory or which GPU to use in each process """
    # We need to do this in the process since otherwise TF can't access cuDNN
    # for some reason. But, we only need to do this the first time we create the
    # process. It'll error on any subsequent calls (since the pool re-uses
    # process).
    try:
        set_gpu_memory(gpumem)
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


def get_models_to_evaluate():
    """
    Returns the models to evaluate based on what is in logdir and modeldir
    specified as command line arguments. The matching pattern is specified by
    the match argument.

    Returns: [(log_dir, model_dir, config), ...]
    """
    files = pathlib.Path(FLAGS.logdir).glob(FLAGS.match)
    models_to_evaluate = []

    for log_dir in files:
        config = file_utils.get_config(log_dir)
        model_dir = os.path.join(FLAGS.modeldir, log_dir.stem)
        assert os.path.exists(model_dir), "Model does not exist "+str(model_dir)
        models_to_evaluate.append((str(log_dir), model_dir, config))

    return models_to_evaluate


def save_results(process_results, results_filename):
    all_process_results = []

    for pr in process_results:
        log_dir, model_dir, config, results, max_accuracy_step, \
            max_accuracy = pr

        all_process_results.append({
            "logdir": log_dir,
            "modeldir": model_dir,
            "config": config,
            "results": results,
            "max_accuracy_step": max_accuracy_step,
            "max_accuracy": max_accuracy,
        })

    # Write the config file
    with open(results_filename, "w") as f:
        yaml.dump(all_process_results, f)


def process_model(log_dir, model_dir, config, gpumem, multi_gpu):
    """ Evaluate a model on the train/test data and compute the results """
    setup_gpu_for_process(gpumem, multi_gpu)

    dataset_name = config["dataset"]
    method_name = config["method"]
    model_name = config["model"]
    sources = config["sources"]
    target = config["target"]
    moving_average = config["moving_average"]
    ensemble_size = config["ensemble"]
    share_most_weights = config["share_most_weights"]

    # Load datasets
    source_datasets, target_dataset = load_datasets.load_da(dataset_name,
        sources, target, test=FLAGS.test)

    # Load the method, model, etc.
    # Note: {global,num}_step are for training, so it doesn't matter what
    # we set them to here
    method = methods.get_method(method_name,
        source_datasets=source_datasets,
        target_dataset=target_dataset,
        model_name=model_name,
        global_step=1, total_steps=1,
        moving_average=moving_average,
        ensemble_size=ensemble_size,
        share_most_weights=share_most_weights)

    # Load model from checkpoint (if there's anything in the checkpoint)
    if len(method.checkpoint_variables) > 0:
        checkpoint = tf.train.Checkpoint(**method.checkpoint_variables)
        checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir)

        if FLAGS.selection == "last":
            checkpoint_manager.restore_latest()
            max_accuracy_step = checkpoint_manager.latest_step()
            max_accuracy = None  # We don't really care...
            found = checkpoint_manager.found_last
        elif FLAGS.selection == "best_source":
            checkpoint_manager.restore_best_source()
            max_accuracy_step = checkpoint_manager.best_step_source()
            max_accuracy = checkpoint_manager.best_validation_source
            found = checkpoint_manager.found_best_source
        elif FLAGS.selection == "best_target":
            checkpoint_manager.restore_best_target()
            max_accuracy_step = checkpoint_manager.best_step_target()
            max_accuracy = checkpoint_manager.best_validation_target
            found = checkpoint_manager.found_best_target
        else:
            raise NotImplementedError("unknown --selection argument")
    else:
        max_accuracy_step = None
        max_accuracy = None
        found = True

        # Metrics
    has_target_domain = target_dataset is not None
    metrics = Metrics(log_dir, method, source_datasets, target_dataset,
        has_target_domain)

    # If not found, give up
    if not found:
        return log_dir, model_dir, config, {}, None, None

    # Evaluate on both datasets
    metrics.train_eval()
    metrics.test(evaluation=True)

    # Get results
    results = metrics.results()

    return log_dir, model_dir, config, results, max_accuracy_step, max_accuracy


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

    if jobs == 1:  # Eases debugging, printing even if it errors
        process_results = []

        for c in commands:
            process_results.append(process_model(*c))
    else:
        process_results = run_job_pool(process_model, commands, cores=jobs)

    # Save results
    save_results(process_results, FLAGS.output_file)


if __name__ == "__main__":
    app.run(main)
