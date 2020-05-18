#!/usr/bin/env python3
"""
Time-series adaptation
"""
import os
import sys
import time
import tensorflow as tf

from absl import app
from absl import flags

import models
import methods
import file_utils
import load_datasets

from datasets import datasets
from metrics import Metrics
from checkpoints import CheckpointManager
from gpu_memory import set_gpu_memory


FLAGS = flags.FLAGS

flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
flags.DEFINE_enum("method", None, methods.list_methods(), "What method of domain adaptation to perform (or none)")
flags.DEFINE_enum("model", "fcn", models.list_models(), "What model to use (note: ignored for vrada/rdann methods)")
flags.DEFINE_enum("dataset", None, datasets.list_datasets(), "What dataset to use (e.g. \"ucihar\")")
flags.DEFINE_string("sources", None, "Which source domains to use (e.g. \"1,2,3\")")
flags.DEFINE_string("target", "", "What target domain to use (e.g. \"4\", can be blank for no target)")
flags.DEFINE_string("uid", None, "A unique ID saved in the log/model folder names to avoid conflicts")
flags.DEFINE_integer("ensemble", 1, "Number of models in the ensemble, 1 = no ensemble")
flags.DEFINE_integer("steps", 30000, "Number of training steps to run")
flags.DEFINE_float("gpumem", 2300, "GPU memory to let TensorFlow use, in MiB (0 for all)")
flags.DEFINE_integer("model_steps", 0, "Save the model every so many steps (0 for only when log_val_steps)")
flags.DEFINE_integer("log_train_steps", 500, "Log training information every so many steps (0 for never)")
flags.DEFINE_integer("log_val_steps", 4000, "Log validation information every so many steps (also saves model, 0 for only at end)")
flags.DEFINE_integer("log_plots_steps", 0, "Log plots every so many steps (0 for never)")
flags.DEFINE_boolean("test", False, "Use real test set for evaluation rather than validation set")
flags.DEFINE_boolean("subdir", True, "Save models/logs in subdirectory of prefix")
flags.DEFINE_boolean("debug", False, "Start new log/model/images rather than continuing from previous run")
flags.DEFINE_boolean("time_training", False, "Print how long each step takes, instead of every 100 steps")
flags.DEFINE_boolean("moving_average", False, "Whether to use an exponential moving average of the weights rather than the weights directly (requires tensorflow_addons)")
flags.DEFINE_boolean("share_most_weights", False, "Instead of regularizing weights in heterogeneous domain adaptation, share same-shape weights")
flags.DEFINE_integer("debugnum", -1, "Specify exact log/model/images number to use rather than incrementing from last. (Don't pass both this and --debug at the same time.)")

flags.mark_flag_as_required("method")
flags.mark_flag_as_required("dataset")
flags.mark_flag_as_required("sources")
flags.mark_flag_as_required("uid")


def get_directory_names():
    """ Figure out the log and model directory names """
    prefix = FLAGS.dataset+"-"+FLAGS.uid+"-"+FLAGS.method

    # Use the number specified on the command line (higher precedence than --debug)
    if FLAGS.debugnum >= 0:
        attempt = FLAGS.debugnum
        print("Debugging attempt:", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    # Find last one, increment number
    elif FLAGS.debug:
        attempt = file_utils.last_modified_number(FLAGS.logdir, prefix+"*")
        attempt = attempt+1 if attempt is not None else 1
        print("Debugging attempt:", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    # If no debugging modes, use the model and log directory with only the "prefix"
    # (even though it's not actually a prefix in this case, it's the whole name)
    elif FLAGS.subdir:
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    else:
        model_dir = FLAGS.modeldir
        log_dir = FLAGS.logdir

    return model_dir, log_dir


def main(argv):
    # Allow running multiple at once
    set_gpu_memory(FLAGS.gpumem)

    # Figure out the log and model directory filenames
    assert FLAGS.uid != "", "uid cannot be an empty string"
    model_dir, log_dir = get_directory_names()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Write config file about what dataset we're using, sources, target, etc.
    file_utils.write_config_from_args(log_dir)

    # Load datasets
    source_datasets, target_dataset = load_datasets.load_da(FLAGS.dataset,
        FLAGS.sources, FLAGS.target, test=FLAGS.test)

    # Need to know which iteration for learning rate schedule
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Load the method, model, etc.
    method = methods.get_method(FLAGS.method,
        source_datasets=source_datasets,
        target_dataset=target_dataset,
        model_name=FLAGS.model,
        global_step=global_step,
        total_steps=FLAGS.steps,
        ensemble_size=FLAGS.ensemble,
        moving_average=FLAGS.moving_average,
        share_most_weights=FLAGS.share_most_weights)

    # Check that this method is supposed to be trainable. If not, we're done.
    # (Basically, we just wanted to write the config file for non-trainable
    # models.)
    if not method.trainable:
        print("Method not trainable. Exiting now.")
        return

    # Checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step, **method.checkpoint_variables)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir)
    checkpoint_manager.restore_latest()

    # Metrics
    has_target_domain = target_dataset is not None
    metrics = Metrics(log_dir, method, source_datasets, target_dataset,
        has_target_domain)

    # Start training
    #
    # TODO maybe eventually rewrite this in the more-standard Keras way
    # See: https://www.tensorflow.org/guide/keras/train_and_evaluate
    for i in range(int(global_step), FLAGS.steps+1):
        t = time.time()
        data_sources, data_target = method.train_step()
        global_step.assign_add(1)
        t = time.time() - t

        if FLAGS.time_training:
            print(int(global_step), t, sep=",")
            continue  # skip evaluation, checkpointing, etc. when timing

        if i%1000 == 0:
            print("step %d took %f seconds"%(int(global_step), t))
            sys.stdout.flush()  # otherwise waits till the end to flush on Kamiak

        # Metrics on training/validation data
        if FLAGS.log_train_steps != 0 and i%FLAGS.log_train_steps == 0:
            metrics.train(data_sources, data_target, global_step, t)

        # Evaluate every log_val_steps but also at the last step
        validation_accuracy_source = None
        validation_accuracy_target = None
        if (FLAGS.log_val_steps != 0 and i%FLAGS.log_val_steps == 0) \
                or i == FLAGS.steps:
            validation_accuracy_source, validation_accuracy_target \
                = metrics.test(global_step)

        # Checkpoints -- Save either if at the right model step or if we found
        # a new validation accuracy. If this is better than the previous best
        # model, we need to make a new checkpoint so we can restore from this
        # step with the best accuracy.
        if (FLAGS.model_steps != 0 and i%FLAGS.model_steps == 0) \
                or validation_accuracy_source is not None:
            checkpoint_manager.save(int(global_step-1),
                validation_accuracy_source, validation_accuracy_target)

        # Plots
        if FLAGS.log_plots_steps != 0 and i%FLAGS.log_plots_steps == 0:
            metrics.plots(global_step)

    # We're done -- used for hyperparameter tuning
    file_utils.write_finished(log_dir)


if __name__ == "__main__":
    app.run(main)
