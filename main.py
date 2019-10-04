#!/usr/bin/env python3
"""
Time-series adaptation
"""
import os
import time
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

import models
import file_utils
import load_datasets

from metrics import Metrics
from checkpoints import CheckpointManager
from gpu_memory import set_gpu_memory


FLAGS = flags.FLAGS

methods = [
    # No adaptation or training on target
    "none",

    # Multi-source domain adaptation (works with it...)
    "dann_grl", "dann_grl_gs", "dann_smooth",

    # Domain generalization
    "dann_grl_dg", "sleep_dg",
    #"aflac_dg", "ciddg_dg", TODO
]

flags.DEFINE_enum("model", None, models.names(), "What model type to use")
flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
flags.DEFINE_enum("method", None, methods, "What method of domain adaptation to perform (or none)")
flags.DEFINE_string("dataset", None, "What dataset to use (e.g. \"ucihar\")")
flags.DEFINE_string("sources", None, "Which source domains to use (e.g. \"1,2,3\")")
flags.DEFINE_string("target", "", "What target domain to use (e.g. \"4\", can be blank for no target)")
flags.DEFINE_integer("uid", 0, "A unique ID saved in the log/model folder names to avoid conflicts")
flags.DEFINE_integer("steps", 80000, "Number of training steps to run")
flags.DEFINE_float("lr", 0.001, "Learning rate for training")
flags.DEFINE_float("lr_domain_mult", 1.0, "Learning rate multiplier for training domain classifier")
flags.DEFINE_float("gpumem", 3350, "GPU memory to let TensorFlow use, in MiB (0 for all)")
flags.DEFINE_integer("model_steps", 4000, "Save the model every so many steps")
flags.DEFINE_integer("log_train_steps", 500, "Log training information every so many steps")
flags.DEFINE_integer("log_val_steps", 4000, "Log validation information every so many steps (also saves model)")
flags.DEFINE_integer("log_plots_steps", 4000, "Log plots every so many steps")
flags.DEFINE_boolean("compile_metrics", True, "Compile metrics loop with tf.function for subsequent speed (disable if std::terminate)")
flags.DEFINE_boolean("test", False, "Use real test set for evaluation rather than validation set")
flags.DEFINE_boolean("subdir", True, "Save models/logs in subdirectory of prefix")
flags.DEFINE_boolean("debug", False, "Start new log/model/images rather than continuing from previous run")
flags.DEFINE_boolean("time_training", False, "Print how long each step takes, instead of every 100 steps")
flags.DEFINE_integer("debugnum", -1, "Specify exact log/model/images number to use rather than incrementing from last. (Don't pass both this and --debug at the same time.)")

flags.mark_flag_as_required("model")
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
        logging.info("Debugging attempt: %s", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    # Find last one, increment number
    elif FLAGS.debug:
        attempt = file_utils.last_modified_number(FLAGS.logdir, prefix+"*")
        attempt = attempt+1 if attempt is not None else 1
        logging.info("Debugging attempt: %s", attempt)

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


@tf.function
def train_step_grl(data_a, data_b, model, opt, d_opt,
        task_loss, domain_loss, generalize, num_domains):
    """ Compiled DANN (with GRL) training step that we call many times """
    x_a, y_a, domain_a = data_a
    x_b, y_b, domain_b = data_b

    if generalize:
        x = x_a
        task_y_true = y_a
        domain_y_true = domain_a
        domain = "source"
    else:
        # Concatenate for adaptation - concatenate source labels with all-zero
        # labels for target since we can't use the target labels during
        # unsupervised domain adaptation
        x = tf.concat((x_a, x_b), axis=0)
        task_y_true = tf.concat((y_a, tf.zeros_like(y_b)), axis=0)
        domain = "both"

        if FLAGS.method == "dann_grl_gs":
            # We want to ignore the source labels and group them all together.
            # We can keep using the target as domain 0, but for the source,
            # set them all to be domain 1 (instead of 1, 2, ..., n)
            source_domain = tf.ones_like(domain_a)
            domain_y_true = tf.concat((source_domain, domain_b), axis=0)
        else:
            domain_y_true = tf.concat((domain_a, domain_b), axis=0)

    with tf.GradientTape(persistent=True) as tape:
        if FLAGS.method == "dann_smooth":
            task_y_pred, domain_y_pred, fe_output = model(x, training=True, domain=domain)

            # Method of feeding everything through, then filtering.
            assert isinstance(domain_y_pred, list), \
                "domain output should be list if dann_smooth"

            # MDAN losses - domain classifiers' losses weighted by task
            # classifier's loss per domain
            # https://github.com/KeiraZhao/MDAN/blob/master/model.py
            # https://github.com/KeiraZhao/MDAN/blob/master/main_amazon.py
            task_losses = []
            domain_losses = []

            for i in range(1, num_domains):
                # Skip if in this batch there's no examples of this domain
                domain_equals_i = tf.equal(domain_y_true, i)
                # count = tf.reduce_sum(tf.cast(domain_equals_i, tf.float32))

                # # TODO remove, slows down running I think
                # if count == 0:
                #     tf.print("Warning: no examples of domain", i, "in batch")

                # For classification loss, we want to ignore the target labels,
                # so same as above but without domain_y_true == 0
                where_task = tf.where(domain_equals_i)
                i_task_true = tf.gather(task_y_true, where_task)
                i_task_pred = tf.gather(task_y_pred, where_task)

                # Selection of domain i and domain 0 (target) for each domain
                # classifier's output
                where_domain = tf.where(
                    tf.math.logical_or(tf.equal(domain_y_true, 0),
                        tf.equal(domain_y_true, i)))
                i_domain_true = tf.gather(domain_y_true, where_domain)
                i_domain_pred = tf.gather(domain_y_pred[i-1], where_domain)

                # training=False to make sure we don't do any further slicing
                # that gets rid of the last half of the batch as we normally
                # would to ignore target labels with GRL
                task_losses.append(task_loss(i_task_true, i_task_pred, training=False))
                domain_losses.append(domain_loss(i_domain_true, i_domain_pred))

            # The defaults in their code?
            #gamma = 10.0
            #mu = 1e-2
            # But, for now we'll use the DANN learning rate schedule and just
            # set these to 1 for more direct comparison with DANN
            gamma = 1
            mu = 1

            d_loss = None  # Don't update separately
            loss = tf.math.log(tf.reduce_sum(
                tf.exp(gamma*(task_losses+mu*domain_losses))))/gamma

            # # MDAN losses - domain classifiers' losses weighted by task
            # # classifier's loss per domain
            # # https://github.com/KeiraZhao/MDAN/blob/master/model.py
            # # https://github.com/KeiraZhao/MDAN/blob/master/main_amazon.py
            # task_losses = []
            # domain_losses = []

            # # Ignore the domain-specific stuff we're not using
            # domain = "source"

            # for i in range(1, num_domains):
            #     # Feed data from the source domain i and the target domain (i=0)
            #     # through the model using the ith's domain classifier
            #     where_domain = tf.squeeze(tf.where(
            #         tf.math.logical_or(
            #             tf.equal(domain_y_true, 0),  # Target data
            #             tf.equal(domain_y_true, i))))  # source i data
            #     i_domain_true = tf.gather(domain_y_true, where_domain)
            #     i_x = tf.gather(x, where_domain)

            #     # Run this domain's data through the model (assuming there is
            #     # data from this domain). This is much faster than feeding in
            #     # all the data then filtering (~0.45 vs.  seconds per iteration)
            #     i_task_pred, i_domain_pred, _ = model(i_x, training=True,
            #         domain=domain, domain_classifier=i-1)

            #     # The true labels from only the source domain
            #     where_task = tf.squeeze(tf.where(tf.equal(domain_y_true, i)))
            #     i_task_true = tf.gather(task_y_true, where_task)

            #     # The predicted labels from the source domain (i.e. we can't
            #     # look at the true labels of the target domain, so throw out
            #     # those predictions - since we had to run both through the model
            #     # for the domain classifier loss)
            #     where_task_valid = tf.squeeze(tf.where(tf.equal(i_domain_true, i)))
            #     i_task_pred_valid = tf.gather(i_task_pred, where_task_valid)

            #     # training=False to make sure we don't do any further slicing
            #     # that gets rid of the last half of the batch as we normally
            #     # would to ignore target labels with GRL
            #     task_losses.append(task_loss(i_task_true, i_task_pred_valid, training=False))
            #     domain_losses.append(domain_loss(i_domain_true, i_domain_pred))

            # # The defaults in their code?
            # #gamma = 10.0
            # #mu = 1e-2
            # # But, for now we'll use the DANN learning rate schedule and just
            # # set these to 1 for more direct comparison with DANN
            # gamma = 1
            # mu = 1

            # d_loss = None  # Don't update separately
            # loss = tf.math.log(tf.reduce_sum(
            #     tf.exp(gamma*(task_losses+mu*domain_losses))))/gamma
        else:
            task_y_pred, domain_y_pred, fe_output = model(x, training=True, domain=domain)
            d_loss = domain_loss(domain_y_true, domain_y_pred)
            loss = task_loss(task_y_true, task_y_pred, training=True) + d_loss

        # TODO
        # https://www.tensorflow.org/beta/guide/keras/overview#configure_the_layers
        # https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models#layers_recursively_collect_losses_created_during_the_forward_pass
        # https://www.tensorflow.org/beta/guide/keras/training_and_evaluation#low-level_handling_of_extra_losses
        # e.g. weight decay, note sum these once after each call of model()
        # regularization = sum(model.losses)
        # loss += regularization
        # d_loss += regularization

    grad = tape.gradient(loss, model.trainable_variables_task_domain)
    if d_loss is not None:
        d_grad = tape.gradient(d_loss, model.trainable_variables_domain)
    del tape

    opt.apply_gradients(zip(grad, model.trainable_variables_task_domain))
    # Update discriminator again
    if d_loss is not None:
        d_opt.apply_gradients(zip(d_grad, model.trainable_variables_domain))


@tf.function
def train_step_none(data_a, data_b, model, opt, d_opt,
        task_loss, domain_loss):
    """ Compiled no adaptation training step that we call many times """
    x_a, y_a, _ = data_a

    with tf.GradientTape() as tape:
        task_y_pred, _, _ = model(x_a, training=True, domain="source")
        task_y_true = y_a
        loss = task_loss(task_y_true, task_y_pred, training=True)

    grad = tape.gradient(loss, model.trainable_variables_task)
    opt.apply_gradients(zip(grad, model.trainable_variables_task))


def main(argv):
    # Allow running multiple at once
    set_gpu_memory(FLAGS.gpumem)

    # Figure out the log and model directory filenames
    model_dir, log_dir = get_directory_names()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Write config file about what dataset we're using, sources, target, etc.
    file_utils.write_config_from_args(log_dir)

    # Which methods adapt (look at unlabeled target data) vs. generalize (only
    # use target data during testing)
    adapt = FLAGS.method in ["dann_grl", "dann_grl_gs", "dann_smooth"]
    generalize = FLAGS.method in ["dann_grl_dg", "sleep_dg", "aflac_dg", "ciddg_dg"]

    # For adaptation, we'll be concatenating together half source and half target
    # data, so to keep the batch_size about the same, we'll cut it in half
    train_batch = FLAGS.train_batch

    # At the moment, all methods except no adaptation use GRL
    use_grl = FLAGS.method != "none"

    if adapt and use_grl:
        train_batch = train_batch // 2

    # Input training data
    #
    # Note: "It is worth noting that only the training sets of the small image
    # datasets were used during training; the test sets used for reporting
    # scores only." (self-ensembling) -- so, only use *_test for evaluation.
    # However, for now we'll use 1000 random target test samples for the
    # validation dataset (as is common).
    source_datasets, target_dataset = load_datasets.load_da(FLAGS.dataset,
        FLAGS.sources, FLAGS.target, test=FLAGS.test, train_batch=train_batch)

    # Iterator and evaluation datasets if we have the dataset
    source_iter = iter(source_dataset.train)
    source_dataset_eval = source_dataset.test_evaluation
    target_iter = iter(target_dataset.train) \
        if target_dataset is not None else None
    target_dataset_eval = target_dataset.test_evaluation \
        if target_dataset is not None else None

    # Information about domains
    num_classes = target_dataset.num_classes
    num_domains = target_dataset.num_domains

    # Loss functions
    task_loss = models.make_task_loss(adapt and use_grl)
    domain_loss = models.make_domain_loss(adapt or generalize)

    # We need to know where we are in training for the GRL lambda schedule
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Build the model
    model = models.DomainAdaptationModel(num_classes, num_domains,
        FLAGS.model, global_step, FLAGS.steps, use_grl=use_grl)

    # Optimizers
    optimizer = tf.keras.optimizers.Adam
    opt = optimizer(FLAGS.lr)
    d_opt = optimizer(FLAGS.lr*FLAGS.lr_domain_mult)

    # Checkpoints -- but can't handle None in checkpoint
    checkpoint = {
        "global_step": global_step,
        "opt": opt,
        "d_opt": d_opt,
        "model": model,
    }

    checkpoint = tf.train.Checkpoint(**checkpoint)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir)
    checkpoint_manager.restore_latest()

    # Metrics
    has_target_domain = target_dataset is not None
    # Note: assuming that all sources have same num_classes, etc. choose source 0
    metrics = Metrics(log_dir, source_datasets[0],
        task_loss, domain_loss, has_target_domain,
        enable_compile=FLAGS.compile_metrics)

    # Start training
    for i in range(int(global_step), FLAGS.steps+1):
        # Get data for this iteration
        data_a = next(source_iter)
        data_b = next(target_iter) if target_iter is not None else None

        t = time.time()

        # Train the task model (possibly with adaptation)
        if model is not None:
            step_args = (data_a, data_b, model, opt, d_opt, task_loss, domain_loss)

            if adapt or generalize:
                assert use_grl, "currently all MS-DA/DG methods use GRL"
                train_step_grl(*step_args, generalize, num_domains)
            else:
                train_step_none(*step_args)

        global_step.assign_add(1)
        t = time.time() - t

        if FLAGS.time_training:
            print(int(global_step), t, sep=",")
            continue  # skip evaluation, checkpointing, etc. when timing

        if i%100 == 0:
            logging.info("step %d took %f seconds", int(global_step), t)

        # Metrics on training/validation data
        if i%FLAGS.log_train_steps == 0:
            metrics.train(model, data_a, data_b, global_step, t)

        # Evaluate every log_val_steps but also at the last step
        validation_accuracy = None
        if i%FLAGS.log_val_steps == 0 or i == FLAGS.steps:
            validation_accuracy = metrics.test(model, source_dataset_eval,
                target_dataset_eval, global_step)

        # Checkpoints -- Save either if at the right model step or if we found
        # a new validation accuracy. If this is better than the previous best
        # model, we need to make a new checkpoint so we can restore from this
        # step with the best accuracy.
        if i%FLAGS.model_steps == 0 or validation_accuracy is not None:
            checkpoint_manager.save(int(global_step-1), validation_accuracy)

        # Plots
        if i%FLAGS.log_plots_steps == 0:
            metrics.plots(model, source_dataset_eval, target_dataset_eval,
                adapt, global_step)

    # We're done -- used for hyperparameter tuning
    file_utils.write_finished(log_dir)


if __name__ == "__main__":
    app.run(main)
