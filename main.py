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

import ot
from scipy.spatial.distance import cdist

import models
import load_datasets

from metrics import Metrics
from checkpoints import CheckpointManager
from file_utils import last_modified_number, write_finished
from gpu_memory import set_gpu_memory
from datasets import inversions

FLAGS = flags.FLAGS

flags.DEFINE_enum("model", None, models.names(), "What model type to use")
flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
flags.DEFINE_enum("method", None, ["none", "cyclegan", "forecast", "cyclegan_dann", "cycada", "dann", "deepjdot", "pseudo", "instance", "rdann", "vrada"], "What method of domain adaptation to perform (or none)")
flags.DEFINE_boolean("task", True, "Whether to perform task (classification) if true or just the mapping if false")
flags.DEFINE_enum("cyclegan_loss", "wgan", ["gan", "lsgan", "wgan", "wgan-gp"], "When using CycleGAN, which loss to use")
flags.DEFINE_enum("source", None, load_datasets.names(), "What dataset to use as the source")
flags.DEFINE_enum("target", "", [""]+load_datasets.names(), "What dataset to use as the target")
flags.DEFINE_integer("steps", 80000, "Number of training steps to run")
flags.DEFINE_integer("pretrain_steps", 0, "Number of training steps to pretrain feature extractor and task classifier on source data")
flags.DEFINE_float("lr", 0.001, "Learning rate for training")
flags.DEFINE_float("lr_domain_mult", 1.0, "Learning rate multiplier for training domain classifier")
flags.DEFINE_float("lr_target_mult", 0.5, "Learning rate multiplier for training target classifier")
flags.DEFINE_float("lr_mapping_mult", 1.0, "Learning rate multiplier for training domain mapping GAN")
flags.DEFINE_float("lr_map_d_loss_mult", 1.0, "Learning rate multiplier for training domain mapping discriminator")
flags.DEFINE_float("map_cyc_mult", 10.0, "Multiplier for cycle consistency loss for training domain mapping generator")
flags.DEFINE_boolean("minimize_true_error", False, "If the actual transform is known, minimize it to verify mapping network is capable of learning the correct mapping")
flags.DEFINE_float("jdot_alpha", 0.001, "For DeepJDOT")
flags.DEFINE_float("jdot_tloss", 0.0001, "For DeepJDOT")
flags.DEFINE_float("jdot_sloss", 1.0, "For DeepJDOT")
flags.DEFINE_float("gpumem", 3350, "GPU memory to let TensorFlow use, in MiB (0 for all)")
flags.DEFINE_integer("model_steps", 4000, "Save the model every so many steps")
flags.DEFINE_integer("log_train_steps", 500, "Log training information every so many steps")
flags.DEFINE_integer("log_val_steps", 4000, "Log validation information every so many steps (also saves model)")
flags.DEFINE_integer("log_plots_steps", 4000, "Log plots every so many steps")
flags.DEFINE_boolean("use_grl", False, "Use gradient reversal layer for training discriminator for adaptation")
flags.DEFINE_boolean("use_alt_weight", False, "Use alternate weighting for target classifier")
flags.DEFINE_boolean("use_domain_confidence", True, "Use domain classifier for confidence instead of task classifier")
flags.DEFINE_boolean("domain_invariant", True, "Train feature extractor to be domain-invariant")
flags.DEFINE_boolean("compile_metrics", True, "Compile metrics loop with tf.function for subsequent speed (disable if std::terminate)")
flags.DEFINE_boolean("test", False, "Use real test set for evaluation rather than validation set")
flags.DEFINE_boolean("subdir", True, "Save models/logs in subdirectory of prefix")
flags.DEFINE_boolean("debug", False, "Start new log/model/images rather than continuing from previous run")
flags.DEFINE_integer("debugnum", -1, "Specify exact log/model/images number to use rather than incrementing from last. (Don't pass both this and --debug at the same time.)")

flags.mark_flag_as_required("model")
flags.mark_flag_as_required("method")
flags.mark_flag_as_required("source")


def get_directory_names():
    """ Figure out the log and model directory names """
    prefix = FLAGS.source+"-"+FLAGS.target+"-"+FLAGS.model

    methods_suffix = {
        "none": "-none",
        "cyclegan": "-cyclegan",
        "cyclegan_dann": "-cyclegan_dann",
        "forecast": "-forecast",
        "cycada": "-cycada",
        "dann": "-dann",
        "deepjdot": "-deepjdot",
        "rdann": "-rdann",  # same as DANN but use with --model=rdann
        "vrada": "-vrada",  # use with "vrada" model
        "pseudo": "-pseudo",
        "instance": "-instance",
    }

    prefix += methods_suffix[FLAGS.method]

    # Use the number specified on the command line (higher precedence than --debug)
    if FLAGS.debugnum >= 0:
        attempt = FLAGS.debugnum
        logging.info("Debugging attempt: %s", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    # Find last one, increment number
    elif FLAGS.debug:
        attempt = last_modified_number(FLAGS.logdir, prefix+"*")
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
        task_loss, domain_loss):
    """ Compiled DANN (with GRL) training step that we call many times """
    x_a, y_a = data_a
    x_b, y_b = data_b

    # Concatenate for adaptation - concatenate source labels with all-zero
    # labels for target since we can't use the target labels during
    # unsupervised domain adaptation
    x = tf.concat((x_a, x_b), axis=0)
    task_y_true = tf.concat((y_a, tf.zeros_like(y_b)), axis=0)

    half_batch_size = tf.shape(x)[0] / 2
    source_domain = tf.zeros([half_batch_size, 1])
    target_domain = tf.ones([half_batch_size, 1])
    domain_y_true = tf.concat((source_domain, target_domain), axis=0)

    with tf.GradientTape() as tape, tf.GradientTape() as d_tape:
        task_y_pred, domain_y_pred, _ = model(x, training=True, domain="both")
        d_loss = domain_loss(domain_y_true, domain_y_pred)
        loss = task_loss(task_y_true, task_y_pred, training=True) + d_loss

    grad = tape.gradient(loss, model.trainable_variables_task_domain)
    opt.apply_gradients(zip(grad, model.trainable_variables_task_domain))

    # Update discriminator again
    d_grad = d_tape.gradient(d_loss, model.domain_classifier.trainable_variables)
    d_opt.apply_gradients(zip(d_grad, model.domain_classifier.trainable_variables))


def compute_vrnn_loss(vrnn_state, x, epsilon=1e-9):
    """
    Compute VRNN loss

    KL loss/divergence:
    - https://stats.stackexchange.com/q/7440
    - https://github.com/kimkilho/tensorflow-vrnn/blob/master/main.py

    Negative log likelihood loss:
    - https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf
    - https://fairyonice.github.io/Create-a-neural-net-with-a-negative-log-likelihood-as-a-loss.html
    """
    encoder_mu, encoder_sigma, decoder_mu, decoder_sigma, \
        prior_mu, prior_sigma = vrnn_state

    kl_loss = tf.reduce_mean(tf.reduce_mean(
        tf.math.log(tf.maximum(epsilon, prior_sigma)) - tf.math.log(tf.maximum(epsilon, encoder_sigma))
        + 0.5*(tf.square(encoder_sigma) + tf.square(encoder_mu - prior_mu))
        / tf.maximum(epsilon, tf.square(prior_sigma))
        - 0.5, axis=1), axis=1)

    likelihood_loss = 0.5*tf.reduce_mean(tf.reduce_mean(
        tf.square(decoder_mu - x) / tf.maximum(epsilon, tf.square(decoder_sigma))
        + tf.math.log(tf.maximum(epsilon, tf.square(decoder_sigma))),
        axis=1), axis=1)

    return tf.reduce_mean(kl_loss) + tf.reduce_mean(likelihood_loss)


@tf.function
def train_step_gan(data_a, data_b, model, opt, d_opt,
        task_loss, domain_loss, grl_schedule, global_step,
        epsilon=1e-8):
    """ Compiled multi-step (GAN-like, see Shu et al. VADA paper) adaptation
    training step that we call many times

    Feed through separately so we get different batch normalizations for each
    domain. Also optimize in a GAN-like manner rather than with GRL."""
    x_a, y_a = data_a
    x_b, _ = data_b

    # The VADA "replacing gradient reversal" (note D(f(x)) = probability of
    # being target) with non-saturating GAN-style training
    with tf.GradientTape(persistent=True) as tape:
        task_y_pred_a, domain_y_pred_a, fe_a = model(x_a, training=True, domain="source")
        _, domain_y_pred_b, fe_b = model(x_b, training=True, domain="target")

        # Correct task labels (only for source domain)
        task_y_true_a = y_a

        # Correct domain labels
        # Source domain = 0, target domain = 1
        domain_y_true_a = tf.zeros_like(domain_y_pred_a)
        domain_y_true_b = tf.ones_like(domain_y_pred_b)

        # Update feature extractor and task classifier to correctly predict
        # labels on source domain
        t_loss = task_loss(task_y_true_a, task_y_pred_a)

        if FLAGS.domain_invariant:
            # Update feature extractor to fool discriminator - min_theta step
            # (swap ones/zeros from correct, update FE rather than D weights)
            d_loss_fool = domain_loss(domain_y_true_b, domain_y_pred_a) \
                + domain_loss(domain_y_true_a, domain_y_pred_b)

            # Weight by same schedule as GRL to make this more equivalent
            d_loss_fool *= grl_schedule(global_step)

        # Update discriminator - min_D step
        # (train D to be correct, update D weights)
        d_loss_true = domain_loss(domain_y_true_a, domain_y_pred_a) \
            + domain_loss(domain_y_true_b, domain_y_pred_b)

        if FLAGS.method == "vrada":
            vrnn_state_a = fe_a[1]
            vrnn_state_b = fe_b[1]
            vrnn_loss = compute_vrnn_loss(vrnn_state_a, x_a) + \
                compute_vrnn_loss(vrnn_state_b, x_b)

    t_grad = tape.gradient(t_loss, model.trainable_variables_task)
    if FLAGS.domain_invariant:
        f_grad = tape.gradient(d_loss_fool, model.feature_extractor.trainable_variables)
    if FLAGS.method == "vrada":
        v_grad = tape.gradient(vrnn_loss, model.feature_extractor.trainable_variables)
    d_grad = tape.gradient(d_loss_true, model.domain_classifier.trainable_variables)
    del tape

    # Use opt (not d_opt) for updating FE in both cases, so Adam keeps track of
    # the updates to the FE weights
    opt.apply_gradients(zip(t_grad, model.trainable_variables_task))
    if FLAGS.domain_invariant:
        opt.apply_gradients(zip(f_grad, model.feature_extractor.trainable_variables))
    if FLAGS.method == "vrada":  # use d_opt since different updates than above?
        d_opt.apply_gradients(zip(v_grad, model.feature_extractor.trainable_variables))
    d_opt.apply_gradients(zip(d_grad, model.domain_classifier.trainable_variables))

    # TODO for inference, use the exponential moving average of the batch norm
    # statistics on the *target* data -- above will mix them probably.

    # TODO for inference use exponential moving average of *weights* (see VADA)

    # For instance weighting, we need to calculate the weights to use
    if FLAGS.method == "instance":
        # Get "confidence" from domain (probability it's target data)
        # classifier or task classifier (softmax probability of the
        # prediction / max value)
        if FLAGS.use_domain_confidence:
            weights = tf.sigmoid(domain_y_pred_a)

            # Alternative weighting, like Algorithm 23 of Daume's ML book
            #
            # Note: we use 1-weights since the weights above is P(s=target)
            # whereas in Alg. 23 it's P(s=source) = 1 - P(s=target).
            # Then we do 1/P(...) - 1 as in the algorithm, but clip so it's
            # not too large.
            if FLAGS.use_alt_weight:
                weights = 1/((1-weights)+epsilon) - 1
                weights = tf.clip_by_value(weights, 0, 100)
        else:
            # TODO this is *really* bad which probably indicates it has to
            # do with how I'm weighting in both this case and in pseudo-
            # labeling rather than domain vs. task confidence.
            weights = tf.reduce_max(task_y_pred_a, axis=1)

        return weights

    # If not doing instance weighting, just weight by 1
    else:
        return tf.ones_like(domain_y_pred_a)


def l2dist(x, y):
    """
    Compute the squared L2 distance between two matrices
    From: DeepJDOT L2_dist()
    """
    dist = tf.keras.backend.reshape(tf.keras.backend.sum(tf.keras.backend.square(x), 1), (-1, 1))
    dist += tf.keras.backend.reshape(tf.keras.backend.sum(tf.keras.backend.square(y), 1), (1, -1))
    dist -= 2.0*tf.keras.backend.dot(x, tf.keras.backend.transpose(y))
    return dist


def deepjdot_compute_gamma(x_a_embedding, x_b_embedding, task_y_true_a,
        task_y_pred_b):
    """ Based on DeepJDOT fit() """
    # TODO test out GPU versions, if so, install cupy-cuda100
    # Distance computation between source and target in deep layer
    C0 = cdist(x_a_embedding, x_b_embedding, metric="sqeuclidean")
    #C0 = ot.gpu.dist(x_a_embedding, x_b_embedding, metric="sqeuclidean")
    # Ground metric for the target classification loss
    C1 = cdist(task_y_true_a, task_y_pred_b, metric="sqeuclidean")
    #C1 = ot.gpu.cdist(task_y_true_a, task_y_pred_b, metric="sqeuclidean")
    # JDOT optimal coupling (gamma)
    C = FLAGS.jdot_alpha*C0 + FLAGS.jdot_tloss*C1

    # TODO maybe class balancing like they do
    gamma = ot.emd(ot.unif(x_a_embedding.shape[0]), ot.unif(x_b_embedding.shape[0]), C)

    return tf.cast(gamma, tf.float32)


@tf.function
def train_step_deepjdot(data_a, data_b, model, opt, d_opt,
        task_loss, domain_loss, lr_schedule, global_step, gamma):
    """
    DeepJDOT

    Based on: https://github.com/bbdamodaran/deepJDOT/blob/master/Deepjdot.py
    """
    x_a, y_a = data_a
    x_b, _ = data_b

    with tf.GradientTape() as tape:
        task_y_pred_a, _, x_a_embedding = model(x_a, training=True, domain="source")
        task_y_pred_b, _, x_b_embedding = model(x_b, training=True, domain="target")

        # Correct task labels (only for source domain)
        task_y_true_a = y_a

        # See: DeepJDOT classifier_cat_loss()
        source_loss = tf.keras.backend.mean(task_loss(task_y_true_a, task_y_pred_a))
        ypred_t = tf.keras.backend.log(task_y_pred_b)
        loss = -tf.keras.backend.dot(task_y_true_a, tf.keras.backend.transpose(ypred_t))

        # source loss + target loss (flipped?)
        ce_loss = FLAGS.jdot_tloss*tf.keras.backend.sum(gamma * loss) + \
            FLAGS.jdot_sloss*source_loss

        # See: DeepJDOT align_loss() - alignment loss after feature extractor
        gdist = l2dist(x_a_embedding, x_b_embedding)
        align_loss = FLAGS.jdot_alpha * tf.keras.backend.sum(gamma * gdist)

        # Total is sum?
        total_loss = ce_loss + align_loss

        # Learning rate decay
        total_loss *= lr_schedule(global_step)

    t_grad = tape.gradient(total_loss, model.trainable_variables_task)
    opt.apply_gradients(zip(t_grad, model.trainable_variables_task))

    return x_a_embedding, x_b_embedding, task_y_true_a, task_y_pred_b


@tf.function
def train_step_none(data_a, data_b, model, opt, d_opt,
        task_loss, domain_loss):
    """ Compiled no adaptation training step that we call many times """
    x_a, y_a = data_a

    with tf.GradientTape() as tape:
        task_y_pred, _, _ = model(x_a, training=True, domain="source")
        task_y_true = y_a
        loss = task_loss(task_y_true, task_y_pred, training=True)

    grad = tape.gradient(loss, model.trainable_variables_task)
    opt.apply_gradients(zip(grad, model.trainable_variables_task))


@tf.function
def pseudo_label_domain(x, model, epsilon=1e-8):
    """ Compiled step for pseudo-labeling target data based on domain classifier
    confidence that the data is source-like """
    # Run target data through model, return the predictions and probability
    # of being source data
    task_y_pred, domain_y_pred, _ = model(x, training=True, domain="target")

    # The domain classifier output is logits, so we need to pass through sigmoid
    # to get a probability before using as a weight.
    domain_prob_target = tf.sigmoid(domain_y_pred)
    domain_prob_source = 1 - domain_prob_target

    # If desired, perform weighting more like Algorithm 23 of Daume's ML book
    # Note: didn't really help, at least in testing so far.
    if FLAGS.use_alt_weight:
        domain_prob_source = 1/(domain_prob_target+epsilon) - 1

        # Clip so not too large
        domain_prob_source = tf.clip_by_value(domain_prob_source, 0, 100)

    return task_y_pred, domain_prob_source


@tf.function
def pseudo_label_task(x, model, epsilon=1e-8):
    """ Compiled step for pseudo-labeling target data based on task classifier
    confidence """
    task_y_pred, _, _ = model(x, training=True, domain="target")

    # For each prediction in the batch, the get the max (the actual prediction,
    # since the other softmax outputs are lower) and use this as the confidence.
    # For example, if three labels for example 1 we may predict [0.05, 0.05, 0.9]
    # for 0.05 probability for class 0 or 1 but 0.9 probability for class 2.
    # We say the "confidence" then is 0.9 that it's class 2. In contrast, we have
    # much less confidence if it were [0.33, 0.33, 0.34].
    task_confidence = tf.reduce_max(task_y_pred, axis=1)

    # TODO maybe normalize by number of classes?

    return task_y_pred, task_confidence


@tf.function
def train_step_target(data_b, weights, model, opt, weighted_task_loss):
    """ Compiled train step for pseudo-labeled target data """
    x, task_y_pseudo = data_b

    # Run data through model and compute loss
    with tf.GradientTape() as tape:
        task_y_pred, domain_y_pred, _ = model(x, target=True, training=True, domain="target")
        loss = weighted_task_loss(task_y_pseudo, task_y_pred, weights, training=True)

    # Only update feature extractor and target classifier
    trainable_vars = model.trainable_variables_target

    # Update model
    grad = tape.gradient(loss, trainable_vars)
    opt.apply_gradients(zip(grad, trainable_vars))


def wgan_gradient_penalty(real, fake, discriminator):
    """
    Calculating WGAN-GP gradient penalty

    See:
    https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py#L106
    """
    epsilon = tf.random.uniform(tf.shape(real))
    x_hat = epsilon*real + (1 - epsilon)*fake

    # Run x_hat through D
    disc = discriminator(x_hat, training=False)

    grad = tf.gradients(disc, x_hat)[0]
    # compute the euclidean norm by squaring ...
    grad_sqr = tf.square(grad)
    # ... summing over the rows ...
    grad_sqr_sum = tf.reduce_sum(grad_sqr,
        axis=tf.keras.backend.arange(1, len(grad.shape)))
    # ... and sqrt
    grad_l2_norm = tf.sqrt(grad_sqr_sum)
    # compute lambda*(1-||grad||)^2 still for each single sample
    gradient_penalty = tf.math.squared_difference(grad_l2_norm, 1)

    # return the mean as loss over all the batch samples
    return tf.reduce_mean(gradient_penalty)


@tf.function
def train_step_cyclegan(data_a, data_b, mapping_model, opt, loss, invert_name=None,
        task_loss=None, classify_model=None, semantic_consistency=False):
    """ Training domain mapping with CycleGAN-like setup, return data_a mapped
    to the target domain (domain B) for training a classifier on it later

    For CyCADA, set semantic_consistency=True and pass in classification model
    (includes both source and target models?)."""
    x_a, y_a = data_a
    x_b, _ = data_b

    with tf.GradientTape(persistent=True) as tape:
        gen_AtoB, gen_AtoBtoA, disc_Areal, disc_Bfake = mapping_model(x_a, "target", training=True)
        gen_BtoA, gen_BtoAtoB, disc_Breal, disc_Afake = mapping_model(x_b, "source", training=True)

        # Need a/b for both since they could be of different sizes
        zeros_a = tf.zeros_like(disc_Areal)
        zeros_b = tf.zeros_like(disc_Breal)
        ones_a = tf.ones_like(disc_Areal)
        ones_b = tf.ones_like(disc_Breal)

        # Generators should by cycle consistent
        cyc_loss = tf.reduce_mean(tf.abs(x_a - gen_AtoBtoA)) \
            + tf.reduce_mean(tf.abs(x_b - gen_BtoAtoB))

        g_loss = cyc_loss*FLAGS.map_cyc_mult

        # penalize negative weights https://stackoverflow.com/q/50158467
        # for weight in mapping_model.trainable_variables_generators:
        #     if "kernel" in weight.name:
        #         g_loss += tf.nn.l2_loss(tf.nn.relu(tf.negative(weight)))

        # For generator step, we want the discriminator to output a 1, i.e.
        # incorrect label
        # For discriminator step, the discriminator should correctly classify
        # the original real data and the generated fake data
        # Note: we're saying 0 is fake and 1 is real, i.e. D(x) = P(x == real)
        if FLAGS.cyclegan_loss == "wgan" or FLAGS.cyclegan_loss == "wgan-gp":
            g_adv = -tf.reduce_mean(disc_Bfake) - tf.reduce_mean(disc_Afake)
            d_loss = ((tf.reduce_mean(disc_Afake) - tf.reduce_mean(disc_Areal))
                + (tf.reduce_mean(disc_Bfake) - tf.reduce_mean(disc_Breal)))/2

            if FLAGS.cyclegan_loss == "wgan-gp":
                mapping_model.set_learning_phase(False)  # Don't update BN stats
                gp1 = wgan_gradient_penalty(x_a, gen_BtoA, mapping_model.source_discriminator)
                gp2 = wgan_gradient_penalty(x_b, gen_AtoB, mapping_model.target_discriminator)
                d_loss += 10*(gp1 + gp2)
        elif FLAGS.cyclegan_loss == "lsgan":
            g_adv = tf.reduce_mean(tf.math.squared_difference(disc_Bfake, 1)) \
                + tf.reduce_mean(tf.math.squared_difference(disc_Afake, 1))
            d_loss = (
                tf.reduce_mean(tf.math.squared_difference(disc_Afake, 0))
                + tf.reduce_mean(tf.math.squared_difference(disc_Areal, 1))
                + tf.reduce_mean(tf.math.squared_difference(disc_Bfake, 0))
                + tf.reduce_mean(tf.math.squared_difference(disc_Breal, 1))
            )/2
        elif FLAGS.cyclegan_loss == "gan":
            g_adv = loss(ones_b, disc_Bfake) + loss(ones_a, disc_Afake)
            # Note: divided by two, see CycleGAN paper 7.1
            d_loss = (loss(zeros_a, disc_Afake) + loss(ones_a, disc_Areal)
               + loss(zeros_b, disc_Bfake) + loss(ones_b, disc_Breal))/2

        # Semantic consistency loss
        if semantic_consistency:
            # Check consistency between true source labels and classifier on
            # mapped source to target data
            # Note: this is using the domain-invariant model. Maybe CyCADA used
            # a separate one trained just on the source data? Not sure.
            task_y_true_a = y_a
            task_y_pred_mapped_a, _, _ = classify_model(gen_AtoB, training=False, domain="target")

            # Check consistency between classifier on target data vs. classifier
            # on target data mapped back to source
            task_y_pred_b, _, _ = classify_model(x_b, training=False, domain="target")
            task_y_pred_mapped_b, _, _ = classify_model(gen_BtoA, training=False, domain="source")

            # TODO only if classifier has "reasonably low loss", i.e. in CyCADA
            # code if it's (run classify_model on x_a) less than 1.0
            g_semantic = task_loss(task_y_true_a, task_y_pred_mapped_a) \
                + task_loss(task_y_pred_b, task_y_pred_mapped_b)
        else:
            g_semantic = 0.0

        # If the true mapping is known, calculate the true loss
        if invert_name is not None:
            true_target = inversions.map_to_target[invert_name](x_a)
            true_source = inversions.map_to_source[invert_name](x_b)
            true_loss = tf.reduce_mean(tf.abs(true_target - gen_AtoB)) \
                + tf.reduce_mean(tf.abs(true_source - gen_BtoA))

        # Total loss
        #
        # Optionally minimize that true error to make sure the network is
        # capable of learning the correct mapping.
        if FLAGS.minimize_true_error and invert_name is not None:
            g_loss = true_loss
        else:
            g_loss += g_adv

            if semantic_consistency:
                g_loss += g_semantic

        # WGAN and WGAN-GP used 5 iterations, so maybe this is ~equivalent if
        # set to 5.0?
        d_loss *= FLAGS.lr_map_d_loss_mult

    g_grad = tape.gradient(g_loss, mapping_model.trainable_variables_generators)
    d_grad = tape.gradient(d_loss, mapping_model.trainable_variables_discriminators)
    del tape

    # No overlapping variables between these, so just use one optimizer
    opt.apply_gradients(zip(g_grad, mapping_model.trainable_variables_generators))
    opt.apply_gradients(zip(d_grad, mapping_model.trainable_variables_discriminators))

    # WGAN weight clipping
    if FLAGS.cyclegan_loss == "wgan":
        for weight in mapping_model.trainable_variables_discriminators:
            weight.assign(tf.clip_by_value(weight, -0.01, 0.01))

    # For plotting -- tf.function doesn't support a dictionary
    additional_loss_names = [
        "mapping/generator",
        "mapping/cycle_consistency",
        "mapping/discriminator",
        "mapping/generator_adversarial",
    ]
    additional_loss_values = [
        g_loss,
        cyc_loss,
        d_loss / FLAGS.lr_map_d_loss_mult,
        g_adv,
    ]

    if invert_name is not None:
        additional_loss_names.append("mapping/true")
        additional_loss_values.append(true_loss)

    if semantic_consistency:
        additional_loss_names.append("mapping/semantic_consistency")
        additional_loss_values.append(g_semantic)

    # Return source data mapped to target domain, so we have the labels
    return (gen_AtoB, y_a), (additional_loss_names, additional_loss_values)


@tf.function
def train_step_forecast(data_a, data_b, model, opt, loss, invert_name=None):
    """ Train mapping with forecasting loss """
    x_a, y_a = data_a
    x_b, _ = data_b

    with tf.GradientTape(persistent=True) as tape:
        # Split data so we have a bit of data at the end of each time window
        # to forecast -- use this one for computing differences, otherwise the
        # mapped values and original are of slightly different sizes, which
        # won't work
        x_a_short, forecast_Areal = model.split(x_a)
        x_b_short, forecast_Breal = model.split(x_b)

        gen_AtoB, gen_AtoBtoA, forecast_Afake, _ = model(x_a_short, "target", training=True)
        gen_BtoA, gen_BtoAtoB, forecast_Bfake, _ = model(x_b_short, "source", training=True)

        # Generators should by cycle consistent
        cyc_loss = tf.reduce_mean(tf.abs(x_a_short - gen_AtoBtoA)) \
            + tf.reduce_mean(tf.abs(x_b_short - gen_BtoAtoB))

        # Mapping might be useful for forecasting
        forecast_loss = tf.reduce_mean(tf.abs(forecast_Areal - forecast_Afake)) \
            + tf.reduce_mean(tf.abs(forecast_Breal - forecast_Bfake))

        # If the true mapping is known, calculate the true loss
        if invert_name is not None:
            true_target = model.trim(
                inversions.map_to_target[invert_name](x_a_short),
                gen_AtoB.shape[1])
            true_source = model.trim(
                inversions.map_to_source[invert_name](x_b_short),
                gen_BtoA.shape[1])
            true_loss = tf.reduce_mean(tf.abs(true_target - gen_AtoB)) \
                + tf.reduce_mean(tf.abs(true_source - gen_BtoA))

        # Total loss
        #
        # Optionally minimize that true error to make sure the network is
        # capable of learning the correct mapping.
        if FLAGS.minimize_true_error and invert_name is not None:
            g_loss = true_loss
        else:
            g_loss = cyc_loss*FLAGS.map_cyc_mult + forecast_loss

    g_grad = tape.gradient(g_loss, model.trainable_variables_generators)
    forecast_grad = tape.gradient(forecast_loss, model.trainable_variables_forecasters)
    del tape

    # No overlapping variables between these, so just use one optimizer
    opt.apply_gradients(zip(g_grad, model.trainable_variables_generators))
    opt.apply_gradients(zip(forecast_grad, model.trainable_variables_forecasters))

    # Since gen_AtoB is actually of a different size, we need to make the
    # correct-sized one for the later classifiers
    gen_AtoB = model.map_to_target(x_a)

    # For plotting -- tf.function doesn't support a dictionary
    additional_loss_names = [
        "mapping/generator",
        "mapping/cycle_consistency",
        "mapping/forecast",
    ]
    additional_loss_values = [
        g_loss,
        cyc_loss,
        forecast_loss,
    ]

    if invert_name is not None:
        additional_loss_names.append("mapping/true")
        additional_loss_values.append(true_loss)

    # Return source data mapped to target domain, so we have the labels
    return (gen_AtoB, y_a), (additional_loss_names, additional_loss_values)


def main(argv):
    # Allow running multiple at once
    set_gpu_memory(FLAGS.gpumem)

    # Figure out the log and model directory filenames
    model_dir, log_dir = get_directory_names()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # We adapt for any method other than "none", "cyclegan", or "forecast"
    adapt = FLAGS.method in ["cycada", "dann", "pseudo", "instance",
        "cyclegan_dann", "rdann", "vrada"]

    assert not (adapt and not FLAGS.task), \
        "If adapting (e.g. method=dann), must not pass --notask"

    # For adaptation, we'll be concatenating together half source and half target
    # data, so to keep the batch_size about the same, we'll cut it in half
    train_batch = FLAGS.train_batch

    if adapt and FLAGS.use_grl:
        train_batch = train_batch // 2

    # Input training data
    #
    # Note: "It is worth noting that only the training sets of the small image
    # datasets were used during training; the test sets used for reporting
    # scores only." (self-ensembling) -- so, only use *_test for evaluation.
    # However, for now we'll use 1000 random target test samples for the
    # validation dataset (as is common).
    if FLAGS.target != "":
        source_dataset, target_dataset = load_datasets.load_da(FLAGS.source,
            FLAGS.target, test=FLAGS.test, train_batch=train_batch)
        assert source_dataset.num_classes == target_dataset.num_classes, \
            "Adapting from source to target with different classes not supported"
    else:
        assert FLAGS.method not in ["cyclegan", "cycada", "forecast"], \
            "mapping methods require both source and target data"
        source_dataset, _ = load_datasets.load_da(FLAGS.source,
            None, test=FLAGS.test, train_batch=train_batch)
        target_dataset = None

    # Iterator and evaluation datasets if we have the dataset
    source_iter = iter(source_dataset.train)
    source_dataset_eval = source_dataset.test_evaluation
    target_iter = iter(target_dataset.train) \
        if target_dataset is not None else None
    target_dataset_eval = target_dataset.test_evaluation \
        if target_dataset is not None else None

    # Information about domains
    num_classes = source_dataset.num_classes

    # Loss functions
    task_loss = models.make_task_loss(adapt and FLAGS.use_grl)
    domain_loss = models.make_domain_loss(adapt)
    weighted_task_loss = models.make_weighted_loss()
    mapping_loss = models.make_mapping_loss()

    # We need to know where we are in training for the GRL lambda schedule
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Build the (mapping and/or task) models
    if FLAGS.task:
        model = models.DomainAdaptationModel(num_classes, FLAGS.model,
            global_step, FLAGS.steps, use_grl=FLAGS.use_grl)
    else:
        model = None

    # For DeepJDOT, we need to initialize gamma and get learning rate schedule
    if FLAGS.method == "deepjdot":
        gamma = tf.keras.backend.zeros((train_batch, train_batch), tf.float32)
        deepjdot_schedule = models.DeepJdotSchedule(FLAGS.steps)

    # For mapping, we need to know the source and target sizes
    # Note: first dimension is batch size, so drop that
    source_first_x, _ = next(iter(source_dataset.train))
    source_x_shape = source_first_x.shape[1:]
    if target_dataset is not None:
        target_first_x, _ = next(iter(target_dataset.train))
        target_x_shape = target_first_x.shape[1:]

    if FLAGS.method in ["cyclegan", "cycada", "cyclegan_dann"]:
        mapping_model = models.CycleGAN(source_x_shape, target_x_shape)
    elif FLAGS.method == "forecast":
        mapping_model = models.ForecastGAN(source_x_shape, target_x_shape)
    else:
        mapping_model = None

    # Optimizers
    opt = tf.keras.optimizers.Adam(FLAGS.lr)
    d_opt = tf.keras.optimizers.Adam(FLAGS.lr*FLAGS.lr_domain_mult)
    if FLAGS.cyclegan_loss == "wgan":
        mapping_opt = tf.keras.optimizers.RMSprop(FLAGS.lr*FLAGS.lr_mapping_mult)
    else:
        mapping_opt = tf.keras.optimizers.Adam(FLAGS.lr*FLAGS.lr_mapping_mult)

    # For GAN-like training (train_step_gan), we'll weight by the GRL schedule
    # to make it more equivalent to when use_grl=True.
    grl_schedule = models.DannGrlSchedule(FLAGS.steps)

    # Target classifier optimizer if target_classifier, otherwise the optimizer
    # for the task-classifier when running on pseudo-labeled data
    has_target_classifier = FLAGS.method in ["pseudo", "instance"]
    t_opt = tf.keras.optimizers.Adam(FLAGS.lr*FLAGS.lr_target_mult)

    # Checkpoints -- but can't handle None in checkpoint
    checkpoint = {
        "global_step": global_step,
        "opt": opt,
        "d_opt": d_opt,
        "t_opt": t_opt,
        "mapping_opt": mapping_opt,
    }

    if model is not None:
        checkpoint["model"] = model
    if mapping_model is not None:
        checkpoint["mapping_model"] = mapping_model

    checkpoint = tf.train.Checkpoint(**checkpoint)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir,
        target=has_target_classifier)
    checkpoint_manager.restore_latest()

    # Metrics
    has_target_domain = target_dataset is not None
    metrics = Metrics(log_dir, source_dataset,
        task_loss, domain_loss, has_target_domain, has_target_classifier,
        enable_compile=FLAGS.compile_metrics)

    # Pretrain FE+TaskC on source data (optional, by default skip this)
    #
    # Note: we don't save the model till after pretraining, so if it's killed
    # during this, it'll restart pretraining. I'm assuming pretraining will
    # be quick.
    if int(global_step) == 0 and model is not None:
        for i in range(FLAGS.pretrain_steps):
            t = time.time()
            data_a = next(source_iter)
            train_step_none(data_a, None, model, opt, d_opt, task_loss, domain_loss)
            t = time.time() - t

            if i%100 == 0:
                logging.info("pretrain step %d took %f seconds", i, t)

    # Start training
    for i in range(int(global_step), FLAGS.steps+1):
        # Get data for this iteration
        data_a = next(source_iter)
        orig_data_a = data_a  # needed for evaluation
        data_b = next(target_iter) if target_iter is not None else None

        t = time.time()

        # The GAN performing domain mapping, if desired
        additional_losses = None

        if mapping_model is not None:
            # Trains GAN to map source data_a to look like target data and
            # returns the mapped data so we can train a classifier (below
            # as usual) on the now-labeled target-like data. data_b stays
            # the same and may be used (if method != none) to further adapt
            # the classifier to have a fake target vs. real target invariant
            # representation.
            if FLAGS.method == "forecast":
                data_a, additional_losses = train_step_forecast(data_a, data_b,
                    mapping_model, mapping_opt, mapping_loss,
                    source_dataset.invert_name)
            else:
                data_a, additional_losses = train_step_cyclegan(data_a, data_b,
                    mapping_model, mapping_opt, mapping_loss,
                    source_dataset.invert_name, task_loss, model,
                    semantic_consistency=FLAGS.method == "cycada")

        # Train the task model (possibly with adaptation)
        if model is not None:
            # The feature extractor, classifiers, etc.
            step_args = (data_a, data_b, model, opt, d_opt, task_loss, domain_loss)

            if FLAGS.method == "deepjdot":
                # Train network
                x_a_embedding, x_b_embedding, task_y_true_a, task_y_pred_b = \
                    train_step_deepjdot(*step_args, deepjdot_schedule, global_step, gamma)
                # Compute new coupling
                gamma = deepjdot_compute_gamma(x_a_embedding, x_b_embedding,
                    task_y_true_a, task_y_pred_b)
            elif adapt and FLAGS.use_grl:
                train_step_grl(*step_args)
            elif adapt:
                instance_weights = train_step_gan(*step_args, grl_schedule, global_step)
            else:
                train_step_none(*step_args)

            if FLAGS.method == "pseudo":
                # We'll ignore the real labels, so just get the data
                x, _ = data_b

                # Pseudo-label target data
                if FLAGS.use_domain_confidence:
                    task_y_pred, weights = pseudo_label_domain(x, model)
                else:
                    task_y_pred, weights = pseudo_label_task(x, model)

                # Create new data with same input by pseudo-labels not true labels
                data_b_pseudo = (x, task_y_pred)

                # Train target classifier on pseudo-labeled data, weighted
                # by probability that it's source data (i.e. higher confidence)
                train_step_target(data_b_pseudo, weights, model,
                    t_opt, weighted_task_loss)
            elif FLAGS.method == "instance":
                # Train target classifier on source data, but weighted
                # by probability that it's target data
                train_step_target(data_a, instance_weights, model,
                    t_opt, weighted_task_loss)

        global_step.assign_add(1)
        t = time.time() - t

        if i%100 == 0:
            logging.info("step %d took %f seconds", int(global_step), t)

        # Metrics on training/validation data
        if i%FLAGS.log_train_steps == 0:
            # Note: orig_data_a is not mapped whereas data_a is (assuming)
            # we have a mapping model. If we are evaluating the RMSE of the
            # mapping when we have an invertible synthetic mapping, we need
            # the original data. Otherwise, we can just use the mapped data.
            metrics.train(model, mapping_model, orig_data_a, data_a, data_b,
                global_step, t, additional_losses)

        # Evaluate every log_val_steps but also at the last step
        validation_accuracy = None
        target_validation_accuracy = None
        if i%FLAGS.log_val_steps == 0 or i == FLAGS.steps:
            validation_accuracy, target_validation_accuracy = metrics.test(
                model, mapping_model, source_dataset_eval, target_dataset_eval,
                global_step)

        # Checkpoints -- Save either if at the right model step or if we found
        # a new validation accuracy. If this is better than the previous best
        # model, we need to make a new checkpoint so we can restore from this
        # step with the best accuracy.
        if i%FLAGS.model_steps == 0 or validation_accuracy is not None:
            checkpoint_manager.save(int(global_step-1), validation_accuracy,
                target_validation_accuracy)

        # Plots
        if i%FLAGS.log_plots_steps == 0:
            metrics.plots(model, mapping_model,
                source_dataset_eval, target_dataset_eval, adapt, global_step)

    # We're done -- used for hyperparameter tuning
    write_finished(log_dir)


if __name__ == "__main__":
    app.run(main)
