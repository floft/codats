"""
Methods
"""
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from absl import flags

import models
import load_datasets

from class_balance import class_balance

FLAGS = flags.FLAGS

flags.DEFINE_float("lr", 0.0001, "Learning rate for training")
flags.DEFINE_float("lr_domain_mult", 1.0, "Learning rate multiplier for training domain classifier")
flags.DEFINE_float("hda_l2", 1000.0, "Weight for regularizing each domain's feature extractor weights to be similar")
flags.DEFINE_boolean("hda_by_layer", False, "Regularize lower layers less and higher layers more, only matters if hda_l2 != 0")
flags.DEFINE_boolean("ensemble_same_data", False, "Train each model on the same batch of data, or if false use a different random batch for each model")

methods = {}


def register_method(name):
    """ Add method to the list of methods, e.g. add @register_method("name")
    before a class definition """
    assert name not in methods, "duplicate method named " + name

    def decorator(cls):
        methods[name] = cls
        return cls

    return decorator


def get_method(name, *args, **kwargs):
    """ Based on the given name, call the correct method """
    assert name in methods.keys(), \
        "Unknown method name " + name
    return methods[name](*args, **kwargs)


def list_methods():
    """ Returns list of all the available methods """
    return list(methods.keys())


class MethodBase:
    def __init__(self, source_datasets, target_dataset, model_name,
            *args, ensemble_size=1, trainable=True, moving_average=False,
            share_most_weights=False, **kwargs):
        self.source_datasets = source_datasets
        self.target_dataset = target_dataset
        self.moving_average = moving_average
        self.ensemble_size = ensemble_size
        assert ensemble_size > 0, "ensemble_size should be >= 1"
        self.share_most_weights = share_most_weights  # for HeterogeneousBase

        # Support multiple targets when we add that functionality
        self.num_source_domains = len(source_datasets)
        self.num_domains = len(source_datasets)

        if target_dataset is not None:
            if isinstance(target_dataset, list):
                self.num_domains += len(target_dataset)
            elif isinstance(target_dataset, load_datasets.Dataset):
                self.num_domains += 1
            else:
                raise NotImplementedError("target_dataset should be either one "
                    "load_datasets.Dataset() or a list of them, "
                    "but is "+str(target_dataset))

        # How to calculate the number of domain outputs
        self.domain_outputs = self.calculate_domain_outputs()

        # We need to know the num_classes for creating the model
        # We'll just pick the first source since we have to have at least one
        # source and we've already verified they're all the same in load_da()
        self.num_classes = source_datasets[0].num_classes

        # What we want in the checkpoint
        self.checkpoint_variables = {}

        # Initialize components -- support ensemble, training all simultaneously
        # I think will be faster / more efficient overall time-wise
        self.create_iterators()
        self.opt = [self.create_optimizers() for _ in range(ensemble_size)]
        self.model = [self.create_model(model_name) for _ in range(ensemble_size)]
        self.create_losses()

        # Checkpoint/save the model and optimizers
        for i, model in enumerate(self.model):
            self.checkpoint_variables["model_" + str(i)] = model

        for i, opt_dict in enumerate(self.opt):
            for name, opt in opt_dict.items():
                self.checkpoint_variables["opt_" + name + "_" + str(i)] = opt

        # Names of the losses returned in compute_losses
        self.loss_names = ["total"]

        # Should this method be trained (if not, then in main.py the config
        # is written and then it exits)
        self.trainable = trainable

    def calculate_domain_outputs(self):
        """ Calculate the number of outputs for the domain classifier. By
        default it's the number of domains. However, for example, in domain
        generalization we ignore the target, so it'll actually be the number of
        source domains only, in which case override this function. """
        return self.num_domains

    def create_iterators(self):
        """ Get the source/target train/eval datasets """
        self.source_train_iterators = [iter(x.train) for x in self.source_datasets]
        self.source_train_eval_datasets = [x.train_evaluation for x in self.source_datasets]
        self.source_test_eval_datasets = [x.test_evaluation for x in self.source_datasets]

        if self.target_dataset is not None:
            self.target_train_iterator = iter(self.target_dataset.train)
            self.target_train_eval_dataset = self.target_dataset.train_evaluation
            self.target_test_eval_dataset = self.target_dataset.test_evaluation
        else:
            self.target_train_iterator = None
            self.target_train_eval_dataset = None
            self.target_test_eval_dataset = None

    def create_optimizer(self, *args, **kwargs):
        """ Create a single optimizer """
        opt = tf.keras.optimizers.Adam(*args, **kwargs)

        if self.moving_average:
            opt = tfa.optimizers.MovingAverage(opt)

        return opt

    def create_optimizers(self):
        return {"opt": self.create_optimizer(learning_rate=FLAGS.lr)}

    def create_model(self, model_name):
        return models.BasicModel(self.num_classes, self.domain_outputs,
            model_name=model_name)

    def create_losses(self):
        self.task_loss = make_loss()

    def get_next_train_data(self):
        """ Get next batch of training data """
        # Note we will use this same exact data in Metrics() as we use in
        # train_step()
        data_sources = [next(x) for x in self.source_train_iterators]
        data_target = next(self.target_train_iterator) \
            if self.target_train_iterator is not None else None
        return self.get_next_batch_both(data_sources, data_target)

    def domain_label(self, index, is_target):
        """ Default domain labeling. Indexes should be in [0,+inf) and integers.
        0 = target
        1 = source #0
        2 = source #1
        3 = source #2
        ...
        """
        if is_target:
            return 0
        else:
            return index+1

    @tf.function
    def get_next_batch_both(self, data_sources, data_target):
        """ Compile for training. Don't for evaluation (called directly,
        not this _both function). """
        data_sources = self.get_next_batch_multiple(data_sources, is_target=False)
        data_target = self.get_next_batch_single(data_target, is_target=True)
        return data_sources, data_target

    def get_next_batch_multiple(self, data, is_target):
        """
        Get next set of training data. data should be a list of data (probably
        something like [next(x) for x in iterators]).

        Returns: (
            [x_a1, x_a2, x_a3, ...],
            [y_a1, y_a2, y_a3, ...],
            [domain_a1, domain_a2, domain_a3, ...]
        )
        """
        if data is None:
            return None

        assert not is_target or len(data) == 1, \
            "only support one target at present"

        xs = []
        ys = []
        ds = []

        for i, (x, y) in enumerate(data):
            xs.append(x)
            ys.append(y)
            ds.append(tf.ones_like(y)*self.domain_label(index=i,
                is_target=is_target))

        return (xs, ys, ds)

    def get_next_batch_single(self, data, is_target, index=0):
        """
        Get next set of training data. data should be a single batch (probably
        something like next(iterator)). When processing target data, index
        must be 0 since we only support one target at the moment. However,
        during evaluation we evaluate each source's data individually so if
        is_target is False, then index can be whichever source domain was
        passed.

        Returns: (x, y, domain)
        """
        if data is None:
            return None

        assert not is_target or index == 0, \
            "only support one target at present"

        x, y = data
        d = tf.ones_like(y)*self.domain_label(index=index, is_target=is_target)
        data_target = (x, y, d)

        return data_target

    # Allow easily overriding each part of the train_step() function, without
    # having to override train_step() in its entirety
    def prepare_data(self, data_sources, data_target):
        """ Prepare the data for the model, e.g. by concatenating all sources
        together. Note: do not put code in here that changes the domain labels
        since you presumably want that during evaluation too. Put that in
        domain_label() """
        # By default (e.g. for no adaptation or domain generalization), ignore
        # the target data
        x_a, y_a, domain_a = data_sources
        x = tf.concat(x_a, axis=0)
        task_y_true = tf.concat(y_a, axis=0)
        domain_y_true = tf.concat(domain_a, axis=0)
        return x, task_y_true, domain_y_true

    def prepare_data_eval(self, data, is_target):
        """ Prepare the data for the model, e.g. by concatenating all sources
        together. This is like prepare_data() but use during evaluation. """
        x, y, domain = data

        assert isinstance(x, list), \
            "Must pass x=[...] even if only one domain for tf.function consistency"
        assert isinstance(y, list), \
            "Must pass y=[...] even if only one domain for tf.function consistency"
        assert isinstance(domain, list), \
            "Must pass domain=[...] even if only one domain for tf.function consistency"

        # Concatenate all the data (e.g. if multiple source domains)
        x = tf.concat(x, axis=0)
        y = tf.concat(y, axis=0)
        domain = tf.concat(domain, axis=0)

        return x, y, domain

    def post_data_eval(self, task_y_true, task_y_pred, domain_y_true,
            domain_y_pred):
        """ Optionally do something with the data after feeding through the
        model. Since the model outputs logits, here we actually take the softmax
        so that during evaluation we have probability distributions. """
        task_y_pred = tf.nn.softmax(task_y_pred)
        domain_y_pred = tf.nn.softmax(domain_y_pred)
        return task_y_true, task_y_pred, domain_y_true, domain_y_pred

    def call_model(self, x, which_model, is_target=None, **kwargs):
        return self.model[which_model](x, **kwargs)

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, which_model, training):
        # Maybe: regularization = sum(model.losses) and add to loss
        return self.task_loss(task_y_true, task_y_pred)

    def compute_gradients(self, tape, loss, which_model):
        return tape.gradient(loss,
            self.model[which_model].trainable_variables_task_fe)

    def apply_gradients(self, grad, which_model):
        self.opt[which_model]["opt"].apply_gradients(zip(grad,
            self.model[which_model].trainable_variables_task_fe))

    def train_step(self):
        """
        Get batch of data, prepare data, run through model, compute losses,
        apply the gradients

        Override the individual parts with prepare_data(), call_model(),
        compute_losses(), compute_gradients(), and apply_gradients()

        We return the batch of data so we can use the exact same training batch
        for the "train" evaluation metrics.
        """
        # TensorFlow errors constructing the graph (with tf.function, which
        # makes training faster) if we don't know the data size. Thus, first
        # load batches, then pass to compiled train step.
        all_data_sources = []
        all_data_target = []

        for i in range(self.ensemble_size):
            data_sources, data_target = self.get_next_train_data()
            all_data_sources.append(data_sources)
            all_data_target.append(data_target)

            # If desired, use the same batch for each of the models.
            if FLAGS.ensemble_same_data:
                break

        self._train_step(all_data_sources, all_data_target)

        # We return the first one since we don't really care about the "train"
        # evaluation metrics that much.
        return all_data_sources[0], all_data_target[0]

    @tf.function
    def _train_step(self, all_data_sources, all_data_target):
        """ The compiled part of train_step. We can't compile everything since
        some parts of the model need to know the shape of the data apparently.

        The first batch is passed in because to compile this, TF needs to know
        the shape. Doesn't look pretty... but it runs...
        """
        for i in range(self.ensemble_size):
            # Get random batch for this model in the ensemble (either same for
            # all or different for each)
            if FLAGS.ensemble_same_data:
                data_sources = all_data_sources[0]
                data_target = all_data_target[0]
            else:
                data_sources = all_data_sources[i]
                data_target = all_data_target[i]

            # Prepare
            x, task_y_true, domain_y_true = self.prepare_data(data_sources,
                data_target)

            # Run batch through the model and compute loss
            with tf.GradientTape(persistent=True) as tape:
                task_y_pred, domain_y_pred, fe_output = self.call_model(
                    x, which_model=i, training=True)
                losses = self.compute_losses(x, task_y_true, domain_y_true,
                    task_y_pred, domain_y_pred, fe_output, which_model=i,
                    training=True)

            # Update model
            gradients = self.compute_gradients(tape, losses, which_model=i)
            del tape
            self.apply_gradients(gradients, which_model=i)

    def eval_step(self, data, is_target):
        """ Evaluate a batch of source or target data, called in metrics.py.
        This preprocesses the data to have x, y, domain always be lists so
        we can use the same compiled tf.function code in eval_step_list() for
        both sources and target domains. """
        x, y, domain = data

        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        if not isinstance(domain, list):
            domain = [domain]

        return self.eval_step_list((x, y, domain), is_target)

    def add_multiple_losses(self, losses, average=False):
        """
        losses = [
            [total_loss1, task_loss1, ...],
            [total_loss2, task_loss2, ...],
            ...
        ]

        returns [total_loss, task_loss, ...] either the sum or average
        """
        losses_added = None

        for loss_list in losses:
            # If no losses yet, then just set to this
            if losses_added is None:
                losses_added = loss_list
            # Otherwise, add to the previous loss values
            else:
                assert len(losses_added) == len(loss_list), \
                    "subsequent losses have different length than the first"

                for i, loss in enumerate(loss_list):
                    losses_added[i] += loss

        assert losses_added is not None, \
            "must return losses from at least one domain"

        if average:
            averaged_losses = []

            for loss in losses_added:
                averaged_losses.append(loss / len(losses))

            return averaged_losses
        else:
            return losses_added

    #@tf.function  # faster not to compile
    def eval_step_list(self, data, is_target):
        """ Override preparation in prepare_data_eval() """
        x, orig_task_y_true, orig_domain_y_true = self.prepare_data_eval(data,
            is_target)

        task_y_true_list = []
        task_y_pred_list = []
        domain_y_true_list = []
        domain_y_pred_list = []
        losses_list = []

        for i in range(self.ensemble_size):
            # Run through model
            task_y_pred, domain_y_pred, fe_output = self.call_model(x,
                which_model=i, is_target=is_target, training=False)

            # Calculate losses
            losses = self.compute_losses(x, orig_task_y_true,
                orig_domain_y_true, task_y_pred, domain_y_pred, fe_output,
                which_model=i, training=False)

            if not isinstance(losses, list):
                losses = [losses]

            losses_list.append(losses)

            # Post-process data (e.g. compute softmax from logits)
            task_y_true, task_y_pred, domain_y_true, domain_y_pred = \
                self.post_data_eval(orig_task_y_true, task_y_pred,
                    orig_domain_y_true, domain_y_pred)

            task_y_true_list.append(task_y_true)
            task_y_pred_list.append(task_y_pred)
            domain_y_true_list.append(domain_y_true)
            domain_y_pred_list.append(domain_y_pred)

        # Combine information from each model in the ensemble -- averaging.
        #
        # Note: this is how the ensemble predictions are made with InceptionTime
        # having an ensemble of 5 models -- they average the softmax outputs
        # over the ensemble (and we now have softmax after the post_data_eval()
        # call). See their code:
        # https://github.com/hfawaz/InceptionTime/blob/master/classifiers/nne.py
        task_y_true_avg = tf.math.reduce_mean(task_y_true_list, axis=0)
        task_y_pred_avg = tf.math.reduce_mean(task_y_pred_list, axis=0)
        domain_y_true_avg = tf.math.reduce_mean(domain_y_true_list, axis=0)
        domain_y_pred_avg = tf.math.reduce_mean(domain_y_pred_list, axis=0)
        losses_avg = self.add_multiple_losses(losses_list, average=True)

        return task_y_true_avg, task_y_pred_avg, domain_y_true_avg, \
            domain_y_pred_avg, losses_avg


#
# Homogeneous domain adaptation
#

# The base method class performs no adaptation
@register_method("none")
class MethodNone(MethodBase):
    pass


# with the model architecture and multi-source support = CoDATS
@register_method("dann")
class MethodDann(MethodBase):
    def __init__(self, source_datasets, target_dataset,
            global_step, total_steps, *args, **kwargs):
        self.global_step = global_step  # should be TF variable
        self.total_steps = total_steps
        super().__init__(source_datasets, target_dataset, *args, **kwargs)
        self.loss_names += ["task", "domain"]

    def create_model(self, model_name):
        return models.DannModel(self.num_classes, self.domain_outputs,
            self.global_step, self.total_steps, model_name=model_name)

    def create_optimizers(self):
        opt = super().create_optimizers()
        # We need an additional optimizer for DANN
        opt["d_opt"] = self.create_optimizer(
            learning_rate=FLAGS.lr*FLAGS.lr_domain_mult)
        return opt

    def create_losses(self):
        # Note: at the moment these are the same, but if we go back to
        # single-source, then the domain classifier may be sigmoid not softmax
        super().create_losses()
        self.domain_loss = make_loss()

    def prepare_data(self, data_sources, data_target):
        assert data_target is not None, "cannot run DANN without target"
        x_a, y_a, domain_a = data_sources
        x_b, y_b, domain_b = data_target

        # Concatenate all source domains' data
        x_a = tf.concat(x_a, axis=0)
        y_a = tf.concat(y_a, axis=0)
        domain_a = tf.concat(domain_a, axis=0)

        # Concatenate for adaptation - concatenate source labels with all-zero
        # labels for target since we can't use the target labels during
        # unsupervised domain adaptation
        x = tf.concat((x_a, x_b), axis=0)
        task_y_true = tf.concat((y_a, tf.zeros_like(y_b)), axis=0)
        domain_y_true = tf.concat((domain_a, domain_b), axis=0)

        return x, task_y_true, domain_y_true

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, which_model, training):
        nontarget = tf.where(tf.not_equal(domain_y_true, 0))
        task_y_true = tf.gather(task_y_true, nontarget)
        task_y_pred = tf.gather(task_y_pred, nontarget)

        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = self.domain_loss(domain_y_true, domain_y_pred)
        total_loss = task_loss + d_loss
        return [total_loss, task_loss, d_loss]

    def compute_gradients(self, tape, losses, which_model):
        total_loss, task_loss, d_loss = losses
        grad = tape.gradient(total_loss,
            self.model[which_model].trainable_variables_task_fe_domain)
        d_grad = tape.gradient(d_loss,
            self.model[which_model].trainable_variables_domain)
        return [grad, d_grad]

    def apply_gradients(self, gradients, which_model):
        grad, d_grad = gradients
        self.opt[which_model]["opt"].apply_gradients(zip(grad,
            self.model[which_model].trainable_variables_task_fe_domain))
        # Update discriminator again
        self.opt[which_model]["d_opt"].apply_gradients(zip(d_grad,
            self.model[which_model].trainable_variables_domain))


@register_method("dann_gs")
class MethodDannGS(MethodDann):
    """ Same as DANN but only has 2 domains, any source is domain 1 (i.e. group
    them) and the target is still domain 0 """
    def calculate_domain_outputs(self):
        assert self.num_domains > 1, "cannot do GS-DANN with only 1 domain"
        return 2

    def domain_label(self, index, is_target):
        """
        Replace all source domains' domain labels with 1, i.e. group all
        sources together

        New domain labeling:
        0 = target
        1 = source #0
        1 = source #1
        1 = source #2
        ...
        """
        if is_target:
            return 0
        else:
            return 1


@register_method("dann_smooth")
class MethodDannSmooth(MethodDannGS):
    """ MDAN Smooth method based on MethodDannGS since we want binary source = 1,
    target = 0 for the domain labels, very similar to HeterogeneousBase()
    code except this has multiple DC's not multiple FE's  """
    def create_model(self, model_name):
        return models.DannSmoothModel(
            self.num_classes, self.domain_outputs,  # Note: domain_outputs=2
            self.global_step, self.total_steps,
            model_name=model_name,
            num_domain_classifiers=self.num_source_domains)

    def prepare_data(self, data_sources, data_target):
        """ Prepare a batch of source i with target data to run through each
        # of the domain classifiers """
        assert data_target is not None, "cannot run Smooth without target"
        x_a, y_a, domain_a = data_sources
        x_b, y_b, domain_b = data_target

        x = []
        task_y_true = []
        domain_y_true = []

        for i in range(len(self.source_datasets)):
            x.append(tf.concat((x_a[i], x_b), axis=0))
            task_y_true.append(tf.concat((y_a[i], tf.zeros_like(y_b)), axis=0))
            # Each domain classifier is binary, it's 0 = target, 1 = source
            # Note: we do zeros_like for consistency, but domain_b is already
            # zeros
            domain_y_true.append(tf.concat((
                tf.ones_like(domain_a[i]), tf.zeros_like(domain_b)), axis=0))

        return x, task_y_true, domain_y_true

    def prepare_data_eval(self, data, is_target):
        """ Don't concatenate elements of the list like in the base class since
        we want to handle all the source domains separately, to pass to the
        right domain classifiers.

        Note: during evaluation, target data will be fed through the first
        domain classifier. Any would work since they're all trained with half
        (some) source domain and half target domain data. """
        x, y, domain = data

        assert isinstance(x, list), \
            "Must pass x=[...] even if only one domain for tf.function consistency"
        assert isinstance(y, list), \
            "Must pass y=[...] even if only one domain for tf.function consistency"
        assert isinstance(domain, list), \
            "Must pass domain=[...] even if only one domain for tf.function consistency"

        return x, y, domain

    def call_model(self, x, which_model, is_target=None, **kwargs):
        """ Run each source-target pair through model separately, using the
        corresponding domain classifier. """
        task_y_pred = []
        domain_y_pred = []
        fe_output = []

        # For each source domain (or during evaluation, for the target domain,
        # which will be a list of size 1 here)
        assert len(x) <= len(self.source_datasets), \
            "Too many domains fed to model, should be <= number of source domains"

        for i in range(len(x)):
            i_task_y_pred, i_domain_y_pred, i_fe_output = \
                self.model[which_model](x[i], which_dc=i, **kwargs)
            task_y_pred.append(i_task_y_pred)
            domain_y_pred.append(i_domain_y_pred)
            fe_output.append(i_fe_output)

        return task_y_pred, domain_y_pred, fe_output

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, which_model, training):
        """
        MDAN losses - domain classifiers' losses weighted by task
        classifier's loss per domain
        https://github.com/KeiraZhao/MDAN/blob/master/model.py
        https://github.com/KeiraZhao/MDAN/blob/master/main_amazon.py
        """
        task_losses = []
        domain_losses = []

        # For each source domain (or during evaluation, for the target domain,
        # which will be a list of size 1 here)
        assert len(task_y_true) <= len(self.source_datasets), \
            "Too many domains fed to model, should be <= number of source domains"

        for i in range(len(task_y_true)):
            # For task loss, ignore target data
            nontarget = tf.where(tf.not_equal(domain_y_true[i], 0))
            i_task_y_true = tf.gather(task_y_true[i], nontarget)
            i_task_y_pred = tf.gather(task_y_pred[i], nontarget)

            # Their code does nll_loss(log_softmax(...)) which should be
            # equivalent to cross entropy
            task_losses.append(self.task_loss(i_task_y_true, i_task_y_pred))
            domain_losses.append(self.domain_loss(domain_y_true[i], domain_y_pred[i]))

        # Defaults were 10.0 and 1e-2, which appear to do about the same as the
        # other methods. Seems like setting them to 1 does better though in some
        # initial tests.
        gamma = 1.0
        mu = 1.0

        return tf.math.log(tf.reduce_sum(
            tf.exp(gamma*(task_losses+tf.multiply(mu, domain_losses)))))/gamma

    def post_data_eval(self, task_y_true, task_y_pred, domain_y_true,
            domain_y_pred):
        """ After computing the losses, for evaluation we need to concatenate
        the data from all domains """
        task_y_true = tf.concat(task_y_true, axis=0)
        task_y_pred = tf.concat(task_y_pred, axis=0)
        domain_y_true = tf.concat(domain_y_true, axis=0)
        domain_y_pred = tf.concat(domain_y_pred, axis=0)
        return super().post_data_eval(task_y_true, task_y_pred, domain_y_true,
            domain_y_pred)

    def compute_gradients(self, tape, losses, which_model):
        """ We have one loss, update everything with it """
        return tape.gradient(losses,
            self.model[which_model].trainable_variables_task_fe_domain)

    def apply_gradients(self, gradients, which_model):
        self.opt[which_model]["opt"].apply_gradients(zip(gradients,
            self.model[which_model].trainable_variables_task_fe_domain))


@register_method("rdann")
class MethodRDann(MethodDann):
    """ Same as DANN but uses a different model -- LSTM with some dense layers """
    def create_model(self, model_name):
        return models.RDannModel(self.num_classes, self.domain_outputs,
            self.global_step, self.total_steps, model_name=model_name)


@register_method("vrada")
class MethodVrada(MethodDann):
    """ DANN but with the VRADA model and VRNN loss """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_names += ["vrnn"]

    def create_model(self, model_name):
        return models.VradaModel(self.num_classes, self.domain_outputs,
            self.global_step, self.total_steps, model_name=model_name)

    def compute_vrnn_loss(self, vrnn_state, x, epsilon=1e-9):
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

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, which_model, training):
        _, task_loss, d_loss = super().compute_losses(
            x, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, which_model, training)
        vrnn_state = fe_output[1]  # fe_output = (vrnn_output, vrnn_state)
        vrnn_loss = self.compute_vrnn_loss(vrnn_state, x)
        total_loss = task_loss + d_loss + vrnn_loss
        return [total_loss, task_loss, d_loss, vrnn_loss]

    def compute_gradients(self, tape, losses, which_model):
        # We only use vrnn_loss for plotting -- for computing gradients it's
        # included in the total loss
        total_loss, task_loss, d_loss, _ = losses
        return super().compute_gradients(tape, [total_loss, task_loss, d_loss],
            which_model)


@register_method("daws")
class MethodDaws(MethodDann):
    """ Domain adaptation with weak supervision (in this case, target-domain
    label proportions)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_names += ["weak"]
        self.compute_p_y()
        # Used in loss
        self.grl_schedule = models.DannGrlSchedule(self.total_steps)

    def compute_p_y(self):
        """ Compute P(y) (i.e. class balance) of the training target dataset

        Note: we simulate the self-report label proportions from looking at
        the target training labels (not validation or test sets). However, after
        this function call, we don't use the labels themselves (outside of
        computing evaluation accuracy), just the computed proportions for the
        training.
        """
        # Compute proportion of each class
        # Note: we use the "eval" train dataset since it doesn't repeat infinitely
        # and we use "train" not test since we don't want to peak at the
        # validation data we use for model selection.
        self.p_y = class_balance(self.target_train_eval_dataset, self.num_classes)

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, which_model, training):
        # DANN losses
        nontarget = tf.where(tf.not_equal(domain_y_true, 0))
        task_y_true_nontarget = tf.gather(task_y_true, nontarget)
        task_y_pred_nontarget = tf.gather(task_y_pred, nontarget)

        task_loss = self.task_loss(task_y_true_nontarget, task_y_pred_nontarget)
        d_loss = self.domain_loss(domain_y_true, domain_y_pred)

        # DA-WS regularizer
        #
        # Get predicted target-domain labels. We ignore label proportions for
        # the source domains since we train to predict the correct labels there.
        # We don't know the target-domain labels, so instead we try using this
        # additional P(y) label proportion information. Thus, we use it and the
        # adversarial domain-invariant FE objectives as sort of auxiliary
        # losses.
        target = tf.where(tf.equal(domain_y_true, 0))
        task_y_pred_target = tf.gather(task_y_pred, target)

        # Idea:
        # argmax, one-hot, reduce_sum(..., axis=1), /= batch_size, KL with p_y
        # However, argmax yields essentially useless gradients (as far as I
        # understand it, e.g. we use cross entropy loss for classification not
        # the actual 0-1 loss or loss on the argmax of the softmax outputs)
        #
        # Thus, a soft version. Idea: softmax each, reduce sum vertically,
        #   /= batch_size, then KL
        # This is different than per-example-in-batch KLD because we average
        # over the softmax outputs across the batch before KLD. So, the
        # difference is whether averaging before or after KLD.
        #
        # Note: this depends on a large enough batch size. If you can't set it
        # >=64 or so (like what we use in SS-DA for the target data, i.e. half
        # the 128 batch size), then accumulate this gradient over multiple steps
        # and then apply.
        #
        # cast batch_size to float otherwise:
        # "x and y must have the same dtype, got tf.float32 != tf.int32"
        batch_size = tf.cast(tf.shape(task_y_pred_target)[0], dtype=tf.float32)
        p_y_batch = tf.reduce_sum(tf.nn.softmax(task_y_pred_target), axis=0) / batch_size
        daws_loss = tf.keras.losses.KLD(self.p_y, p_y_batch)

        # Sum up individual losses for the total
        #
        # Note: daws_loss doesn't have the DANN learning rate schedule because
        # it goes with the task_loss. We want to learn predictions for the task
        # classifier that both correctly predicts labels on the source data and
        # on the target data aligns with the correct label proportions.
        # Separately, we want the FE representation to also be domain invariant,
        # which we apply the learning rate schedule to, I think, to help the
        # adversarial part converge properly (recall GAN training instability
        # stuff).
        total_loss = task_loss + d_loss + daws_loss

        return [total_loss, task_loss, d_loss, daws_loss]

    def compute_gradients(self, tape, losses, which_model):
        # We only use daws_loss for plotting -- for computing gradients it's
        # included in the total loss
        return super().compute_gradients(tape, losses[:-1], which_model)


#
# Heterogeneous domain adaptation
#

class HeterogeneousBase:
    """ Handle multiple feature extractors, very similar to MethodDannSmooth()
    code except this has multiple FE's not multiple DC's """
    def __init__(self, *args, **kwargs):
        # Otherwise, with multiple inheritance, the other init's aren't called.
        super().__init__(*args, **kwargs)

        # For regularizing the weights to be similar
        self.regularizer = tf.keras.regularizers.L1L2(l2=FLAGS.hda_l2)
        self.loss_names += ["fe_regularizer"]

    def regularize_fe_weights_similar(self, which_model):
        """
        Regularize the target feature extractor to be similar to each
        (probably only 1) source feature extractors. We assume the last
        FE is the target (see ordering in prepare_data).
        """
        # We should now have multiple feature extractors
        assert len(self.model[which_model].feature_extractor) > 1, \
            "for HDA must have >= 2 FE's"

        regularizer_loss = 0
        total_weights = 0

        target_fe = self.model[which_model].feature_extractor[-1]
        target_vars = target_fe.trainable_variables
        num_vars = len(target_vars)

        for source_fe in self.model[which_model].feature_extractor[:-1]:
            source_vars = source_fe.trainable_variables
            assert len(source_vars) == num_vars, \
                "FE's must have the same number of weights"

            # Regularize to be similar, i.e. the difference toward zero
            for i, (source_weight, target_weight) in \
                    enumerate(zip(source_vars, target_vars)):
                # Skip the BN weights since we expect those might be different
                # between source/target. For example, some DA methods (see
                # survey) rely *entirely* on different BN weights per-domain.
                #
                # Note: alternative is skip all but those with "kernel" in their
                # name.
                if "batch_normalization" in source_weight.name:
                    continue

                # If they're not the same shape, then we can't do this.
                # For example, if this is the first layer and the source/target
                # input feature shapes differ. Or, if this is not the FCN but
                # InceptionTime model, then the shortcuts also differ in shape.
                if source_weight.shape != target_weight.shape:
                    continue

                # Regularize different layers by different amounts, smaller at
                # the beginning and larger at the end/top of the network.
                #
                # Note: this makes it so the first layer differences don't
                # matter regardless of if they're the same shape or not.
                if FLAGS.hda_by_layer:
                    layer_weight = i/num_vars
                    # layer_weight = (num_vars-i)/num_vars
                else:
                    layer_weight = 1

                regularizer_loss += \
                    layer_weight * self.regularizer(target_weight - source_weight)
                total_weights += 1

        # Normalize by the number of weights, which hopefully helps this not
        # differ too much between models
        regularizer_loss /= total_weights

        return regularizer_loss

    def create_model(self, model_name):
        # For now we assume all sources have the same feature space. So, we need
        # two feature extractors -- one for source and one for target.
        num_feature_extractors = 2

        return models.HeterogeneousDannModel(
            self.num_classes, self.domain_outputs,
            self.global_step, self.total_steps,
            model_name=model_name,
            num_feature_extractors=num_feature_extractors,
            share_most_weights=self.share_most_weights)

    def prepare_data(self, data_sources, data_target):
        """ Prepare a batch of all source(s) data and target data separately,
        so we run through the source/target feature extractors separately """
        assert data_target is not None, \
            "cannot run Heterogeneous DA without target"
        x_a, y_a, domain_a = data_sources
        x_b, y_b, domain_b = data_target

        # Note: x_b, etc. isn't a list so doesn't need concat
        x = [tf.concat(x_a, axis=0), x_b]
        task_y_true = [tf.concat(y_a, axis=0), y_b]
        domain_y_true = [tf.concat(domain_a, axis=0), domain_b]

        return x, task_y_true, domain_y_true

    def prepare_data_eval(self, data, is_target):
        """ Don't concatenate elements of the list like in the base class since
        we want to handle the source/target domains separately, to pass to the
        right feature extractors."""
        x, y, domain = data

        assert isinstance(x, list), \
            "Must pass x=[...] even if only one domain for tf.function consistency"
        assert isinstance(y, list), \
            "Must pass y=[...] even if only one domain for tf.function consistency"
        assert isinstance(domain, list), \
            "Must pass domain=[...] even if only one domain for tf.function consistency"

        return x, y, domain

    def call_model(self, x, which_model, is_target=None, training=None, **kwargs):
        """ Run each source/target through appropriate feature extractor.
        If is_target=None, then this is training. If is_target=True, then this
        is evaluation of target data, and if is_target=False, then this is
        evaluation of source data. """
        task_y_pred = []
        domain_y_pred = []
        fe_output = []

        # Should be 2 for source/target or 1 during evaluation for just one
        assert (training is True and is_target is None and len(x) == 2) \
            or (training is False and (is_target is True or is_target is False)
                and len(x) == 1), \
            "is_target=None and len(x)=2 during training but " \
            "is_target=True/False and len(x)=1 during testing"

        for i in range(len(x)):
            # At test time, we set source/target explicitly -- use appropriate
            # feature extractor: sources = 0, target = 1 (see ordering in
            # prepare_data)
            if is_target is not None:
                which_fe = 1 if is_target else 0
            else:
                which_fe = i

            i_task_y_pred, i_domain_y_pred, i_fe_output = \
                self.model[which_model](x[i], which_fe=which_fe,
                    training=training, **kwargs)
            task_y_pred.append(i_task_y_pred)
            domain_y_pred.append(i_domain_y_pred)
            fe_output.append(i_fe_output)

        return task_y_pred, domain_y_pred, fe_output

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, which_model, training):
        """ Concatenate, then parent class's loss (e.g. DANN or DA-WS) """
        # Should all be the same length
        num = len(x)
        assert len(task_y_true) == num
        assert len(domain_y_true) == num
        assert len(task_y_pred) == num
        assert len(domain_y_pred) == num
        assert len(fe_output) == num

        # Since source/target may be different sizes, we can't concat everything
        # but instead need to compute the loss for source/target separately
        losses = []

        for i in range(num):
            losses.append(super().compute_losses(
                x[i], task_y_true[i], domain_y_true[i], task_y_pred[i],
                domain_y_pred[i], fe_output[i], which_model, training))

        # The returned losses are scalars, so now we can add them together, e.g.
        # if each is [total_loss, task_loss, d_loss] then add the source and
        # target total_loss's together, etc. Note: if inheriting from DAWS,
        # then we have a 4th loss "daws".
        losses_added = self.add_multiple_losses(losses, average=False)

        # We additionally regularize the FE's to be similar which we add to the
        # total loss. Element [0] is the total_loss from the parent class since
        # that is the loss we use to compute gradients w.r.t. the FE weights.
        #
        # Note: this errors at evaluation (post-training eval, not
        # during-training eval) since we run data through the model one domain
        # at a time, meaning that the model won't have populated
        # trainable_variables yet. Thus, since we don't look at the loss during
        # post-training evaluation anyway... just set it to zero.
        if training:
            fe_regularizer_loss = self.regularize_fe_weights_similar(which_model)
            losses_added[0] += fe_regularizer_loss
            losses_added.append(fe_regularizer_loss)
        else:
            losses_added.append(0)

        return losses_added

    def post_data_eval(self, task_y_true, task_y_pred, domain_y_true,
            domain_y_pred):
        """ Concatenate, then parent class's post_data_eval """
        # Unlike in compute_losses() where source/target may be different sizes
        # so we can't just concatenate, here during evaluation we only ever run
        # all source data or all target data through the model at a time, i.e.
        # we actually can concatenate the data here. Though, alternatively we
        # could follow the compute_losses() logic instead...
        task_y_true = tf.concat(task_y_true, axis=0)
        task_y_pred = tf.concat(task_y_pred, axis=0)
        domain_y_true = tf.concat(domain_y_true, axis=0)
        domain_y_pred = tf.concat(domain_y_pred, axis=0)
        return super().post_data_eval(task_y_true, task_y_pred, domain_y_true,
            domain_y_pred)

    def compute_gradients(self, tape, losses, which_model):
        # We only use fe_regularizer_loss for plotting -- for computing
        # gradients it's included in the total loss
        #
        # Note: if we inherit from DANN, then we have 4 here and skip the 4th.
        # If we inherit from DAWS though, then we have 5 here and skip the 5th,
        # passing in the 4th to get removed in the DAWS (identical)
        # compute_gradients() function.
        return super().compute_gradients(tape, losses[:-1], which_model)


@register_method("dann_hda")
class MethodHeterogeneousDann(HeterogeneousBase, MethodDann):
    pass


@register_method("daws_hda")
class MethodHeterogeneousDaws(HeterogeneousBase, MethodDaws):
    pass


class HeterogeneousBaselineBase:
    """ Base class for HDA drop/pad baselines (since there's a lot of shared
    code) """
    def _prepare_both(self, x_a, x_b, features_a, features_b):
        raise NotImplementedError("must implement _prepare_both()")
        return x_a, x_b

    def _prepare_one(self, x, features_a, features_b, is_target):
        raise NotImplementedError("must implement _prepare_one()")
        return x

    def prepare_data(self, data_sources, data_target):
        assert data_target is not None, "cannot run pad/drop baseline without target"
        x_a, y_a, domain_a = data_sources
        x_b, y_b, domain_b = data_target

        # Compare shapes, if different number of features then pad
        assert len(data_sources) > 0, "must have a least one source"
        batch_a, timesteps_a, features_a = x_a[0].shape
        batch_b, timesteps_b, features_b = x_b.shape
        assert timesteps_a == timesteps_b, \
            "should have padded source/target to get same number of timesteps"

        # Do the padding or dropping of features
        x_a, x_b = self._prepare_both(x_a, x_b, features_a, features_b)
        assert x_a[0].shape[-1] == x_b.shape[-1], "failed - not same shape"

        # Put back together and run the normal DANN prepare_data()
        data_sources = [x_a, y_a, domain_a]
        data_target = [x_b, y_b, domain_b]
        return super().prepare_data(data_sources, data_target)

    def prepare_data_eval(self, data, is_target):
        x, y, domain = data

        assert isinstance(x, list), \
            "Must pass x=[...] even if only one domain for tf.function consistency"
        assert isinstance(y, list), \
            "Must pass y=[...] even if only one domain for tf.function consistency"
        assert isinstance(domain, list), \
            "Must pass domain=[...] even if only one domain for tf.function consistency"

        source_features = self.source_datasets[0].feature_subset
        target_features = self.target_dataset.feature_subset

        assert source_features is not None \
            and target_features is not None, \
            "for HDA baselines, must specify --{source,target}_feature_subset " \
            "so we know how many features are in each domain (to pad/drop " \
            "accordingly)"

        features_a = len(source_features)
        features_b = len(target_features)

        for s in self.source_datasets[1:]:
            assert s.feature_subset == features_a, \
                "sources must have the same shape"

        # Do the padding/dropping of features
        x = self._prepare_one(x, features_a, features_b, is_target)

        # Put back together and run the normal prepare_data_eval()
        data = x, y, domain
        return super().prepare_data_eval(data, is_target)


class HeterogeneousPadBase(HeterogeneousBaselineBase):
    """ Pad source/target to same shape with zeros, then method """
    def _pad_features(self, x, desired_features):
        current_features = x.shape[-1]
        assert desired_features >= current_features, \
            "padding requires desired >= current"
        return tf.pad(x,
            [[0, 0], [0, 0], [0, desired_features - current_features]],
            "CONSTANT", constant_values=0)

    def _prepare_both(self, x_a, x_b, features_a, features_b):
        # pad x_b to make same shape as x_a
        if features_a > features_b:
            x_b = self._pad_features(x_b, desired_features=features_a)
        # pad each x_a to make same shape as x_b
        elif features_a < features_b:
            new_x_a = []
            for x in x_a:
                assert x.shape == x_a[0].shape, \
                    "sources must have the same shape"
                new_x_a.append(self._pad_features(x,
                    desired_features=features_b))
            x_a = new_x_a

        return x_a, x_b

    def _prepare_one(self, x, features_a, features_b, is_target):
        # pad x_b to make same shape as x_a (if x is x_b, i.e. is_target)
        if features_a > features_b and is_target:
            new_x = []
            for v in x:
                new_x.append(self._pad_features(v, desired_features=features_a))
            x = new_x
        # pad each x_a to make same shape as x_b (if x is x_a, i.e. not is_target)
        elif features_a < features_b and not is_target:
            new_x = []
            for v in x:
                new_x.append(self._pad_features(v, desired_features=features_b))
            x = new_x

        return x


@register_method("none_pad")
class MethodNonePadBaseline(HeterogeneousPadBase, MethodNone):
    pass


@register_method("dann_pad")
class MethodDannPadBaseline(HeterogeneousPadBase, MethodDann):
    pass


class HeterogeneousDropBase(HeterogeneousBaselineBase):
    """ Drop features from source/target to make the same shape, then DANN """
    def _drop_features(self, x, desired_features):
        """ Take first desired_length features """
        batch, timesteps, current_features = x.shape
        assert desired_features <= current_features, \
            "dropping requires desired <= current"
        return tf.slice(x, [0, 0, 0], [batch, timesteps, desired_features])

    def _prepare_both(self, x_a, x_b, features_a, features_b):
        # drop features from each x_a to match x_b's shape
        if features_a > features_b:
            new_x_a = []
            for x in x_a:
                assert x.shape == x_a[0].shape, \
                    "sources must have the same shape"
                new_x_a.append(self._drop_features(x,
                    desired_features=features_b))
            x_a = new_x_a
        # drop features from x_b to match x_a's shape
        elif features_a < features_b:
            x_b = self._drop_features(x_b, desired_features=features_a)

        return x_a, x_b

    def _prepare_one(self, x, features_a, features_b, is_target):
        # drop features from each x_a to match x_b's shape (if x is x_a, i.e.
        # not is_target)
        if features_a > features_b and not is_target:
            new_x = []
            for v in x:
                new_x.append(self._drop_features(v, desired_features=features_b))
            x = new_x
        # drop features from each x_b to match x_a's shape (if x is x_b, i.e.
        # is_target)
        elif features_a < features_b and is_target:
            new_x = []
            for v in x:
                new_x.append(self._drop_features(v, desired_features=features_a))
            x = new_x

        return x


@register_method("none_drop")
class MethodNoneDropBaseline(HeterogeneousDropBase, MethodNone):
    pass


@register_method("dann_drop")
class MethodDannDropBaseline(HeterogeneousDropBase, MethodDann):
    pass


#
# Domain generalization
#

@register_method("dann_dg")
class MethodDannDG(MethodDann):
    """
    DANN but to make it generalization rather than adaptation:

    - calculate_domain_outputs: don't include a softmax output for the target domain
    - domain_label: don't include target as domain 0, so start sources at 0
    - prepare_data: ignore the target domain when preparing the data for training
    - compute_losses: don't throw out domain 0 data since domain 0 is no longer
      the target
    """
    def calculate_domain_outputs(self):
        # SparseCategoricalCrossentropy gives an error if there's only one class.
        # Thus, throw in an extra, unused class (so softmax output always has 2).
        # Really, why would anybody do DG if there's only one domain...
        #
        # a=tf.constant([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94],
        #   [0.1, 0.6, 0.3]])
        # t=tf.constant([0,1,2,1])
        # cce(t,a)  ## works
        #
        # b=tf.constant([[1.0], [1.0], [1.0], [1.0]])
        # t=tf.constant([0,0,0,0])
        # cce(t,b)  ## errors:
        #   "ValueError: Shape mismatch: The shape of labels (received (342,))
        #   should equal the shape of logits except for the last dimension
        #   (received (1, 4))."
        if self.num_source_domains == 1:
            domain_outputs = 2
        else:
            domain_outputs = self.num_source_domains

        return domain_outputs

    def domain_label(self, index, is_target):
        """
        Shift down the domain labels so 0 is not source 1 since we don't have a
        target domain.

        Note: during evaluation, if target data is used, then the results will
        be wrong since target=0 and source #1=0 for the domain label.

        New domain labeling:
        0 = target
        0 = source #0
        1 = source #1
        2 = source #2
        ...
        """
        if is_target:
            return 0
        else:
            return index

    def prepare_data(self, data_sources, data_target):
        # Ignore target domain data when doing domain generalization
        x_a, y_a, domain_a = data_sources
        x = tf.concat(x_a, axis=0)
        task_y_true = tf.concat(y_a, axis=0)
        domain_y_true = tf.concat(domain_a, axis=0)
        return x, task_y_true, domain_y_true

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, which_model, training):
        # Since we don't have target domain data, don't throw out anything like
        # we did in MethodDANN()
        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = self.domain_loss(domain_y_true, domain_y_pred)
        total_loss = task_loss + d_loss
        return [total_loss, task_loss, d_loss]


@register_method("sleep_dg")
class MethodSleepDG(MethodDannDG):
    """ Same as DANN-DG but uses sleep model that feeds task classifier output
    to domain classifier """
    def create_model(self, model_name):
        return models.SleepModel(self.num_classes, self.domain_outputs,
            self.global_step, self.total_steps, model_name=model_name)


@register_method("aflac_dg")
class MethodAflacDG(MethodDannDG):
    """ AFLAC uses KL divergence rather than GRL

    The domain classifier is updated to correctly classify the domain. The task
    classifier to correctly classify the task. However, the feature extractor is
    updated with a combination of making the task classifier correct while also
    wanting the domain classifier's output probabilities to match P(d|y) for
    each known label y.

    For example, if an example in the dataset is y=0 then the domain classifier
    should output the probability of being in each domain such that it matches
    the probability of being in that domain out of the examples that have that
    label 0 in the source domain training data (i.e. P(d|y)).

    At least, that's how I understand it.

    Based on:
    https://github.com/akuzeee/AFLAC/blob/master/learner.py
    https://github.com/akuzeee/AFLAC/blob/master/DAN.py
    https://github.com/akuzeee/AFLAC/blob/master/AFLAC.py
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_names = ["fe_tc", "domain", "task", "kl"]
        self.mle_for_p_d_given_y()
        # Not fed to the model, but used in the loss
        self.grl_schedule = models.DannGrlSchedule(self.total_steps)

    def mle_for_p_d_given_y(self):
        """ Compute P(d|y)
        https://github.com/akuzeee/AFLAC/blob/master/AFLAC.py#L14

        Note: doing this rather than mle_for_p_d() since default is "dependent_y"
        in their code https://github.com/akuzeee/AFLAC/blob/master/run.py#L138
        """
        # Get lists of all labels and domains so we can compute how many there
        # are of each
        ys = []
        ds = []

        # The domain is 0 for source 0, 1 for source 1, etc.
        # Note: we use the "eval" train dataset since it doesn't repeat infinitely
        for d, dataset in enumerate(self.source_train_eval_datasets):
            for _, y in dataset:
                ys.append(y)
                ds.append(tf.ones_like(y)*d)

        # Fix Tensorflow bug / problem: expand, transpose, concat, then squeeze.
        # What I wanted to do is just tf.concat(ys, axis=0)... since ys is an
        # array of 1D tensors. But, it gives an error:
        # "Expected concatenating dimensions in the range [0, 0)"
        ys = [tf.transpose(tf.expand_dims(x, axis=0)) for x in ys]
        ds = [tf.transpose(tf.expand_dims(x, axis=0)) for x in ds]
        y = tf.cast(tf.squeeze(tf.concat(ys, axis=0)), dtype=tf.int32)
        d = tf.cast(tf.squeeze(tf.concat(ds, axis=0)), dtype=tf.int32)

        # Convert to numpy to ease converting the AFLAC code
        y = y.numpy()
        d = d.numpy()

        num_y_keys = len(np.unique(y))
        num_d_keys = len(np.unique(d))
        # Note: do <= not == since sometimes a person doesn't perform any of
        # a certain class, so it may be less. Though, for domains it really
        # should be equal unless one of the domains has no data.
        assert num_y_keys <= self.num_classes
        assert num_d_keys <= self.num_source_domains

        # Note: using domain_outputs not num_source_domains, since we have an
        # extra domain label if there's only one source domain.
        p_d_given_y = np.zeros((self.num_classes, self.domain_outputs),
            dtype=np.float32)

        # Classes are numbered 0, 1, ..., num_classes-1
        for y_key in range(self.num_classes):
            indices = np.where(y == y_key)
            d_given_key = d[indices]
            d_keys, d_counts = np.unique(d_given_key, return_counts=True)
            p_d_given_key = np.zeros((self.num_source_domains,))
            p_d_given_key[d_keys] = d_counts
            p_d_given_y[y_key] = p_d_given_key

        # Normalize so for each class, the domain counts sum to one
        p_d_given_y = tf.constant(p_d_given_y, dtype=tf.float32)
        p_d_given_y /= tf.reduce_sum(tf.math.abs(p_d_given_y), axis=1, keepdims=True)

        self.p_d_given_y = p_d_given_y

    def create_model(self, model_name):
        return models.BasicModel(self.num_classes, self.domain_outputs,
            model_name=model_name)

    def compute_losses(self, x, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, which_model, training):
        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = self.domain_loss(domain_y_true, domain_y_pred)

        # Gather the P(d|y) for the true y's for each example.
        # Note: this doesn't leak target-domain label information since this
        # is DG not MS-DA, so we have no data (x or y) for the target domain.
        d_true = tf.gather(self.p_d_given_y, tf.cast(task_y_true, dtype=tf.int32))

        # p_d_given_y (above, now d_true) is already normalized, but
        # domain_y_pred is just "logits" (no softmax in model), so pass the
        # domain_y_pred through softmax before computing KLD.
        #
        # Also, we could implement KL divergence as done in
        # https://github.com/akuzeee/AFLAC/blob/master/utils.py#L183 with
        # something like:
        #   cce = tf.keras.losses.CategoricalCrossentropy()
        #   kl_d = -cce(q, q) + cce(q, p)
        # However, it's equivalent to using the KLD function, so we'll just use
        # that.
        #
        # Pf: -cce(q,q) + cce(q,p)
        #   = sum_i^D q_i log q_i - sum_i^D q_i log p_i (for D domains)
        #   = sum_i^D q_i * (log q_i - log p_i)
        #   = sum_i^D q_i log(q_i/p_i)
        #   = D_KL(q||p)
        # (then of course, this is done for each example in the batch)
        #
        # See:
        # https://en.wikipedia.org/wiki/Cross_entropy
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        kl_loss = tf.keras.losses.KLD(d_true, tf.nn.softmax(domain_y_pred))

        # Looking at Figure 2 -- https://arxiv.org/pdf/1904.12543.pdf
        # They backpropagate the task and KL (weighted by alpha) losses to FE
        # (and task... but KL doesn't matter for updating the task classifier).
        # They backpropagate the domain loss for updating DC.
        #
        # Their code:
        # https://github.com/akuzeee/AFLAC/blob/master/AFLAC.py#L158
        # Note that y_optimizer only updates FE and TC and d_optimizer only
        # updates DC. Rather than putting in GradMultiplyLayerF into network,
        # I'll just calculate alpha here and weight the KL loss by it since
        # we're ignoring the gradient throughout DC anyway, don't need it to be
        # weighted only through part of the network.
        alpha = self.grl_schedule(self.global_step)
        fe_tc_loss = task_loss + alpha*kl_loss

        return [fe_tc_loss, d_loss, task_loss, kl_loss]

    def compute_gradients(self, tape, losses, which_model):
        fe_tc_loss, d_loss, _, _ = losses
        grad = tape.gradient(fe_tc_loss,
            self.model[which_model].trainable_variables_task_fe)
        d_grad = tape.gradient(d_loss,
            self.model[which_model].trainable_variables_domain)
        return [grad, d_grad]

    def apply_gradients(self, gradients, which_model):
        grad, d_grad = gradients
        self.opt[which_model]["opt"].apply_gradients(zip(grad,
            self.model[which_model].trainable_variables_task_fe))
        self.opt[which_model]["d_opt"].apply_gradients(zip(d_grad,
            self.model[which_model].trainable_variables_domain))


def make_loss(from_logits=True):
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

    def loss(y_true, y_pred):
        return cce(y_true, y_pred)

    return loss
