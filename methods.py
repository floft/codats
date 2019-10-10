"""
Methods
"""
import numpy as np
import tensorflow as tf

from absl import flags

import models
import load_datasets

FLAGS = flags.FLAGS

flags.DEFINE_enum("model", "fcn", models.names(), "What model to use for the method")
flags.DEFINE_float("lr", 0.0001, "Learning rate for training")
flags.DEFINE_float("lr_domain_mult", 1.0, "Learning rate multiplier for training domain classifier")


class MethodBase:
    def __init__(self, source_datasets, target_dataset, *args, **kwargs):
        self.source_datasets = source_datasets
        self.target_dataset = target_dataset

        # Support multiple targets when we add that functionality
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

        # We need to know the num_classes for creating the model
        # We'll just pick the first source since we have to have at least one
        # source and we've already verified they're all the same in load_da()
        self.num_classes = source_datasets[0].num_classes

        # What we want in the checkpoint
        self.checkpoint_variables = {}

        # Initialize components
        self.create_iterators()
        self.create_optimizers()
        self.create_model()
        self.create_losses()

        # Always save the model in the checkpoint
        self.checkpoint_variables["model"] = self.model

        # Names of the losses returned in compute_losses
        self.loss_names = ["total"]

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

    def create_optimizers(self):
        self.opt = tf.keras.optimizers.Adam(FLAGS.lr)
        self.checkpoint_variables["opt"] = self.opt

    def create_model(self):
        self.model = models.FcnModelBase(self.num_classes, self.num_domains)

    def create_losses(self):
        self.task_loss = make_loss()

    def get_next_train_data(self):
        """ Get next batch of training data """
        # This is a separate function from train_step() so that we can evaluate
        # in Metrics() with the exact same data as we used in train_step()
        data_sources = [next(x) for x in self.source_train_iterators]
        data_target = next(self.target_train_iterator) \
            if self.target_train_iterator is not None else None
        return self.get_next_data(data_sources, data_target)

    def get_next_batch(self, source_iterators, target_iterator):
        """ Get next batch of data, same as get_next_train_batch()
        except allows you to pass in your own iterators """
        data_sources = [next(x) for x in source_iterators] \
            if source_iterators is not None else None
        data_target = next(target_iterator) \
            if target_iterator is not None else None
        return self.get_next_data(data_sources, data_target)

    @tf.function
    def get_next_data(self, data_sources, data_target):
        """
        Get next set of training data. Note: needs to handle when either one of
        data_sources or data_target is None, which is used during evaluation to
        evaluate the datasets separately.

        Returns:
            data_sources, data_target

        where data_sources = (
            [x_a1, x_a2, x_a3, ...],
            [y_a1, y_a2, y_a3, ...],
            [domain_a1, domain_a2, domain_a3, ...]
        ) or None if no source domains
        and data_target = (x_b, y_b, domain_b) or None if no target domain
        """
        if data_sources is not None:
            # Don't concatenate/stack here since different methods may handle the
            # domains' data differently.
            x_sources = []
            y_sources = []
            domain_sources = []

            # Domain numbers: for now we'll use 0 = target, 1 = source #1,
            # 2 = source #2, 3 = source #3, etc.
            for i, (x, y) in enumerate(data_sources):
                d = tf.ones_like(y)*(i+1)
                x_sources.append(x)
                y_sources.append(y)
                domain_sources.append(d)

            data_sources = (x_sources, y_sources, domain_sources)
        else:
            data_sources = None

        # If there's a target domain, it'll have domain label 0
        if data_target is not None:
            x_target, y_target = data_target
            domain_target = tf.zeros_like(y_target)
            data_target = (x_target, y_target, domain_target)
        else:
            data_target = None

        return data_sources, data_target

    # Allow easily overriding each part of the train_step() function, without
    # having to override train_step() in its entirety
    def prepare_data(self, data_sources, data_target):
        """ Prepare the data for the model, e.g. by concatenating all sources
        together. Note: do not put code in here that changes the domain labels
        since you presumably want that during evaluation too. Put that in
        get_next_data() """
        # By default (e.g. for no adaptation or domain generalization), ignore
        # the target data
        x_a, y_a, domain_a = data_sources
        x = tf.concat(x_a, axis=0)
        task_y_true = tf.concat(y_a, axis=0)
        domain_y_true = tf.concat(domain_a, axis=0)
        return x, task_y_true, domain_y_true

    def prepare_data_eval(self, data):
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

    def call_model(self, x, **kwargs):
        return self.model(x, **kwargs)

    def compute_losses(self, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, training):
        # Maybe: regularization = sum(model.losses) and add to loss
        return self.task_loss(task_y_true, task_y_pred)

    def compute_gradients(self, tape, loss):
        return tape.gradient(loss, self.model.trainable_variables_task)

    def apply_gradients(self, grad):
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables_task))

    @tf.function
    def train_step(self, data_sources, data_target):
        """
        Prepare data, run through model, compute losses, apply the gradients

        Override the individual parts with prepare_data(), call_model(),
        compute_losses(), compute_gradients(), and apply_gradients()
        """
        x, task_y_true, domain_y_true = self.prepare_data(data_sources, data_target)

        with tf.GradientTape(persistent=True) as tape:
            task_y_pred, domain_y_pred, fe_output = self.call_model(x, training=True)
            losses = self.compute_losses(task_y_true, domain_y_true,
                task_y_pred, domain_y_pred, fe_output, training=True)

        gradients = self.compute_gradients(tape, losses)
        del tape
        self.apply_gradients(gradients)

    def eval_step(self, data):
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

        return self.eval_step_list((x, y, domain))

    @tf.function
    def eval_step_list(self, data):
        """ Override preparation in prepare_data_eval() """
        x, task_y_true, domain_y_true = self.prepare_data_eval(data)

        # Run through model
        task_y_pred, domain_y_pred, fe_output = self.call_model(x, training=False)

        # Calculate losses
        losses = self.compute_losses(task_y_true, domain_y_true,
            task_y_pred, domain_y_pred, fe_output, training=False)

        # Post-process data (e.g. compute softmax from logits)
        task_y_true, task_y_pred, domain_y_true, domain_y_pred = \
            self.post_data_eval(task_y_true, task_y_pred, domain_y_true,
            domain_y_pred)

        return task_y_true, task_y_pred, domain_y_true, domain_y_pred, losses


# The base method class performs no adaptation
class MethodNone(MethodBase):
    pass


class MethodDann(MethodBase):
    def __init__(self, source_datasets, target_dataset,
            global_step, total_steps, *args, **kwargs):
        self.global_step = global_step  # should be TF variable
        self.total_steps = total_steps
        super().__init__(source_datasets, target_dataset, *args, **kwargs)
        self.loss_names += ["task", "domain"]

    def create_model(self):
        self.model = models.DannModel(self.num_classes, self.num_domains,
            self.global_step, self.total_steps)

    def create_optimizers(self):
        super().create_optimizers()
        # We need an additional optimizer for DANN
        self.d_opt = tf.keras.optimizers.Adam(FLAGS.lr*FLAGS.lr_domain_mult)
        self.checkpoint_variables["d_opt"] = self.d_opt

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

    def compute_losses(self, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, training):
        nontarget = tf.where(tf.squeeze(tf.not_equal(domain_y_true, 0)))
        task_y_true = tf.gather(task_y_true, nontarget)
        task_y_pred = tf.gather(task_y_pred, nontarget)

        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = self.domain_loss(domain_y_true, domain_y_pred)
        total_loss = task_loss + d_loss
        return [total_loss, task_loss, d_loss]

    def compute_gradients(self, tape, losses):
        total_loss, task_loss, d_loss = losses
        grad = tape.gradient(total_loss, self.model.trainable_variables_task_domain)
        d_grad = tape.gradient(d_loss, self.model.trainable_variables_domain)
        return [grad, d_grad]

    def apply_gradients(self, gradients):
        grad, d_grad = gradients
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables_task_domain))
        # Update discriminator again
        self.d_opt.apply_gradients(zip(d_grad, self.model.trainable_variables_domain))


class MethodDannGS(MethodDann):
    """ Same as DANN but only has 2 domains, any source is domain 1 (i.e. group
    them) and the target is still domain 0 """
    def create_model(self):
        assert self.num_domains > 1, "cannot do GS-DANN with only 1 domain"
        self.model = models.DannModel(self.num_classes, 2,
            self.global_step, self.total_steps)

    @tf.function
    def get_next_data(self, data_sources, data_target):
        """ Replace all source domains' domain labels with 1, i.e. group all
        sources together """
        data_sources, data_target = super().get_next_data(
            data_sources, data_target)

        if data_sources is not None:
            x_sources, y_sources, domain_sources = data_sources

            new_domain_sources = []

            for domain in domain_sources:
                new_domain_sources.append(tf.ones_like(domain))

            data_sources = (x_sources, y_sources, new_domain_sources)

        return data_sources, data_target


class MethodDannSmooth(MethodDannGS):
    """ MDAN Smooth method based on MethodDannGS since we want binary source = 1,
    target = 0 for the domain labels """
    def create_model(self):
        num_source_domains = len(self.source_datasets)
        self.model = models.DannSmoothModel(self.num_classes, 2,
            self.global_step, self.total_steps,
            num_domain_classifiers=num_source_domains)

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

    def prepare_data_eval(self, data):
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

    def call_model(self, x, **kwargs):
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
                self.model(x[i], domain_classifier=i, **kwargs)
            task_y_pred.append(i_task_y_pred)
            domain_y_pred.append(i_domain_y_pred)
            fe_output.append(i_fe_output)

        return task_y_pred, domain_y_pred, fe_output

    def compute_losses(self, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, training):
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
            nontarget = tf.where(tf.squeeze(tf.not_equal(domain_y_true[i], 0)))
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

    def compute_gradients(self, tape, losses):
        """ We have one loss, update everything with it """
        return tape.gradient(losses, self.model.trainable_variables_task_domain)

    def apply_gradients(self, gradients):
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables_task_domain))


class MethodDannDG(MethodDann):
    """
    DANN but to make it generalization rather than adaptation:

    - create_model: don't include a softmax output for the target domain
    - get_next_data: shift down the domain labels so 0 is now source 1
    - prepare_data: ignore the target domain when preparing the data
    - compute_losses: don't throw out domain 0 data since domain 0 is no longer
      the target
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_source_domains = len(self.source_datasets)

    def create_model(self):
        num_source_domains = len(self.source_datasets)
        self.model = models.DannModel(self.num_classes, num_source_domains,
            self.global_step, self.total_steps)

    @tf.function
    def get_next_data(self, data_sources, data_target):
        """
        Shift down the domain labels so 0 is not source 1 since we don't have a
        target domain.

        Note: during evaluation, if target data is used, then the results will
        be wrong since target=0 and source #1=0 for the domain label.
        """
        data_sources, data_target = super().get_next_data(
            data_sources, data_target)

        if data_sources is not None:
            x_sources, y_sources, domain_sources = data_sources

            new_domain_sources = []

            for domain in domain_sources:
                new_domain_sources.append(domain - 1)

            data_sources = (x_sources, y_sources, new_domain_sources)

        return data_sources, data_target

    def prepare_data(self, data_sources, data_target):
        # Ignore target domain data when doing domain generalization
        x_a, y_a, domain_a = data_sources
        x = tf.concat(x_a, axis=0)
        task_y_true = tf.concat(y_a, axis=0)
        domain_y_true = tf.concat(domain_a, axis=0)
        return x, task_y_true, domain_y_true

    def compute_losses(self, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, training):
        # Since we don't have target domain data, don't throw out anything like
        # we did in MethodDANN()
        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = self.domain_loss(domain_y_true, domain_y_pred)
        total_loss = task_loss + d_loss
        return [total_loss, task_loss, d_loss]


class MethodSleepDG(MethodDannDG):
    """ Same as DANN-DG but uses sleep model that feeds task classifier output
    to domain classifier """
    def create_model(self):
        self.model = models.SleepModel(self.num_classes, self.num_domains,
            self.global_step, self.total_steps)


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

        # Fix Tensorflow bug / problem: expand, transpose, concat, then squeeze
        # Error: "Expected concatenating dimensions in the range [0, 0)"
        ys = [tf.transpose(tf.expand_dims(x, axis=0)) for x in ys]
        ds = [tf.transpose(tf.expand_dims(x, axis=0)) for x in ds]
        y = tf.cast(tf.squeeze(tf.concat(ys, axis=0)), dtype=tf.int32)
        d = tf.cast(tf.squeeze(tf.concat(ds, axis=0)), dtype=tf.int32)

        # Convert to numpy to ease converting the AFLAC code
        y = y.numpy()
        d = d.numpy()

        num_y_keys = len(np.unique(y))
        num_d_keys = len(np.unique(d))
        assert num_y_keys == self.num_classes
        assert num_d_keys == self.num_source_domains

        p_d_given_y = np.zeros((self.num_classes, self.num_source_domains),
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

    def create_model(self):
        num_source_domains = len(self.source_datasets)
        self.model = models.FcnModelBase(self.num_classes, num_source_domains)

    def compute_losses(self, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, training):
        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = self.domain_loss(domain_y_true, domain_y_pred)

        # Gather the P(d|y) for the true y's for each example.
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

    def compute_gradients(self, tape, losses):
        fe_tc_loss, d_loss, _, _ = losses
        grad = tape.gradient(fe_tc_loss, self.model.trainable_variables_task)
        d_grad = tape.gradient(d_loss, self.model.trainable_variables_domain)
        return [grad, d_grad]

    def apply_gradients(self, gradients):
        grad, d_grad = gradients
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables_task))
        self.d_opt.apply_gradients(zip(d_grad, self.model.trainable_variables_domain))


def make_loss(from_logits=True):
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)

    def loss(y_true, y_pred):
        return cce(y_true, y_pred)

    return loss


# Load a method
def load(name, *args, **kwargs):
    """ Load a method (must be one of methods.names()) """
    assert name in methods.keys(), "Name specified not in methods.names()"
    return methods[name](*args, **kwargs)


# List of methods
methods = {
    # No adaptation or training on target
    "none": MethodNone,

    # Multi-source domain adaptation (works with it...)
    "dann": MethodDann,
    "dann_gs": MethodDannGS,
    "dann_smooth": MethodDannSmooth,

    # Domain generalization
    "dann_dg": MethodDannDG,
    "sleep_dg": MethodSleepDG,
    "aflac_dg": MethodAflacDG,
    #"ciddg_dg": MethodCiddgDG,
}


# Get names
def names():
    """ Returns list of all the available methods """
    return list(methods.keys())
