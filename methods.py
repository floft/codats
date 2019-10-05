"""
Methods
"""
import tensorflow as tf

from absl import flags

import models

FLAGS = flags.FLAGS

flags.DEFINE_enum("model", "fcn", models.names(), "What model to use for the method")
flags.DEFINE_float("lr", 0.001, "Learning rate for training")
flags.DEFINE_float("lr_domain_mult", 1.0, "Learning rate multiplier for training domain classifier")


class MethodBase:
    def __init__(self, source_datasets, target_dataset, *args, **kwargs):
        self.source_datasets = source_datasets
        self.target_dataset = target_dataset

        # Support multiple targets when we add that functionality
        self.num_domains = len(source_datasets)

        if target_dataset is not None:
            if isinstance(target_dataset, int):
                self.num_domains += 1
            elif isinstance(target_dataset, list):
                self.num_domains += len(target_dataset)
            else:
                raise NotImplementedError("target_dataset should be either one "
                    "int or a list")

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

    def create_iterators(self):
        """ Get the source/target train/eval datasets """
        self.source_train_iterators = [iter(x.train) for x in self.source_datasets]
        self.source_eval_datasets = [x.test_evaluation for x in self.source_datasets]

        if self.target_dataset is not None:
            self.target_train_iterator = iter(self.target_dataset.train)
            self.target_eval_dataset = self.target_dataset.test_evaluation
        else:
            self.target_train_iterator = None
            self.target_eval_dataset = None

    def create_optimizers(self):
        self.opt = tf.keras.optimizers.Adam(FLAGS.lr)
        self.checkpoint_variables["opt"] = self.opt

    def create_model(self):
        self.model = models.FcnModelBase(self.num_classes, self.num_domains)

    def create_losses(self):
        self.task_loss = make_loss()

    @tf.function
    def get_next_data(self):
        """
        Get next set of training data

        Returns:
            data_sources, data_target

        where data_sources = (
            [x_a1, x_a2, x_a3, ...],
            [y_a1, y_a2, y_a3, ...],
            [domain_a1, domain_a2, domain_a3, ...]
        )
        and data_target = (x_b, y_b, domain_b) or None if no target domain
        """
        # This is a separate function from train_step() so that we can evaluate
        # in Metrics() with the exact same data as we used in train_step()
        data_sources = [next(x) for x in self.source_train_iterators]
        data_target = next(self.target_train_iterator) \
            if self.target_train_iterator is not None else None

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

        # If there's a target domain, it'll have domain label 0
        if data_target is not None:
            x_target, y_target = data_target
            domain_target = tf.zeros_like(y_target)
            data_target = (x_target, y_target, domain_target)
        else:
            data_target = None

        return (x_sources, y_sources, domain_sources), \
            data_target

    # Allow easily overriding each part of the train_step() function, without
    # having to override train_step() in its entirety
    def prepare_data(self, data_sources, data_target):
        # By default (e.g. for no adaptation or domain generalization), ignore
        # the target data
        x_a, y_a, domain_a = data_sources
        x = tf.concat(x_a, axis=0)
        task_y_true = tf.concat(y_a, axis=0)
        domain_y_true = tf.concat(domain_a, axis=0)
        return x, task_y_true, domain_y_true

    def call_model(self, x, **kwargs):
        return self.model(x, **kwargs)

    def compute_losses(self, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, training):
        # Maybe: regularization = sum(model.losses) and add to loss
        return self.task_loss(task_y_true, task_y_pred, training=training)

    def compute_gradients(self, tape, loss):
        return tape.gradient(loss, self.model.trainable_variables_task)

    def apply_gradients(self, grad):
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables_task))

    @tf.function
    def train_step(self, data_sources, data_target):
        x, task_y_true, domain_y_true = self.prepare_data(data_sources, data_target)

        with tf.GradientTape(persistent=True) as tape:
            task_y_pred, domain_y_pred, fe_output = self.call_model(x, training=True)
            losses = self.compute_losses(task_y_true, domain_y_true,
                task_y_pred, domain_y_pred, fe_output, training=True)

        gradients = self.compute_gradients(tape, losses)
        del tape
        self.apply_gradients(gradients)

    @tf.function
    def eval_step(self, data_sources, data_target):
        # Don't call prepare_data() since that is for preparing the data for
        # training, e.g. it gets rid of target labels
        x_a, y_a, domain_a = data_sources
        x_b, y_b, domain_b = data_target

        # Concatenate all domain data
        x = tf.concat(x_a+[x_b], axis=0)
        task_y_true = tf.concat(y_a+[y_b], axis=0)
        domain_y_true = tf.concat(domain_a+[domain_b], axis=0)

        # Run through model
        task_y_pred, domain_y_pred, fe_output = self.call_model(x, training=False)

        # Calculate losses
        losses = self.compute_losses(task_y_true, domain_y_true,
            task_y_pred, domain_y_pred, fe_output, training=False)

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

        # TODO remove, just want to check that it's initialized like I think
        assert "d_opt" in self.checkpoint_variables, \
            "didn't create domain optimizer?"

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
        # Only take subset of task labels/predictions on the source domains
        # since we don't know the target labels. Note: this is only during
        # training. During testing, we want to evaluate everything.
        if training:
            nontarget = tf.squeeze(tf.not_equal(domain_y_true, 0))
            task_y_true = tf.gather(task_y_true, nontarget)
            task_y_pred = tf.gather(task_y_pred, nontarget)

        task_loss = self.task_loss(task_y_true, task_y_pred, training=training)
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
    them) and the target is domain 0 """
    def create_model(self):
        assert self.num_domains > 1, "cannot do GS-DANN with only 1 domain"
        self.model = models.DannModel(self.num_classes, 2,
            self.global_step, self.total_steps)

    @tf.function
    def get_next_data(self):
        """ Replace all source domains' domain labels with 1, i.e. group all
        sources together """
        data_sources, data_target = super().get_next_data()
        x_sources, y_sources, domain_sources = data_sources

        new_domain_sources = []

        for domain in domain_sources:
            new_domain_sources.append(tf.ones_like(domain))

        return (x_sources, y_sources, new_domain_sources), data_target


class MethodDannSmooth(MethodDann):
    pass # TODO
    def create_model(self):
        self.model = models.DannSmoothModel(self.num_classes, self.num_domains,
            self.global_step, self.total_steps)

    def call_model(self, x, **kwargs):
        # TODO set domain_classifier to source #???
        return self.model(x, domain_classifier=None, **kwargs)

    def compute_losses(self, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, training):
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

        # For MDAN Smooth ideally we'd grab the loss from the right domain
        # classifier based on which domain the sample is from. However,
        # for now just grab the first classifier's output. TODO
        if isinstance(domain_y_pred, list):
            domain_y_pred = domain_y_pred[0] # TODO

        # TODO compute loss here

        return super().compute_losses(task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, training)


class MethodDannDG(MethodDann):
    """
    DANN but to make it generalization rather than adaptation:

    - create_model: don't include a softmax output for the target domain
    - prepare_data:
        - ignore the target domain when preparing the data
        - shift down the domain labels so 0 is now source 1
    - compute_losses: don't throw out domain 0 data since domain 0 is no longer
      the target
    """
    def create_model(self):
        num_source_domains = len(self.source_datasets)
        self.model = models.DannModel(self.num_classes, num_source_domains,
            self.global_step, self.total_steps)

    def prepare_data(self, data_sources, data_target):
        # Ignore target domain data when doing domain generalization
        x_a, y_a, domain_a = data_sources
        domain_a = domain_a - 1  # Get rid of the 0 domain label for target
        x = tf.concat(x_a, axis=0)
        task_y_true = tf.concat(y_a, axis=0)
        domain_y_true = tf.concat(domain_a, axis=0)
        return x, task_y_true, domain_y_true

    def compute_losses(self, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, training):
        # Since we don't have target domain data, don't throw out anything like
        # we did in MethodDANN()
        task_loss = self.task_loss(task_y_true, task_y_pred, training=training)
        d_loss = self.domain_loss(domain_y_true, domain_y_pred)
        total_loss = task_loss + d_loss
        return [total_loss, task_loss, d_loss]


class MethodSleepDG(MethodDannDG):
    """ Same as DANN-DG but uses sleep model that feeds task classifier output
    to domain classifier """
    def create_model(self):
        self.model = models.SleepModel(self.num_classes, self.num_domains,
            self.global_step, self.total_steps)


def make_loss():
    cce = tf.keras.losses.SparseCategoricalCrossentropy()

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
    #"aflac_dg": MethodAflacDG,
    #"ciddg_dg": MethodCiddgDG,
}


# Get names
def names():
    """ Returns list of all the available methods """
    return list(methods.keys())
