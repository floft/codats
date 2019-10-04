"""
Models

Provides the model DomainAdaptationModel() and its components along with the
make_{task,domain}_loss() functions
"""
import tensorflow as tf

from absl import flags

FLAGS = flags.FLAGS


@tf.custom_gradient
def flip_gradient(x, grl_lambda=1.0):
    """ Forward pass identity, backward pass negate gradient and multiply by  """
    grl_lambda = tf.cast(grl_lambda, dtype=tf.float32)

    def grad(dy):
        # the 0 is for grl_lambda, which doesn't have a gradient
        return tf.negative(dy) * grl_lambda * tf.ones_like(x), 0

    return x, grad


class FlipGradient(tf.keras.layers.Layer):
    """
    Gradient reversal layer

    global_step = tf.Variable storing the current step
    schedule = a function taking the global_step and computing the grl_lambda,
        e.g. `lambda step: 1.0` or some more complex function.
    """
    def __init__(self, global_step, grl_schedule, **kwargs):
        super().__init__(**kwargs)
        self.global_step = global_step
        self.grl_schedule = grl_schedule

    def call(self, inputs, **kwargs):
        """ Calculate grl_lambda first based on the current global step (a
        variable) and then create the layer that does nothing except flip
        the gradients """
        grl_lambda = self.grl_schedule(self.global_step)
        return flip_gradient(inputs, grl_lambda)


def ConstantGrlSchedule(constant=1.0):
    """ Constant GRL schedule (always returns the same number) """
    def schedule(step):
        return constant
    return schedule


def DisableGrlSchedule():
    """ Setting grl_lambda=-0.1 removes any effect from it """
    def schedule(step):
        return -1.0
    return schedule


def DannGrlSchedule(num_steps):
    """ GRL schedule from DANN paper """
    num_steps = tf.cast(num_steps, tf.float32)

    def schedule(step):
        step = tf.cast(step, tf.float32)
        return 2/(1+tf.exp(-10*(step/(num_steps+1))))-1

    return schedule


class StopGradient(tf.keras.layers.Layer):
    """ Stop gradient layer """
    def call(self, inputs, **kwargs):
        return tf.stop_gradient(inputs)


def make_fcn_model(num_classes, num_domains):
    """
    FCN (fully CNN) -- but domain classifier has additional dense layers

    From: https://arxiv.org/pdf/1611.06455.pdf
    Tested in: https://arxiv.org/pdf/1809.04356.pdf
    Code from: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
    """
    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding="same",
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding="same",
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same",
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.GlobalAveragePooling1D(),
    ])
    task_classifier = tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    domain_classifier = tf.keras.Sequential([
        # Note: alternative is Dense(128, activation="tanh") like used by
        # https://arxiv.org/pdf/1902.09820.pdf They say dropout of 0.7 but
        # I'm not sure if that means 1-0.7 = 0.3 or 0.7 itself.
        tf.keras.layers.Dense(500, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(500, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_domains, "softmax"),
    ])

    return feature_extractor, task_classifier, domain_classifier


class DomainAdaptationModel(tf.keras.Model):
    """
    Domain adaptation model -- task and domain classifier outputs, depends on
    command line --model=X argument

    Usage:
        model = DomainAdaptationModel(num_classes, "flat",
            global_step, num_steps)

        with tf.GradientTape() as tape:
            task_y_pred, domain_y_pred, embedding = model(x, training=True)
            ...
    """
    def __init__(self, num_classes, num_domains, model_name, global_step,
            num_steps, use_grl=False, **kwargs):
        super().__init__(**kwargs)
        if use_grl:
            grl_schedule = DannGrlSchedule(num_steps)
        else:
            grl_schedule = DisableGrlSchedule()

        # For MDAN Smooth, it's binary classification but we have a separate
        # discriminator for each source-target pair.
        # Actually, for now let's use the right number of domains so we
        # don't have to compute new domain labels
        #if FLAGS.method == "dann_smooth":
        #    args = (num_classes, 2)

        args = (num_classes, num_domains)

        if model_name == "fcn":
            fe, task, domain = make_fcn_model(*args)
        else:
            raise NotImplementedError("Model name: "+str(model_name))

        self.feature_extractor = fe
        self.task_classifier = task
        self.flip_gradient = FlipGradient(global_step, grl_schedule)

        # MDAN Smooth requires multiple domain classifiers
        if FLAGS.method == "dann_smooth":
            # Note: no need for a target domain classifier, so it's actually
            # num_domains-1 domain classifiers
            self.domain_classifier = [domain]

            # Start at 2 since we already have one
            for i in range(2, num_domains):
                self.domain_classifier.append(
                    tf.keras.models.clone_model(domain))
        else:
            self.domain_classifier = domain

        # For sleep generalization method
        if FLAGS.method == "sleep_dg":
            self.concat = tf.keras.layers.Concatenate(axis=1)
            self.stop_gradient = StopGradient()

        # Target classifier (if used) will be the same as the task classifier
        # but will be trained on pseudo-labeled data. Then, call
        # model(..., target=True) to use this classifier rather than the task
        # classifier.
        self.target_classifier = tf.keras.models.clone_model(task)

    @property
    def trainable_variables_task(self):
        return self.feature_extractor.trainable_variables \
            + self.task_classifier.trainable_variables

    @property
    def trainable_variables_domain(self):
        # If multiple domain classifiers, get variables from all of them
        if isinstance(self.domain_classifier, list):
            domain_vars = []

            for dc in self.domain_classifier:
                domain_vars += dc.trainable_variables
        else:
            domain_vars = self.domain_classifier.trainable_variables

        return domain_vars

    @property
    def trainable_variables_task_domain(self):
        return self.feature_extractor.trainable_variables \
            + self.task_classifier.trainable_variables \
            + self.trainable_variables_domain

    @property
    def trainable_variables_target(self):
        return self.feature_extractor.trainable_variables \
            + self.target_classifier.trainable_variables

    def call(self, inputs, domain="source", domain_classifier=None,
            target=False, training=None, **kwargs):
        # Manually set the learning phase since we probably aren't using .fit()
        if training is True:
            tf.keras.backend.set_learning_phase(1)
        elif training is False:
            tf.keras.backend.set_learning_phase(0)

        fe = self.feature_extractor(inputs, domain=domain, **kwargs)

        # If an RNN, then we'll return (rnn_output, rnn_state), so pass the
        # output to the classifiers but return both from call()
        if isinstance(fe, tuple):
            fe_output = fe[0]
        else:
            fe_output = fe

        # If desired, use the target classifier rather than the task classifier
        if target:
            task = self.target_classifier(fe_output, domain=domain, **kwargs)
        else:
            task = self.task_classifier(fe_output, domain=domain, **kwargs)

        # Flip the gradient, if desired. If disabled, then this does nothing.
        grl_output = self.flip_gradient(fe_output, domain=domain, **kwargs)

        # If doing the algorithm from the sleep paper, then for generalization
        # we also concatenate the task classifier's output when fed to the
        # domain classifier.
        if FLAGS.method == "sleep_dg":
            task_stop_gradient = self.stop_gradient(task)
            domain_input = self.concat([grl_output, task_stop_gradient])
        else:
            domain_input = grl_output

        if isinstance(self.domain_classifier, list):
            # We know which one to use (0 = source domain 1, 1 = source domain 2, etc.)
            if domain_classifier is not None:
                domain_output = self.domain_classifier[domain_classifier](domain_input, **kwargs)
            # We don't know which one, so do them all
            else:
                domain_output = []

                for dc in self.domain_classifier:
                    # Note: this feeds all the data through each domain classifier
                    # since at this point we don't know which data is from which
                    # domain. In main.py when computing the loss, we'll take the
                    # proper slices of these outputs.
                    #
                    # Note: can't pass domain=domain since it errors for some reason
                    # due to the layer cloning, but we have CustomSequential disabled
                    # anyway, so it doesn't really matter.
                    domain_output.append(dc(domain_input, **kwargs))
        else:
            domain_output = self.domain_classifier(domain_input, domain=domain, **kwargs)

        return task, domain_output, fe


def make_task_loss(adapt):
    """
    The same as CategoricalCrossentropy() but only on half the batch if doing
    adaptation and in the training phase
    """
    cce = tf.keras.losses.SparseCategoricalCrossentropy()

    def task_loss(y_true, y_pred, training=None):
        """
        Compute loss on the outputs of the task classifier

        Note: domain classifier can use normal tf.keras.losses.CategoricalCrossentropy
        but for the task loss when doing adaptation we need to ignore the second half
        of the batch since this is unsupervised
        """
        if training is None:
            training = tf.keras.backend.learning_phase()

        # If doing domain adaptation, then we'll need to ignore the second half of the
        # batch for task classification during training since we don't know the labels
        # of the target data
        if adapt and training:
            batch_size = tf.shape(y_pred)[0]
            y_pred = tf.slice(y_pred, [0, 0], [batch_size // 2, -1])
            # With the sparse loss, this isn't 2D anymore
            y_true = tf.slice(y_true, [0], [batch_size // 2])

        return cce(y_true, y_pred)

    return task_loss


def make_weighted_loss():
    """ The same as CategoricalCrossentropy() but weighted """
    cce = tf.keras.losses.SparseCategoricalCrossentropy()

    def task_loss(y_true, y_pred, weights, training=None):
        """
        Compute loss on the outputs of a classifier weighted by the specified
        weights
        """
        return cce(y_true, y_pred, sample_weight=weights)

    return task_loss


def make_domain_loss(use_domain_loss):
    """
    Just CategoricalCrossentropy() but for consistency with make_task_loss()
    """
    if use_domain_loss:
        cce = tf.keras.losses.SparseCategoricalCrossentropy()

        def domain_loss(y_true, y_pred):
            """ Compute loss on the outputs of the domain classifier """
            return cce(y_true, y_pred)
    else:
        def domain_loss(y_true, y_pred):
            """ Domain loss only used during adaptation """
            return 0

    return domain_loss


# List of names
models = [
    "fcn",
]


# Get names
def names():
    """
    Returns list of all the available models for use in DomainAdaptationModel()
    """
    return models
