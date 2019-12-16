"""
Models
"""
import tensorflow as tf

from absl import flags

FLAGS = flags.FLAGS


@tf.custom_gradient
def flip_gradient(x, grl_lambda):
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


class FcnModelBase(tf.keras.Model):
    """
    FCN (fully CNN) -- but domain classifier has additional dense layers

    From: https://arxiv.org/pdf/1611.06455.pdf
    Tested in: https://arxiv.org/pdf/1809.04356.pdf
    Code from: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
    """
    def __init__(self, num_classes, num_domains, **kwargs):
        assert FLAGS.model == "fcn", "currently only support FCN"
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.feature_extractor = tf.keras.Sequential([
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
        self.task_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(num_classes),
        ])
        self.domain_classifier = tf.keras.Sequential([
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

            tf.keras.layers.Dense(num_domains),
        ])

    @property
    def trainable_variables_task(self):
        return self.feature_extractor.trainable_variables \
            + self.task_classifier.trainable_variables

    @property
    def trainable_variables_domain(self):
        return self.domain_classifier.trainable_variables

    @property
    def trainable_variables_task_domain(self):
        return self.feature_extractor.trainable_variables \
            + self.task_classifier.trainable_variables \
            + self.trainable_variables_domain

    def set_learning_phase(self, training):
        # Manually set the learning phase since we probably aren't using .fit()
        # but layers like batch norm and dropout still need to know if
        # training/testing
        if training is True:
            tf.keras.backend.set_learning_phase(1)
        elif training is False:
            tf.keras.backend.set_learning_phase(0)

    # Allow easily overriding each part of the call() function, without having
    # to override call() in its entirety
    def call_feature_extractor(self, inputs, **kwargs):
        return self.feature_extractor(inputs, **kwargs)

    def call_task_classifier(self, fe, **kwargs):
        return self.task_classifier(fe, **kwargs)

    def call_domain_classifier(self, fe, task, **kwargs):
        return self.domain_classifier(fe, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        self.set_learning_phase(training)
        fe = self.call_feature_extractor(inputs, **kwargs)
        task = self.call_task_classifier(fe, **kwargs)
        domain = self.call_domain_classifier(fe, task, **kwargs)
        return task, domain, fe


class DannModel(FcnModelBase):
    """ DANN adds a gradient reversal layer before the domain classifier """
    def __init__(self, num_classes, num_domains, global_step,
            total_steps, **kwargs):
        super().__init__(num_classes, num_domains, **kwargs)
        grl_schedule = DannGrlSchedule(total_steps)
        self.flip_gradient = FlipGradient(global_step, grl_schedule)

    def call_domain_classifier(self, fe, task, **kwargs):
        grl_output = self.flip_gradient(fe, **kwargs)
        return self.domain_classifier(grl_output, **kwargs)


class SleepModel(DannModel):
    """ Sleep model is DANN but concatenating task classifier output (with stop
    gradient) with feature extractor output when fed to the domain classifier """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.stop_gradient = StopGradient()

    def call_domain_classifier(self, fe, task, **kwargs):
        grl_output = self.flip_gradient(fe, **kwargs)
        task_stop_gradient = self.stop_gradient(task)
        domain_input = self.concat([grl_output, task_stop_gradient])
        return self.domain_classifier(domain_input, **kwargs)


class DannSmoothModel(DannModel):
    def __init__(self, *args, num_domain_classifiers, **kwargs):
        # For MDAN Smooth, it's binary classification but we have a separate
        # discriminator for each source-target pair.
        super().__init__(*args, **kwargs)

        # MDAN Smooth requires multiple domain classifiers, one for each source
        # domain. Assumes a single target domain.
        new_domain_classifier = [self.domain_classifier]

        # Start at 1 since we already have one
        for i in range(1, num_domain_classifiers):
            new_domain_classifier.append(
                tf.keras.models.clone_model(self.domain_classifier))

        self.domain_classifier = new_domain_classifier

    @property
    def trainable_variables_domain(self):
        # We have multiple domain classifiers, so get all variables
        domain_vars = []

        for dc in self.domain_classifier:
            domain_vars += dc.trainable_variables

        return domain_vars

    def call_feature_extractor(self, inputs, **kwargs):
        # Override so we don't pass domain_classifier argument to model
        return self.feature_extractor(inputs)

    def call_task_classifier(self, fe, **kwargs):
        # Override so we don't pass domain_classifier argument to model
        return self.task_classifier(fe)

    def call_domain_classifier(self, fe, task, domain_classifier=None, **kwargs):
        assert domain_classifier is not None, \
            "must specify which domain classifier to use with method Smooth"
        grl_output = self.flip_gradient(fe, **kwargs)
        # 0 = source domain 1 with target, 1 = source domain 2 with target, etc.
        return self.domain_classifier[domain_classifier](grl_output, **kwargs)


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
