"""
Models
"""
import tensorflow as tf

from absl import flags
from vrnn import VRNN

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


class ModelBase(tf.keras.Model):
    """ Base model class (inheriting from Keras' Model class) """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def trainable_variables_fe(self):
        return self.feature_extractor.trainable_variables

    @property
    def trainable_variables_task(self):
        return self.trainable_variables_fe \
            + self.task_classifier.trainable_variables

    @property
    def trainable_variables_domain(self):
        return self.domain_classifier.trainable_variables

    @property
    def trainable_variables_task_domain(self):
        return self.trainable_variables_fe \
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


class FcnModelBase(ModelBase):
    """
    FCN (fully CNN) -- but domain classifier has additional dense layers

    From: https://arxiv.org/pdf/1611.06455.pdf
    Tested in: https://arxiv.org/pdf/1809.04356.pdf
    Code from: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
    """
    def __init__(self, num_classes, num_domains, **kwargs):
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


class HeterogeneousDannModel(DannModel):
    """ Heterogeneous DANN model has multiple feature extractors,
    very similar to DannSmoothModel() code except this has multiple FE's
    not multiple DC's """
    def __init__(self, *args, num_feature_extractors, **kwargs):
        super().__init__(*args, **kwargs)

        # Requires multiple feature extractors
        new_feature_extractor = [self.feature_extractor]

        # Start at 1 since we already have one
        for i in range(1, num_feature_extractors):
            new_feature_extractor.append(
                tf.keras.models.clone_model(self.feature_extractor))

        self.feature_extractor = new_feature_extractor

    @property
    def trainable_variables_fe(self):
        # We have multiple feature extractors, so get all variables
        fe_vars = []

        for fe in self.feature_extractor:
            fe_vars += fe.trainable_variables

        return fe_vars

    def call_feature_extractor(self, inputs, which_fe=None, **kwargs):
        # Override so we don't pass which_fe argument to model
        return self.feature_extractor(inputs)

        assert which_fe is not None, \
            "must specify which feature extractor to use"
        return self.feature_extractor[which_fe](inputs, **kwargs)

    def call_task_classifier(self, fe, which_fe=None, **kwargs):
        # Override so we don't pass which_fe argument to model
        return self.task_classifier(fe, **kwargs)

    def call_domain_classifier(self, fe, task, which_fe=None, **kwargs):
        # Override so we don't pass which_fe argument to model
        # Copy of the DANN version only with above arg change
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
    """ DANN Smooth model hs multiple domain classifiers,
    very similar to HeterogeneousDannModel() code except this has multiple DC's
    not multiple FE's """
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

    def call_feature_extractor(self, inputs, which_dc=None, **kwargs):
        # Override so we don't pass which_dc argument to model
        return self.feature_extractor(inputs, **kwargs)

    def call_task_classifier(self, fe, which_dc=None, **kwargs):
        # Override so we don't pass which_dc argument to model
        return self.task_classifier(fe, **kwargs)

    def call_domain_classifier(self, fe, task, which_dc=None, **kwargs):
        assert which_dc is not None, \
            "must specify which domain classifier to use with method Smooth"
        grl_output = self.flip_gradient(fe, **kwargs)
        # 0 = source domain 1 with target, 1 = source domain 2 with target, etc.
        return self.domain_classifier[which_dc](grl_output, **kwargs)


class VradaFeatureExtractor(tf.keras.Model):
    """
    Need to get VRNN state, so we can't directly use Sequential since it can't
    return intermediate layer's extra outputs. And, can't use the functional
    API directly since we don't now the input shape.

    Note: only returns state if vrada=True
    """
    def __init__(self, vrada=True, **kwargs):
        super().__init__(**kwargs)
        self.vrada = vrada

        if self.vrada:
            # Use z for predictions in VRADA like in original paper
            self.rnn = VRNN(100, 100, return_z=True, return_sequences=False)
        else:
            self.rnn = tf.keras.layers.LSTM(100, return_sequences=False)

        self.fe = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(100),
        ])

    def call(self, inputs, **kwargs):
        if self.vrada:
            rnn_output, rnn_state = self.rnn(inputs, **kwargs)
        else:
            rnn_output = self.rnn(inputs, **kwargs)
            rnn_state = None

        fe_output = self.fe(rnn_output, **kwargs)

        return fe_output, rnn_state


class RnnModelBase(ModelBase):
    """ RNN-based model - for R-DANN and VRADA """
    def __init__(self, vrada, num_classes, num_domains, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.feature_extractor = VradaFeatureExtractor(vrada)
        self.task_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(num_classes),
        ])
        self.domain_classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(num_domains),
        ])

    def call(self, inputs, training=None, **kwargs):
        """ Since our RNN feature extractor returns two values (output and
        RNN state, which we need for the loss) we need to only pass the output
        to the classifiers, i.e. fe[0] rather than fe """
        self.set_learning_phase(training)
        fe = self.call_feature_extractor(inputs, **kwargs)
        task = self.call_task_classifier(fe[0], **kwargs)
        domain = self.call_domain_classifier(fe[0], task, **kwargs)
        return task, domain, fe


class DannRnnModel(RnnModelBase):
    """ DannModel but for RnnModelBase not FcnModelBase """
    def __init__(self, vrada, num_classes, num_domains, global_step,
            total_steps, **kwargs):
        super().__init__(vrada, num_classes, num_domains, **kwargs)
        grl_schedule = DannGrlSchedule(total_steps)
        self.flip_gradient = FlipGradient(global_step, grl_schedule)

    def call_domain_classifier(self, fe, task, **kwargs):
        grl_output = self.flip_gradient(fe, **kwargs)
        return self.domain_classifier(grl_output, **kwargs)


class VradaModel(DannRnnModel):
    def __init__(self, *args, **kwargs):
        vrada = True
        super().__init__(vrada, *args, **kwargs)


class RDannModel(DannRnnModel):
    def __init__(self, *args, **kwargs):
        vrada = False
        super().__init__(vrada, *args, **kwargs)
