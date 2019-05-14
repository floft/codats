"""
Models

Primarily we can't be using just sequential because of the grl_lambda needing
to be passed around. Also residual connections need to be a separate custom
layer.

Provides the model DomainAdaptationModel() and its components along with the
make_{task,domain}_loss() functions. Also, compute_accuracy() if desired.
"""
import tensorflow as tf

from absl import flags
from tensorflow.python.keras import backend as K

FLAGS = flags.FLAGS

flags.DEFINE_float("dropout", 0.05, "Dropout probability")

flags.register_validator("dropout", lambda v: v != 1, message="dropout cannot be 1")


@tf.custom_gradient
def flip_gradient(x, grl_lambda=1.0):
    """ Forward pass identity, backward pass negate gradient and multiply by  """
    grl_lambda = tf.cast(grl_lambda, dtype=tf.float32)

    def grad(dy):
        return tf.negative(dy) * grl_lambda * tf.ones_like(x)

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


def make_dense_bn_dropout(units, dropout):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(dropout),
    ])


class ResnetBlock(tf.keras.layers.Layer):
    """ Block consisting of other blocks but with residual connections """
    def __init__(self, units, dropout, layers, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [make_dense_bn_dropout(units, dropout) for _ in range(layers)]
        self.add = tf.keras.layers.Add()

    def call(self, inputs, **kwargs):
        """ Like Sequential but with a residual connection """
        shortcut = inputs
        net = inputs

        for block in self.blocks:
            net = block(net, **kwargs)

        return self.add([shortcut, net], **kwargs)


def make_vrada_model(num_classes, global_step, grl_schedule):
    """
    Create model inspired by the VRADA paper model for time-series data

    Note: VRADA model had a VRNN though rather than just flattening data and
    didn't use residual connections.
    """
    fe_layers = 5
    task_layers = 1
    domain_layers = 2
    resnet_layers = 2
    units = 50
    dropout = FLAGS.dropout

    # General classifier used in both the task/domain classifiers
    def make_classifier(layers, num_outputs):
        layers = [make_dense_bn_dropout(units, dropout) for _ in range(layers-1)]
        last = [
            tf.keras.layers.Dense(num_outputs),
            tf.keras.layers.Activation("softmax"),
        ]
        return tf.keras.Sequential(layers + last)

    def make_binary_classifier(layers):
        layers = [make_dense_bn_dropout(units, dropout) for _ in range(layers-1)]
        last = [tf.keras.layers.Dense(1)]
        return tf.keras.Sequential(layers + last)

    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(momentum=0.999),
    ] + [  # First can't be residual since x isn't of size units
        make_dense_bn_dropout(units, dropout) for _ in range(resnet_layers)
    ] + [
        ResnetBlock(units, dropout, resnet_layers) for _ in range(fe_layers-1)
    ])
    task_classifier = tf.keras.Sequential([
        make_classifier(task_layers, num_classes),
    ])
    domain_classifier = tf.keras.Sequential([
        FlipGradient(global_step, grl_schedule),
        make_binary_classifier(domain_layers),
    ])
    return feature_extractor, task_classifier, domain_classifier


def make_dann_mnist_model(num_classes, global_step, grl_schedule):
    """ Figure 4(a) MNIST architecture -- Ganin et al. DANN JMLR 2016 paper """
    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), (1, 1), "valid", activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2), "valid"),
        tf.keras.layers.Conv2D(48, (5, 5), (1, 1), "valid", activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2), "valid"),
        tf.keras.layers.Flatten(),
    ])
    task_classifier = tf.keras.Sequential([
        tf.keras.layers.Dense(100, "relu"),
        tf.keras.layers.Dense(100, "relu"),
        tf.keras.layers.Dense(num_classes, "softmax"),
    ])
    domain_classifier = tf.keras.Sequential([
        FlipGradient(global_step, grl_schedule),
        tf.keras.layers.Dense(100, "relu"),
        tf.keras.layers.Dense(1),
    ])
    return feature_extractor, task_classifier, domain_classifier


def make_dann_svhn_model(num_classes, global_step, grl_schedule):
    """ Figure 4(b) SVHN architecture -- Ganin et al. DANN JMLR 2016 paper """
    dropout = FLAGS.dropout

    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (5, 5), (1, 1), "same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPool2D((3, 3), (2, 2), "same"),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Conv2D(64, (5, 5), (1, 1), "same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPool2D((3, 3), (2, 2), "same"),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Conv2D(128, (5, 5), (1, 1), "same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Flatten(),
    ])
    task_classifier = tf.keras.Sequential([
        tf.keras.layers.Dense(3072),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(2048),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(num_classes, "softmax"),
    ])
    domain_classifier = tf.keras.Sequential([
        FlipGradient(global_step, grl_schedule),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(1024),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(1),
    ])
    return feature_extractor, task_classifier, domain_classifier


def make_dann_gtsrb_model(num_classes, global_step, grl_schedule):
    """ Figure 4(c) SVHN architecture -- Ganin et al. DANN JMLR 2016 paper """
    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Conv2D(96, (5, 5), (1, 1), "valid", activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2), "valid"),
        tf.keras.layers.Conv2D(144, (3, 3), (1, 1), "valid", activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2), "valid"),
        tf.keras.layers.Conv2D(256, (5, 5), (1, 1), "valid", activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2), (2, 2), "valid"),
        tf.keras.layers.Flatten(),
    ])
    task_classifier = tf.keras.Sequential([
        tf.keras.layers.Dense(512, "relu"),
        tf.keras.layers.Dense(num_classes, "softmax"),
    ])
    domain_classifier = tf.keras.Sequential([
        FlipGradient(global_step, grl_schedule),
        tf.keras.layers.Dense(1024, "relu"),
        tf.keras.layers.Dense(1024, "relu"),
        tf.keras.layers.Dense(1),
    ])
    return feature_extractor, task_classifier, domain_classifier


def make_vada_model(num_classes, global_step, grl_schedule,
        small=False):
    """ Table 6 Small CNN -- Shu et al. VADA / DIRT-T ICLR 2018 paper

    Note: they used small for digits, traffic signs, and WiFi and large for
    CIFAR-10 and STL-10."""
    leak_alpha = 0.1

    def conv_blocks(depth):
        return [
            tf.keras.layers.Conv2D(depth, (3, 3), (1, 1), "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(leak_alpha),

            tf.keras.layers.Conv2D(depth, (3, 3), (1, 1), "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(leak_alpha),

            tf.keras.layers.Conv2D(depth, (3, 3), (1, 1), "same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(leak_alpha),
        ]

    def pool_blocks():
        return [
            tf.keras.layers.MaxPool2D((2, 2), (2, 2), "same"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.GaussianNoise(1),
        ]

    feature_extractor = tf.keras.Sequential(
        conv_blocks(64 if small else 96)
        + pool_blocks()
        + conv_blocks(64 if small else 192)
        + pool_blocks())
    task_classifier = tf.keras.Sequential(
        conv_blocks(64 if small else 192)
        + [
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, "softmax"),
        ])
    domain_classifier = tf.keras.Sequential([
        FlipGradient(global_step, grl_schedule),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(100),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(1),
    ])
    return feature_extractor, task_classifier, domain_classifier


def make_resnet50_model(num_classes, global_step, grl_schedule):
    """ ResNet50 pre-trained on ImageNet -- for use with Office-31 datasets
    Input should be 224x224x3 """
    feature_extractor = tf.keras.applications.ResNet50(
        include_top=False, pooling="avg")
    task_classifier = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, "softmax"),
    ])
    domain_classifier = tf.keras.Sequential([
        FlipGradient(global_step, grl_schedule),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),
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
            task_y_pred, domain_y_pred = model(x, training=True)
            ...
    """
    def __init__(self, num_classes, model_name, global_step,
            num_steps, use_grl=False, **kwargs):
        super().__init__(**kwargs)
        if use_grl:
            grl_schedule = DannGrlSchedule(num_steps)
            #grl_schedule = ConstantGrlSchedule(0.01)  # Possibly for VADA
        else:
            grl_schedule = DisableGrlSchedule()

        args = (num_classes, global_step, grl_schedule)

        if model_name == "flat":
            fe, task, domain = make_vrada_model(*args)
        elif model_name == "dann_mnist":
            fe, task, domain = make_dann_mnist_model(*args)
        elif model_name == "dann_svhn":
            fe, task, domain = make_dann_svhn_model(*args)
        elif model_name == "dann_gtsrb":
            fe, task, domain = make_dann_gtsrb_model(*args)
        elif model_name == "vada_small":
            fe, task, domain = make_vada_model(*args, small=True)
        elif model_name == "vada_large":
            fe, task, domain = make_vada_model(*args, small=False)
        elif model_name == "resnet50":
            fe, task, domain = make_resnet50_model(*args)
        else:
            raise NotImplementedError("Model name: "+str(model_name))

        self.feature_extractor = fe
        self.task_classifier = task
        self.domain_classifier = domain

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
    def trainable_variables_task_domain(self):
        return self.feature_extractor.trainable_variables \
            + self.task_classifier.trainable_variables \
            + self.domain_classifier.trainable_variables

    @property
    def trainable_variables_target(self):
        return self.feature_extractor.trainable_variables \
            + self.target_classifier.trainable_variables

    def call(self, inputs, target=False, training=None, **kwargs):
        # Manually set the learning phase since we probably aren't using .fit()
        if training is True:
            tf.keras.backend.set_learning_phase(1)
        elif training is False:
            tf.keras.backend.set_learning_phase(0)

        fe = self.feature_extractor(inputs, **kwargs)
        domain = self.domain_classifier(fe, **kwargs)

        # If desired, use the target classifier rather than the task classifier
        if target:
            task = self.target_classifier(fe, **kwargs)
        else:
            task = self.task_classifier(fe, **kwargs)

        return task, domain


def make_task_loss(adapt):
    """
    The same as CategoricalCrossentropy() but only on half the batch if doing
    adaptation and in the training phase
    """
    cce = tf.keras.losses.CategoricalCrossentropy()

    def task_loss(y_true, y_pred, training=None):
        """
        Compute loss on the outputs of the task classifier

        Note: domain classifier can use normal tf.keras.losses.CategoricalCrossentropy
        but for the task loss when doing adaptation we need to ignore the second half
        of the batch since this is unsupervised
        """
        if training is None:
            training = K.learning_phase()

        # If doing domain adaptation, then we'll need to ignore the second half of the
        # batch for task classification during training since we don't know the labels
        # of the target data
        if adapt and training:
            batch_size = tf.shape(y_pred)[0]
            y_pred = tf.slice(y_pred, [0, 0], [batch_size // 2, -1])
            y_true = tf.slice(y_true, [0, 0], [batch_size // 2, -1])

        return cce(y_true, y_pred)

    return task_loss


def make_weighted_loss():
    """ The same as CategoricalCrossentropy() but weighted """
    cce = tf.keras.losses.CategoricalCrossentropy()

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
        # from_logits=True means we didn't pass the Dense(1) layer through any
        # activation function like sigmoid. If we need the "probability" later,
        # then we'll have to manually pass it through a sigmoid function.
        cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        def domain_loss(y_true, y_pred):
            """ Compute loss on the outputs of the domain classifier """
            return cce(y_true, y_pred)
    else:
        def domain_loss(y_true, y_pred):
            """ Domain loss only used during adaptation """
            return 0

    return domain_loss


def compute_accuracy(y_true, y_pred):
    return tf.reduce_mean(input_tensor=tf.cast(
        tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)),
        tf.float32))


# List of names
models = [
    "flat",
    "dann_mnist",
    "dann_svhn",
    "dann_gtsrb",
    "vada_small",
    "vada_large",
    "resnet50",
]


# Get names
def names():
    """
    Returns list of all the available models for use in DomainAdaptationModel()
    """
    return models
