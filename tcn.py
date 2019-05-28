"""
Temporal Convolutional Networks (alternative to RNNs)

Paper: https://arxiv.org/pdf/1803.01271.pdf
Description: https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-3-7f6633fcc7c7
TF code from: https://colab.research.google.com/drive/1la33lW7FQV1RicpfzyLq9H0SH1VSD4LE#scrollTo=xdTffJqQy-gU
but modified for TensorFlow 2.0

Usage:
    from tcn import TemporalConvNet
    x = tf.random_normal((32, 10, 4)) # (batch_size, length, channel)
    tcn = TemporalConvNet([8, 8, 8, 8], 2, 0.25, return_sequences=True)
    output = tcn(x, training=tf.constant(True))
    last_output = output[:, -1, :]
"""
import tensorflow as tf


class CausalConv1D(tf.keras.layers.Conv1D):
    def __init__(self, filters,
               kernel_size,
               strides=1,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
        )

    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(tensor=inputs, paddings=tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(CausalConv1D, self).call(inputs)


class TemporalBlock(tf.keras.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=tf.nn.relu,
            name="conv1")
        self.conv2 = CausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=tf.nn.relu,
            name="conv2")
        self.down_sample = None

    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = tf.keras.layers.Dropout(self.dropout,
            [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.keras.layers.Dropout(self.dropout,
            [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        if input_shape[channel_dim] != self.n_outputs:
            # self.down_sample = tf.keras.layers.Conv1D(
            #     self.n_outputs, kernel_size=1,
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = tf.keras.layers.Dense(self.n_outputs, activation=None)
        self.built = True

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.layernorm1(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.layernorm2(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)


class TemporalConvNet(tf.keras.layers.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None,
                 return_sequences=False, **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.return_sequences = return_sequences
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(out_channels, kernel_size, strides=1, dilation_rate=dilation_size,
                              dropout=dropout, name="tblock_{}".format(i))
            )

    def call(self, inputs, training=None):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)

        if self.return_sequences:
            return outputs
        else:
            return outputs[:, -1, :]
