"""
TensorFlow 2.0 implementation of VRNN

Based on my 1.x implementation:
https://github.com/floft/deep-activity-learning/blob/tf_1.x/vrnn.py
"""
import tensorflow as tf


class VRNN(tf.keras.layers.Layer):
    """ Wrap VRNNCell into a RNN """
    def __init__(self, h_dim, z_dim, return_z=True, return_sequences=False,
            go_backwards=False, stateful=False, unroll=False, **kwargs):
        super().__init__(**kwargs)
        self.return_z = return_z
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

    def build(self, input_shape):
        num_features = input_shape[-1]
        cell = VRNNCell(num_features, self.h_dim, self.z_dim)
        # We return sequences here so we can compute VRNN reconstruction loss
        self.rnn = tf.keras.layers.RNN(cell,
            return_sequences=True,
            return_state=False, go_backwards=self.go_backwards,
            stateful=self.stateful, unroll=self.unroll)

    def call(self, inputs, **kwargs):
        outputs = self.rnn(inputs, **kwargs)

        h, c, \
            encoder_mu, encoder_sigma, \
            decoder_mu, decoder_sigma, \
            prior_mu, prior_sigma, \
            x_1, z_1 = outputs

        # VRADA uses z not h
        if self.return_z:
            rnn_output = z_1
        else:
            rnn_output = h

        # Get the output at the end of the sequence
        if not self.return_sequences:
            rnn_output = rnn_output[:, -1]

        # For use in loss, note these are return_sequences=True
        other_outputs = [encoder_mu, encoder_sigma, decoder_mu, decoder_sigma,
            prior_mu, prior_sigma]

        return rnn_output, other_outputs


class VRNNCell(tf.keras.layers.Layer):
    """
    VRNN cell implementation for use in VRADA

    Based on:
    - https://github.com/phreeza/tensorflow-vrnn/blob/master/model_vrnn.py
    - https://github.com/kimkilho/tensorflow-vrnn/blob/master/cell.py
    - https://github.com/kimkilho/tensorflow-vrnn/blob/master/main.py
    - https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/rnn_cell_impl.py
    """
    def __init__(self, x_dim, h_dim, z_dim, **kwargs):
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Dimensions of x input, hidden layers, latent variable (z)
        self.n_x = self.x_dim
        self.n_h = self.h_dim
        self.n_z = self.z_dim

        # Dimensions of phi(z)
        self.n_x_1 = self.x_dim
        self.n_z_1 = self.z_dim

        # Dimensions of encoder, decoder, and prior
        self.n_enc_hidden = self.z_dim
        self.n_dec_hidden = self.x_dim
        self.n_prior_hidden = self.z_dim

        # Note: first two are the state of the LSTM
        self.state_size = (
            self.n_h, self.n_h,
            self.n_z, self.n_z,
            self.n_x, self.n_x,
            self.n_z, self.n_z,
            self.n_x_1, self.n_z_1)

        # What cell we're going to use internally for the RNN
        self.cell = tf.keras.layers.LSTMCell(h_dim)

        super().__init__(**kwargs)

    def build(self, input_shape):
        # Input: previous hidden state
        self.prior_h = self.add_weight("prior/hidden/weights",
            shape=(self.n_h, self.n_prior_hidden), initializer="glorot_uniform")
        self.prior_mu = self.add_weight("prior/mu/weights",
            shape=(self.n_prior_hidden, self.n_z), initializer="glorot_uniform")
        self.prior_sigma = self.add_weight("prior/sigma/weights",
            shape=(self.n_prior_hidden, self.n_z), initializer="glorot_uniform")

        self.prior_h_b = self.add_weight("prior/hidden/bias",
            shape=(self.n_prior_hidden,), initializer=tf.constant_initializer())
        self.prior_sigma_b = self.add_weight("prior/sigma/bias",
            shape=(self.n_z,), initializer=tf.constant_initializer())
        self.prior_mu_b = self.add_weight("prior/mu/bias",
            shape=(self.n_z,), initializer=tf.constant_initializer())

        # Input: x
        self.x_1 = self.add_weight("phi_x/weights",
            shape=(self.n_x, self.n_x_1), initializer="glorot_uniform")
        self.x_1_b = self.add_weight("phi_x/bias",
            shape=(self.n_x_1,), initializer=tf.constant_initializer())

        # Input: x and previous hidden state
        self.encoder_h = self.add_weight("encoder/hidden/weights",
            shape=(self.n_x_1+self.n_h, self.n_enc_hidden), initializer="glorot_uniform")
        self.encoder_mu = self.add_weight("encoder/mu/weights",
            shape=(self.n_enc_hidden, self.n_z), initializer="glorot_uniform")
        self.encoder_sigma = self.add_weight("encoder/sigma/weights",
            shape=(self.n_enc_hidden, self.n_z), initializer="glorot_uniform")

        self.encoder_h_b = self.add_weight("encoder/hidden/bias",
            shape=(self.n_enc_hidden,), initializer=tf.constant_initializer())
        self.encoder_sigma_b = self.add_weight("encoder/sigma/bias",
            shape=(self.n_z,), initializer=tf.constant_initializer())
        self.encoder_mu_b = self.add_weight("encoder/mu/bias",
            shape=(self.n_z,), initializer=tf.constant_initializer())

        # Input: z = enc_sigma*eps + enc_mu -- i.e. reparameterization trick
        self.z_1 = self.add_weight("phi_z/weights",
            shape=(self.n_z, self.n_z_1), initializer="glorot_uniform")
        self.z_1_b = self.add_weight("phi_z/bias",
            shape=(self.n_z_1,), initializer=tf.constant_initializer())

        # Input: latent variable (z) and previous hidden state
        self.decoder_h = self.add_weight("decoder/hidden/weights",
            shape=(self.n_z+self.n_h, self.n_dec_hidden), initializer="glorot_uniform")
        self.decoder_mu = self.add_weight("decoder/mu/weights",
            shape=(self.n_dec_hidden, self.n_x), initializer="glorot_uniform")
        self.decoder_sigma = self.add_weight("decoder/sigma/weights",
            shape=(self.n_dec_hidden, self.n_x), initializer="glorot_uniform")

        self.decoder_h_b = self.add_weight("decoder/hidden/bias",
            shape=(self.n_dec_hidden,), initializer=tf.constant_initializer())
        self.decoder_sigma_b = self.add_weight("decoder/sigma/bias",
            shape=(self.n_x,), initializer=tf.constant_initializer())
        self.decoder_mu_b = self.add_weight("decoder/mu/bias",
            shape=(self.n_x,), initializer=tf.constant_initializer())

    def call(self, inputs, states, **kwargs):
        # Get relevant states
        h = states[0]
        c = states[1]  # only passed to the LSTM

        # Input: previous hidden state (h)
        prior_h = tf.nn.relu(tf.matmul(h, self.prior_h) + self.prior_h_b)
        prior_sigma = tf.nn.softplus(tf.matmul(prior_h, self.prior_sigma) + self.prior_sigma_b)  # >= 0
        prior_mu = tf.matmul(prior_h, self.prior_mu) + self.prior_mu_b

        # Input: x
        # TODO removed ReLU since in the dataset not all x values are positive
        x_1 = tf.matmul(inputs, self.x_1) + self.x_1_b

        # Input: x and previous hidden state
        encoder_input = tf.concat((x_1, h), 1)
        encoder_h = tf.nn.relu(tf.matmul(encoder_input, self.encoder_h) + self.encoder_h_b)
        encoder_sigma = tf.nn.softplus(tf.matmul(encoder_h, self.encoder_sigma) + self.encoder_sigma_b)
        encoder_mu = tf.matmul(encoder_h, self.encoder_mu) + self.encoder_mu_b

        # Input: z = enc_sigma*eps + enc_mu -- i.e. reparameterization trick
        batch_size = tf.shape(inputs)[0]
        eps = tf.keras.backend.random_normal((batch_size, self.n_z), dtype=tf.float32)
        z = encoder_sigma*eps + encoder_mu
        z_1 = tf.nn.relu(tf.matmul(z, self.z_1) + self.z_1_b)

        # Input: latent variable (z) and previous hidden state
        decoder_input = tf.concat((z_1, h), 1)
        decoder_h = tf.nn.relu(tf.matmul(decoder_input, self.decoder_h) + self.decoder_h_b)
        decoder_sigma = tf.nn.softplus(tf.matmul(decoder_h, self.decoder_sigma) + self.decoder_sigma_b)
        decoder_mu = tf.matmul(decoder_h, self.decoder_mu) + self.decoder_mu_b

        # Pass to cell (e.g. LSTM). Note that the LSTM has both "h" and "c" that are combined
        # into the same next state vector. We'll combine them together to pass in and split them
        # back out after the LSTM returns the next state.
        rnn_cell_input = tf.concat((x_1, z_1), 1)
        _, (h_next, c_next) = self.cell(rnn_cell_input, [h, c])  # Note: (h,c) in Keras (c,h) in tf contrib

        # VRNN state
        next_state = (
            h_next,
            c_next,
            encoder_mu,
            encoder_sigma,
            decoder_mu,
            decoder_sigma,
            prior_mu,
            prior_sigma,
            x_1,
            z_1,
        )

        #return output, next_state
        return next_state, next_state
