"""
Generate t-SNE, PCA, and reconstruction plots to display in TensorBoard
"""
import io
import numpy as np
import tensorflow as tf

from absl import flags
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Make sure matplotlib is not interactive
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from datasets import inversions


FLAGS = flags.FLAGS

flags.DEFINE_integer("max_plot_embedding", 100, "Max points to plot in t-SNE and PCA plots (0 = skip these plots)")
flags.DEFINE_integer("max_plot_mapping", 5, "Max samples to plot in reconstruction/spectrogram plots (0 = skip these plots)")
flags.DEFINE_float("freq", 50.0, "Sampling frequency in Hz of accelerometers, etc. (for plotting spectrogram)")
flags.DEFINE_integer("nfft", 64, "NFFT for spectrogram, samples per FFT block")
flags.DEFINE_integer("noverlap", 63, "noverlap for spectrogram, overlap between subsequent windows for FFT")


def generate_plots(data_a, data_b, model, mapping_model, adapt, first_time,
        invert_name=None):
    """
    Run the first batch of evaluation data through the feature extractor, then
    generate and return the PCA and t-SNE plots. Optionally, save these to a file
    as well.
    """
    plots = []
    x_a, y_a = data_a

    if data_b is not None:
        x_b, y_b = data_b

    #
    # TSNE and PCA
    #
    if model is not None and FLAGS.max_plot_embedding > 0 and data_b is not None:
        emb_x_a = x_a[:FLAGS.max_plot_embedding]
        emb_x_b = x_b[:FLAGS.max_plot_embedding]
        emb_y_a = y_a[:FLAGS.max_plot_embedding]
        emb_y_b = y_b[:FLAGS.max_plot_embedding]

        # Source then target
        combined_x = tf.concat((emb_x_a, emb_x_b), axis=0)
        combined_labels = tf.concat((emb_y_a, emb_y_b), axis=0)
        source_domain = tf.zeros([tf.shape(emb_x_a)[0], 1], dtype=tf.int32)
        target_domain = tf.ones([tf.shape(emb_x_b)[0], 1], dtype=tf.int32)
        combined_domain = tf.concat((source_domain, target_domain), axis=0)

        # Run through model's feature extractor
        embedding = model.feature_extractor(combined_x, training=False)

        # If an RNN, get only the embedding, not the RNN state
        if isinstance(embedding, tuple):
            embedding = embedding[0]

        # Compute TSNE and PCA
        tsne = TSNE(n_components=2, init='pca', n_iter=3000).fit_transform(embedding)
        pca = PCA(n_components=2).fit_transform(embedding)

        if adapt:
            title = "Domain Adaptation"
        else:
            title = "No Adaptation"

        tsne_plot = plot_embedding(tsne, tf.argmax(combined_labels, axis=1),
            tf.squeeze(combined_domain), title=title + " - t-SNE")
        pca_plot = plot_embedding(pca, tf.argmax(combined_labels, axis=1),
            tf.squeeze(combined_domain), title=title + " - PCA")

        if tsne_plot is not None:
            plots.append(('tsne', tsne_plot))
        if pca_plot is not None:
            plots.append(('pca', pca_plot))

    #
    # Domain mapping
    #
    if mapping_model is not None and FLAGS.max_plot_mapping > 0 and data_b is not None:
        map_x_a = x_a[:FLAGS.max_plot_mapping]
        map_x_b = x_b[:FLAGS.max_plot_mapping]

        num_features_a = tf.shape(map_x_a)[2]
        num_features_b = tf.shape(map_x_b)[2]

        # Generators on original data
        gen_AtoB = mapping_model.map_to_target(map_x_a)
        gen_BtoA = mapping_model.map_to_source(map_x_b)

        # Generators on fake data to map back to original domain (a full cycle)
        gen_AtoBtoA = mapping_model.map_to_source(gen_AtoB)
        gen_BtoAtoB = mapping_model.map_to_target(gen_BtoA)

        if first_time and invert_name is not None:
            true_map_to_target = inversions.map_to_target[invert_name](map_x_a)
            true_map_to_source = inversions.map_to_source[invert_name](map_x_b)

        # Run model on data -- alternative, but slightly slower than the above
        # gen_AtoB, gen_AtoBtoA, _, _ = mapping_model(map_x_a, "target", training=False)
        # gen_BtoA, gen_BtoAtoB, _, _ = mapping_model(map_x_b, "source", training=False)

        for i in range(num_features_a):
            recon_plot = plot_real_time_series(
                gen_AtoBtoA[:, :, i],
                title='Reconstruction (A to B to A, feature '+str(i)+')')
            plots.append(('reconstruction_feature_'+str(i)+'/source', recon_plot))

            map_plot = plot_real_time_series(
                gen_BtoA[:, :, i],
                title='Mapped (B to A, feature '+str(i)+')')
            plots.append(('mapped_feature_'+str(i)+'/to_source', map_plot))

            map_fft_plot = plot_fft(
                gen_BtoA[:, :, i],
                title='FFT Mapped (B to A, feature '+str(i)+')')
            plots.append(('mapped_fft_feature_'+str(i)+'/to_source', map_fft_plot))

            if first_time:
                real_plot = plot_real_time_series(
                    map_x_a[:, :, i],
                    title='Real Data (domain A, feature '+str(i)+')')
                plots.append(('real_feature_'+str(i)+'/source', real_plot))

                real_fft_plot = plot_fft(
                    map_x_a[:, :, i],
                    title='FFT Real (domain A, feature '+str(i)+')')
                plots.append(('real_fft_feature_'+str(i)+'/source', real_fft_plot))

                if invert_name is not None:
                    true_map_plot = plot_real_time_series(
                        true_map_to_source[:, :, i],
                        title='True Map (domain A, feature '+str(i)+')')
                    plots.append(('true_map_feature_'+str(i)+'/to_source', true_map_plot))

                    true_map_fft_plot = plot_fft(
                        true_map_to_source[:, :, i],
                        title='True Map FFT (domain A, feature '+str(i)+')')
                    plots.append(('true_map_fft_feature_'+str(i)+'/to_source', true_map_fft_plot))

        for i in range(num_features_b):
            recon_plot = plot_real_time_series(
                gen_BtoAtoB[:, :, i],
                title='Reconstruction (B to A to B, feature '+str(i)+')')
            plots.append(('reconstruction_feature_'+str(i)+'/target', recon_plot))

            map_plot = plot_real_time_series(
                gen_AtoB[:, :, i],
                title='Mapped (A to B, feature '+str(i)+')')
            plots.append(('mapped_feature_'+str(i)+'/to_target', map_plot))

            map_fft_plot = plot_fft(
                gen_AtoB[:, :, i],
                title='FFT Mapped (A to B, feature '+str(i)+')')
            plots.append(('mapped_fft_feature_'+str(i)+'/to_target', map_fft_plot))

            if first_time:
                real_plot = plot_real_time_series(
                    map_x_b[:, :, i],
                    title='Real Data (domain B, feature '+str(i)+')')
                plots.append(('real_feature_'+str(i)+'/target', real_plot))

                real_fft_plot = plot_fft(
                    map_x_b[:, :, i],
                    title='FFT Real (domain B, feature '+str(i)+')')
                plots.append(('real_fft_feature_'+str(i)+'/target', real_fft_plot))

                if invert_name is not None:
                    true_map_plot = plot_real_time_series(
                        true_map_to_target[:, :, i],
                        title='True Map (domain B, feature '+str(i)+')')
                    plots.append(('true_map_feature_'+str(i)+'/to_target', true_map_plot))

                    true_map_fft_plot = plot_fft(
                        true_map_to_target[:, :, i],
                        title='True Map FFT (domain B, feature '+str(i)+')')
                    plots.append(('true_map_fft_feature_'+str(i)+'/to_target', true_map_fft_plot))

    return plots


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.

    See: https://www.tensorflow.org/tensorboard/r2/image_summaries
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


def plot_embedding(x, y, d, title=None, filename=None):
    """
    Plot an embedding X with the class label y colored by the domain d.

    From: https://github.com/pumpikano/tf-dann/blob/master/utils.py
    """
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    # We'd get an error if nan or inf
    if np.isnan(x).any() or np.isinf(x).any():
        return None

    # XKCD colors: https://matplotlib.org/users/colors.html
    colors = {
        0: 'xkcd:orange',  # source
        1: 'xkcd:darkgreen',  # target
    }

    domain = {
        0: 'S',
        1: 'T',
    }

    # Plot colors numbers
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(111)
    for i in range(x.shape[0]):
        # plot colored number
        plt.text(x[i, 0], x[i, 1], domain[d[i].numpy()]+str(y[i].numpy()),
                 color=colors[d[i].numpy()],
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)

    return plot_to_image(fig)


def plot_random_time_series(mu, sigma, title=None, filename=None):
    """
    Using the mu and sigma given at each time step, generate sample time-series
    using these Gaussian parameters learned by the VRNN

    Input:
        mu, sigma -- each time step, learned in VRNN,
            each shape: [batch_size, time_steps, num_features]
        num_samples -- how many lines/curves you want to plot
        title, filename -- optional
    Output:
        plot of sample time-series

    Note: at the moment we're assuming num_features=1 (plot will be 2D)
    """
    mu = np.squeeze(mu)
    sigma = np.squeeze(sigma)
    length = mu.shape[1]
    num_samples = mu.shape[0]

    # x axis is just 0, 1, 2, 3, ...
    x = np.arange(length)

    # y is values sampled from mu and sigma
    y = sigma*np.random.normal(0, 1, (num_samples, length)) + mu

    fig = plt.figure()
    for i in range(y.shape[0]):
        plt.plot(x, y[i, :])

    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)

    return plot_to_image(fig)


def plot_real_time_series(y, title=None, filename=None):
    """
    Plot the real time-series data for comparison with the reconstruction

    Input:
        input x values (real data)
            shape: [batch_size, time_steps, num_features]
        num_samples -- how many lines/curves you want to plot
        title, filename -- optional
    Output:
        plot of input time-series

    Note: at the moment we're assuming num_features=1 (plot will be 2D)
    """
    y = np.squeeze(y)
    length = y.shape[1]

    # x axis is just 0, 1, 2, 3, ...
    x = np.arange(length)

    fig = plt.figure()
    for i in range(y.shape[0]):
        plt.plot(x, y[i, :])

    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)

    return plot_to_image(fig)


def hide_border(ax):
    """ Hide the four borders of a plot """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_fft(data, title=None, filename=None):
    """
    Create the FFT figure
    data.shape = (num_examples, time_step) since this is called once per feature
    """
    num_examples = data.shape[0]
    time_steps = data.shape[1]
    t = np.arange(0.0, time_steps/FLAGS.freq, 1/FLAGS.freq)
    # For some reason sometimes t ends up being one more sample than data
    t = t[:time_steps]

    fig, axes = plt.subplots(nrows=num_examples, sharex=True,
        figsize=(5, 2*num_examples))

    if title is not None:
        plt.suptitle(title)
        #fig.canvas.set_window_title(title)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
        wspace=0.2, hspace=0.1)

    # For each of the FLAGS.max_plot_mapping
    for i, y in enumerate(data):
        y = np.array(y)

        Pxx, freqs, bins, im = axes[i].specgram(y, NFFT=FLAGS.nfft,
            Fs=FLAGS.freq, noverlap=FLAGS.noverlap, mode="psd",
            xextent=(t[0], t[-1]))
        hide_border(axes[i])
        axes[i].margins(x=0)

        # Only bottom one
        if i == num_examples-1:
            axes[i].set_xlabel('seconds')

        #axes[i].set_ylabel("psd")

        # See: https://stackoverflow.com/a/49037495
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    return plot_to_image(fig)
