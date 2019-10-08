"""
Generate t-SNE, PCA, and reconstruction plots to display in TensorBoard
"""
import io
import numpy as np
import tensorflow as tf

from absl import flags
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Make sure matplotlib is not interactive
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS

flags.DEFINE_integer("max_plot_embedding", 15, "Max points to plot in t-SNE and PCA plots (0 = skip these plots)")


def generate_plots(data_a, data_b, feature_extractor, first_time):
    """
    Run the first batch of evaluation data through the feature extractor, then
    generate and return the PCA and t-SNE plots. Optionally, save these to a file
    as well.

    Note: data_a should be a tuple of lists (since there may be multiple source
    domains)
    """
    plots = []
    x_a, y_a, domain_a = data_a

    if data_b is not None:
        x_b, y_b, domain_b = data_b

    #
    # TSNE and PCA
    #
    if feature_extractor is not None and FLAGS.max_plot_embedding > 0 and data_b is not None:
        # Take a few of the first data from each domain to plot
        num_source_domains = len(x_a)
        assert len(y_a) == num_source_domains
        assert len(domain_a) == num_source_domains

        emb_x_a = []
        emb_y_a = []
        emb_d_a = []

        for i in range(num_source_domains):
            emb_x_a.append(x_a[i][:FLAGS.max_plot_embedding])
            emb_y_a.append(y_a[i][:FLAGS.max_plot_embedding])
            emb_d_a.append(domain_a[i][:FLAGS.max_plot_embedding])

        emb_x_a = tf.concat(emb_x_a, axis=0)
        emb_y_a = tf.concat(emb_y_a, axis=0)
        emb_d_a = tf.concat(emb_d_a, axis=0)

        emb_x_b = x_b[:FLAGS.max_plot_embedding]
        emb_y_b = y_b[:FLAGS.max_plot_embedding]
        emb_d_b = domain_b[:FLAGS.max_plot_embedding]

        # Source then target
        combined_x = tf.concat((emb_x_a, emb_x_b), axis=0)
        combined_labels = tf.concat((emb_y_a, emb_y_b), axis=0)
        combined_domain = tf.concat((emb_d_a, emb_d_b), axis=0)

        # Run through model's feature extractor
        embedding = feature_extractor(combined_x, training=False)

        # If an RNN, get only the embedding, not the RNN state
        if isinstance(embedding, tuple):
            embedding = embedding[0]

        # Compute TSNE and PCA
        tsne = TSNE(n_components=2, init='pca', n_iter=3000).fit_transform(embedding)
        pca = PCA(n_components=2).fit_transform(embedding)

        tsne_plot = plot_embedding(tsne, tf.squeeze(combined_labels),
            tf.squeeze(combined_domain), title="t-SNE")
        pca_plot = plot_embedding(pca, tf.squeeze(combined_labels),
            tf.squeeze(combined_domain), title="PCA")

        if tsne_plot is not None:
            plots.append(('tsne', tsne_plot))
        if pca_plot is not None:
            plots.append(('pca', pca_plot))

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

    # Plot colors numbers
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(111)
    for i in range(x.shape[0]):
        # source or target - default to target
        # XKCD colors: https://matplotlib.org/users/colors.html
        text = "T_"
        color = "xkcd:darkgreen"

        # if source
        domain = int(d[i].numpy())
        if domain != 0:
            text = "S"+str(domain) + "_"
            color = "xkcd:orange"

        # label number
        text += str(int(y[i].numpy()))

        # plot colored number
        plt.text(x[i, 0], x[i, 1], text, color=color,
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    if title is not None:
        plt.title(title)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)

    return plot_to_image(fig)
