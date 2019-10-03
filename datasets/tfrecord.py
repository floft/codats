"""
Functions to write the x,y data to a tfrecord file
"""
import tensorflow as tf


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_example(x, y):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'x': _bytes_feature(tf.io.serialize_tensor(x)),
        'y': _bytes_feature(tf.io.serialize_tensor(y)),
    }))
    return tf_example


def write_tfrecord(filename, x, y):
    """ Output to TF record file """
    assert len(x) == len(y)
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for i in range(len(x)):
            tf_example = create_tf_example(x[i], y[i])
            writer.write(tf_example.SerializeToString())


def tfrecord_filename(dataset_name, train_or_test):
    """
    Version of tfrecord_filename ignoring the pairs and just creating a
    separate file for each domain. This works if there's no changes in the
    data based on the pairing (e.g. no resizing to match image dimensions)
    """
    # Sanity checks
    assert train_or_test in ["train", "valid", "test"], \
        "train_or_test must be train, valid, or test"

    filename = "%s_%s.tfrecord"%(dataset_name, train_or_test)

    return filename
