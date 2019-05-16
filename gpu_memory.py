"""
Set max GPU memory usage (if GPUs are found)

This allows running multiple at once on the same GPU. See:
https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
https://github.com/tensorflow/tensorflow/issues/25138
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/config.py#L524

Note: GPU options must be set at program startup.
"""
import tensorflow as tf


def set_gpu_memory(gpumem):
    """ Set max GPU memory usage (if using a GPU), use 0 for all memory, i.e.
    don't limit the usage """
    physical_devices = tf.config.experimental.list_physical_devices("GPU")

    # Skip if no GPUs or we wish to use all the GPU memory
    if len(physical_devices) == 0 or gpumem == 0:
        return

    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpumem)])

    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print("Cannot set memory growth when virtual devices configured")
        logical_devices = tf.config.experimental.list_logical_devices("GPU")

    assert len(logical_devices) == len(physical_devices) + 1
