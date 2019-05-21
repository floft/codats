"""
Set max GPU memory usage (if GPUs are found)

This allows running multiple at once on the same GPU. See:
https://www.tensorflow.org/alpha/guide/using_gpu#limiting_gpu_memory_growth
https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
https://github.com/tensorflow/tensorflow/issues/25138
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/config.py#L524

Note: GPU options must be set at program startup.
"""
import tensorflow as tf


def set_gpu_memory(gpumem):
    """
    Set max GPU memory usage (if using a GPU), use 0 for all memory, i.e. don't
    limit the usage.

    Note: now gpumem is in MiB not a percentage of the total.
    """
    # Skip if we wish to use all the GPU memory
    if gpumem == 0:
        return

    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpumem)])
        except RuntimeError as e:
            # Virtual devices must be set at program startup
            print(e)
