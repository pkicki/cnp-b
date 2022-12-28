import tensorflow as tf


def huber(x, delta=1.0):
    abs_x = tf.abs(x)
    half = tf.convert_to_tensor(0.5, dtype=abs_x.dtype)
    return tf.where(abs_x <= delta, half * tf.square(x), delta * abs_x - half * tf.square(delta))
