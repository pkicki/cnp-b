import tensorflow as tf


def normalize_xy(xy):
    o = tf.ones_like(xy)[..., 0]
    xy = (xy - tf.stack([1.0 * o, 0.0 * o], axis=-1)) / tf.stack([0.4 * o, 0.4 * o], axis=-1)
    return xy
