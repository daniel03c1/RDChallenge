import numpy as np
import tensorflow as tf


def challenge_score(true, pred):
    return score(to_degrees(true), to_degrees(pred))


def score(true, pred):
    mask = tf.math.logical_xor(tf.equal(true, -1),
                               tf.equal(pred, -1))
    mask = tf.cast(mask, tf.float32)

    diff = tf.cast(true, tf.float32) - tf.cast(pred, tf.float32)
    diff = mask*180 + (1-mask)*diff
    diff *= np.pi / 180
    return tf.reduce_mean(tf.pow(diff, 2))


def to_degrees(dist):
    dist = tf.argmax(dist, axis=-1)
    mask = tf.cast(tf.equal(dist, 10), dist.dtype)
    degrees = mask*-1 + (1-mask)*dist*20
    return degrees

