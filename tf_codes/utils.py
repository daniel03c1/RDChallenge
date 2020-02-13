import numpy as np
import os
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
EPSILON = 1e-8
LOG_EPSILON = np.log(EPSILON)


def azimuth_to_classes(azimuth, n_classes, one_hot=True, smoothing=False):
    assert n_classes in [2, 10, 11]

    if n_classes == 2:
        classes = np.not_equal(azimuth, -1).astype(np.int32) 
    elif n_classes == 10:
        classes = azimuth // 20
    else:
        mask = np.equal(azimuth, -1).astype(np.int32)
        classes = mask * 10 + (1 - mask) * azimuth // 20
    if one_hot:
        if smoothing and n_classes == 11:
            return np.array([[.95, .05, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [.05, .9, .05, 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., .05, .9, .05, 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., .05, .9, .05, 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., .05, .9, .05, 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., .05, .9, .05, 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., .05, .9, .05, 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., .05, .9, .05, 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., .05, .9, .05, 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., .05, .95, 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])[classes]
        return np.eye(n_classes, dtype=np.float32)[classes]
    else:
        return classes


def class_to_azimuth(classes):
    mask = tf.cast(tf.equal(classes, 10), classes.dtype)
    azimuth = mask * -1 + (1 - mask) * classes * 20
    return azimuth


@tf.function
def challenge_score(y_true, y_pred):
    y_true = to_degrees(tf.argmax(y_true, axis=1))
    y_pred = to_degrees(tf.argmax(y_pred, axis=1))
    return score(y_true, y_pred)


@tf.function
def score(y_true, y_pred):
    mask = tf.math.logical_xor(tf.equal(y_true, -1),
                               tf.equal(y_pred, -1))
    mask = tf.cast(mask, tf.float32)
    diff = tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)
    diff = mask*180 + (1-mask)*diff
    diff *= np.pi / 180
    return tf.reduce_mean(tf.pow(diff, 2))


@tf.function
def to_degrees(dist):
    mask = tf.cast(tf.equal(dist, 10), dist.dtype)
    degrees = mask*-1 + (1-mask)*dist*20
    return degrees


def normalize_spec(x, norm, chan=(0, 1)):
    x = x.copy()
    if norm:
        x[:, :, :, chan] /= (np.max(x[:, :, :, chan],
                                    axis=(1, 2),
                                    keepdims=True) + EPSILON) 
    x[:, :, :, chan] = np.log(x[:, :, :, chan] + EPSILON)
    return x


def augment(mask=True, equalizer=True, roll=False, flip=False):
    def _aug(x, y):
        if mask:
            x = freq_mask(x)
            x = time_mask(x)
        if equalizer:
            x = random_equalizer(x)
        if roll:
            x = random_roll(x)
        if flip:
            x, y = random_flip(x, y)
        return x, y
    return _aug


def freq_mask(spec, max_mask_size=32, mask_num=3):
    freq, time, chan = spec.shape
    mask = tf.ones(shape=(freq, 1, 1), dtype=tf.float32)

    for i in range(mask_num):
        size = tf.random.uniform([], maxval=max_mask_size, dtype=tf.int32)
        offset = tf.random.uniform([], maxval=freq-size, dtype=tf.int32)

        mask = tf.cond(
            tf.random.uniform([]) > 0.5,
            lambda: mask*tf.concat((tf.ones(shape=(offset, 1, 1)),
                                    tf.zeros(shape=(size, 1, 1)),
                                    tf.ones(shape=(freq-size-offset, 1, 1))),
                                   0),
            lambda: mask)
    spec = spec * mask
    return tf.cast(spec, dtype=tf.float32)


def time_mask(spec, max_mask_size=32, mask_num=3):
    freq, time, chan = spec.shape
    mask = tf.ones(shape=(1, time, 1), dtype=tf.float32)

    for i in range(mask_num):
        size = tf.random.uniform([], maxval=max_mask_size, dtype=tf.int32)
        offset = tf.random.uniform([], maxval=time-size, dtype=tf.int32)

        mask = tf.cond(
            tf.random.uniform([]) > 0.5,
            lambda: mask*tf.concat((tf.ones(shape=(1, offset, 1)),
                                    tf.zeros(shape=(1, size, 1)),
                                    tf.ones(shape=(1, time-size-offset, 1))),
                                   1),
            lambda: mask)

    spec = spec * mask
    return tf.cast(spec, dtype=tf.float32)


def random_equalizer(spec, mag_only=False):
    freq, time, chan = spec.shape
    
    def _gen(maxval):
        left = tf.random.uniform([], maxval=np.pi*2, dtype=tf.float32)
        right = tf.random.uniform([], maxval=np.pi*3, dtype=tf.float32)
        weight = tf.random.uniform([], maxval=maxval, dtype=tf.float32)
        return tf.math.cos(tf.linspace(left, left+right, freq)) * weight

    # equalizer = tf.random.uniform([], minval=0.6, maxval=1.6, dtype=tf.float32)
    # equalizer = tf.math.log(equalizer + _gen(0.2) + _gen(0.2))
    equalizer = tf.math.log(1 + _gen(0.2) + _gen(0.2))
    equalizer = tf.expand_dims(tf.expand_dims(equalizer, 1), 1)
    equalizer = tf.concat((tf.tile(equalizer, [1, 1, chan//2]),
                           tf.zeros(shape=(freq, 1, chan//2))),
                           2)
    return spec + equalizer


def random_roll(spec):
    freq, time, chan = spec.shape
    shift = tf.random.uniform([], maxval=time, dtype=tf.int32)
    return tf.roll(spec, shift=shift, axis=1)


def random_flip(spec, label, mag_only=False):
    assert label.shape[0] == 11
    label = tf.concat([tf.reverse(label[:10], axis=(0,)),
                       label[10:]],
                       axis=0)

    spec = tf.gather(spec, (1, 0, 3, 2), axis=2)
    return spec, label


def make_dataset(x, y, n_proc, batch_per_node, train=False, **kwargs):
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).cache()
    if train:
        dataset = dataset.map(augment(**kwargs), num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat().shuffle(buffer_size=len(x))
    return dataset.batch(batch_per_node, drop_remainder=True).prefetch(AUTOTUNE)

