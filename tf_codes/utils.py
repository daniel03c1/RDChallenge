import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE


def azimuth_to_classes(azimuth, n_classes, one_hot=True):
    assert n_classes in [2, 10, 11]

    if n_classes == 2:
        classes = np.not_equal(azimuth, -1).astype(np.int32) 
    elif n_classes == 10:
        classes = azimuth // 20
    else:
        mask = np.equal(azimuth, -1).astype(np.int32)
        classes = mask * 10 + (1 - mask) * azimuth // 20
    if one_hot:
        return np.eye(n_classes, dtype=np.float32)[classes]
    else:
        return classes


def class_to_azimuth(classes):
    mask = tf.cast(tf.equal(classes, 10), classes.dtype)
    azimuth = mask * -1 + (1 - mask) * classes * 20
    return azimuth


def normalize_spec(x, log=True, minmax=False, chan=(0, 1)):
    x = x.copy()
    if log:
        x[:, :, :, chan] = np.log(x[:, :, :, chan] + 1e-8)
    if minmax:
        x -= np.min(x, axis=(1, 2), keepdims=True)
        x /= np.max(x, axis=(1, 2), keepdims=True) + 1e-8
    return x


def augment(mask=True, equalizer=True):
    def _aug(x, y):
        if mask:
            x = freq_mask(x)
            x = time_mask(x)
        if equalizer:
            x = random_equalizer(x)
        return x, y
    return _aug


def freq_mask(spec, max_mask_size=6, mask_num=2):
    freq, time, _ = spec.shape
    mask = tf.ones(shape=(freq, 1, 1), dtype=tf.float32)

    for i in range(mask_num):
        size = tf.random.uniform([], minval=0, maxval=max_mask_size, dtype=tf.int32)
        offset = tf.random.uniform([], minval=0, maxval=freq-size, dtype=tf.int32)

        mask *= tf.concat((tf.ones(shape=(offset, 1, 1)),
                           tf.zeros(shape=(size, 1, 1)),
                           tf.ones(shape=(freq-size-offset, 1, 1))),
                           0)
    spec = spec * mask
    return tf.cast(spec, dtype=tf.float32)


def time_mask(spec, max_mask_size=15, mask_num=2):
    freq, time, _ = spec.shape
    mask = tf.ones(shape=(1, time, 1), dtype=tf.float32)

    for i in range(mask_num):
        size = tf.random.uniform([], minval=0, maxval=max_mask_size, dtype=tf.int32)
        offset = tf.random.uniform([], minval=0, maxval=time-size, dtype=tf.int32)

        mask *= tf.concat((tf.ones(shape=(1, offset, 1)),
                           tf.zeros(shape=(1, size, 1)),
                           tf.ones(shape=(1, time-size-offset, 1))),
                           1)
    spec = spec * mask
    return tf.cast(spec, dtype=tf.float32)


def random_equalizer(spec):
    freq, time, chan = spec.shape
    
    def _gen(maxval):
        left = tf.random.uniform([], maxval=np.pi*2, dtype=tf.float32)
        right = tf.random.uniform([], maxval=np.pi*3, dtype=tf.float32)
        weight = tf.random.uniform([], maxval=maxval, dtype=tf.float32)
        return tf.math.cos(tf.linspace(left, left+right, freq)) * weight

    equalizer = _gen(0.2) + _gen(0.2) + 1
    equalizer = tf.expand_dims(tf.expand_dims(equalizer, 1), 1)
    equalizer = tf.concat((tf.tile(equalizer, [1, 1, chan//2]),
                           tf.ones(shape=(freq, 1, chan//2))),
                           2)
    return spec * equalizer


def make_dataset(x, y, n_proc, batch_per_node, train=False, **kwargs):
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).cache()
    if train:
        dataset = dataset.map(augment(**kwargs), num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat().shuffle(buffer_size=6000)
    return dataset.batch(batch_per_node, drop_remainder=True).prefetch(AUTOTUNE)

