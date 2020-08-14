import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
EPSILON = 1e-8
LOG_EPSILON = tf.math.log(EPSILON)


def mask(specs, axis, max_mask_size=None, n_mask=1):
    def make_shape(size):
        # returns (1, ..., size, ..., 1)
        shape = [1] * len(specs.shape)
        shape[axis] = size
        return tuple(shape)

    total = specs.shape[axis]
    mask = tf.ones(make_shape(total), dtype=specs.dtype)
    if max_mask_size is None:
        max_mask_size = total

    for i in range(n_mask):
        size = tf.random.uniform([], maxval=max_mask_size, dtype=tf.int32)
        offset = tf.random.uniform([], maxval=total-size, dtype=tf.int32)

        mask *= tf.concat(
            (tf.ones(shape=make_shape(offset), dtype=mask.dtype),
             tf.zeros(shape=make_shape(size), dtype=mask.dtype),
             tf.ones(shape=make_shape(total-size-offset), dtype=mask.dtype)),
            axis=axis)

    return specs * mask


def random_shift(specs, axis=0, width=16):
    new_specs = tf.pad(specs, [[0]*2 if i != axis else [width]*2
                               for i in range(len(specs.shape))])
    begin = [0] * len(specs.shape)
    end = [s for s in new_specs.shape]

    left = tf.random.uniform([], maxval=width*2, dtype=tf.int32)
    begin[axis] = left
    end[axis] -= width*2 - left + 1

    new_specs = tf.slice(new_specs, begin, end)
    return new_specs


def random_magphase_flip(spec, label):
    flip = tf.cast(tf.random.uniform([]) > 0.5, spec.dtype)
    n_chan = spec.shape[-1] // 2
    chans = tf.reshape(tf.range(n_chan*2), (2, n_chan))
    chans = tf.reshape(tf.reverse(chans, axis=[-1]), (-1,))
    spec = flip*spec + (1-flip)*tf.gather(spec, chans, axis=-1)

    flip = tf.cast(flip, label.dtype)
    label = flip*label \
            + (1-flip)*tf.concat(
                [tf.reverse(label[..., :-1], axis=(-1,)), label[..., -1:]],
                axis=-1)

    return spec, label


def magphase_mixup(alpha=2.):
    beta = tf.compat.v1.distributions.Beta(alpha, alpha)

    def _mixup(specs, labels):
        # preprocessing
        specs = tf.cast(specs, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.float32)

        indices = tf.reduce_mean(
            tf.ones_like(labels, dtype=tf.int32),
            axis=range(1, len(labels.shape)))
        indices = tf.cumsum(indices, exclusive=True)
        indices = tf.random.shuffle(indices)

        # assume mag, phase...
        n_chan = specs.shape[-1] // 2
        mag, phase = specs[..., :n_chan], specs[..., n_chan:]

        real = mag * tf.cos(phase)
        img = mag * tf.sin(phase)

        l = beta.sample()

        real = l*real + (1-l)*tf.gather(real, indices, axis=0)
        img = l*img + (1-l)*tf.gather(img, indices, axis=0)
        
        mag = tf.math.sqrt(real**2 + img**2)
        phase = tf.math.atan2(img, real)

        specs = tf.concat([mag, phase], axis=-1)
        labels = tf.cast(labels, tf.float32)
        labels = l*labels + (1-l)*tf.gather(labels, indices, axis=0)
        
        return specs, labels

    return _mixup


def log_magphase(specs, labels=None):
    n_chan = specs.shape[-1] // 2
    specs = tf.concat(
            [tf.math.log(specs[..., :n_chan]+EPSILON), specs[..., n_chan:]],
            axis=-1)
    if labels is not None:
        return specs, labels
    return specs

