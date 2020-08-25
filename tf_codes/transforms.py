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
    '''
    begin = [0] * len(specs.shape)
    end = [s for s in specs.shape]
    print(begin, end)

    left = tf.random.uniform([], maxval=width*2, dtype=tf.int32)
    begin[axis] = left
    end[axis] += left - 1
    print(begin, end, specs.shape)

    new_specs = tf.slice(new_specs, begin, end)
    '''
    new_specs = tf.image.random_crop(new_specs, specs.shape)
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


def minmax_norm_magphase(specs, labels=None):
    n_chan = specs.shape[-1] // 2
    mag = specs[..., :n_chan]
    phase = specs[..., n_chan:]
    axis = tuple(range(1, len(specs.shape)))

    mag_max = tf.math.reduce_max(mag, axis=axis, keepdims=True)
    mag_min = tf.math.reduce_min(mag, axis=axis, keepdims=True)
    phase_max = tf.math.reduce_max(phase, axis=axis, keepdims=True)
    phase_min = tf.math.reduce_min(phase, axis=axis, keepdims=True)

    specs = tf.concat(
        [(mag-mag_min)/(mag_max-mag_min+EPSILON),
         (phase-phase_min)/(phase_max-phase_min+EPSILON)],
        axis=-1)

    if labels is not None:
        return specs, labels
    return specs


def cutmix(xs, ys):
    '''
    xs : [batch, height, width, chan] np.ndarray
    ys : [batch, n_classes] np.ndarray
    '''
    xs = tf.cast(xs, tf.float32)
    ys = tf.cast(ys, tf.float32)
    batch, height, width, chan = xs.shape

    # select lambda
    sqrt_lmbda = tf.math.sqrt(tf.random.uniform([], maxval=1.))

    # set size of window and recalculate lambda
    size_h = tf.cast(height * sqrt_lmbda, dtype=tf.int32)
    size_w = tf.cast(width * sqrt_lmbda, dtype=tf.int32)

    # select window
    offset_h = tf.random.uniform([], maxval=height-size_h, dtype=tf.int32)
    offset_w = tf.random.uniform([], maxval=width-size_w, dtype=tf.int32)

    windows = tf.ones([size_h, size_w], dtype=tf.float32)
    windows = tf.pad(windows,
                     [[offset_h, height-offset_h-size_h],
                      [offset_w, width-offset_w-size_w]])
    windows = tf.reshape(windows, (1, height, width, 1))

    # shuffle
    indices = tf.math.reduce_mean(tf.ones_like(ys), axis=-1)
    indices = tf.cast(tf.math.cumsum(indices, exclusive=True), dtype='int32')
    indices = tf.random.shuffle(indices)

    # mix
    xs = xs * (1-windows) + tf.gather(xs, indices, axis=0) * windows
    mean = tf.cast((size_h * size_w) / (height * width), dtype=tf.float32)
    ys = ys * (1-mean) + tf.gather(ys, indices, axis=0) * mean

    return xs, ys


def interclass_cutmix(xs, ys):
    '''
    xs : [batch, height, width, chan] np.ndarray
    ys : [batch, n_classes] np.ndarray
    '''
    xs_, ys_ = [], []
    cls = tf.argmax(ys, axis=-1)

    for i in range(ys.shape[-1]):
        select = tf.squeeze(tf.where(cls == i), axis=-1)
        xs_temp = tf.gather(xs, select, axis=0)
        ys_temp = tf.gather(ys, select, axis=0)
        xs_temp, ys_temp = cutmix(xs_temp, ys_temp)
        xs_.append(xs_temp)
        ys_.append(ys_temp)

    return tf.concat(xs_, axis=0), tf.concat(ys_, axis=0)
