import tensorflow as tf
import numpy as np


EPSILON = 1e-8
LOG_EPSILON = np.log(EPSILON)


def sequence_to_windows(sequence, 
                        pad_size, 
                        step_size, 
                        skip=1,
                        padding=True, 
                        const_value=0):
    '''
    SEQUENCE: (time, ...)
    PAD_SIZE:  int -> width of the window // 2
    STEP_SIZE: int -> step size inside the window
    SKIP:      int -> skip windows...
        ex) if skip == 2, total number of windows will be halved.
    PADDING:   bool -> whether the sequence is padded or not
    CONST_VALUE: (int, float) -> value to fill in the padding

    RETURN: (time, window, ...)
    '''
    assert (pad_size-1) % step_size == 0

    window = np.concatenate([np.arange(-pad_size, -step_size, step_size),
                             np.array([-1, 0, 1]),
                             np.arange(step_size+1, pad_size+1, step_size)],
                            axis=0)
    window += pad_size
    output_len = len(sequence) if padding else len(sequence) - 2*pad_size
    window = window[np.newaxis, :] + np.arange(0, output_len, skip)[:, np.newaxis]

    if padding:
        pad = np.ones((pad_size, *sequence.shape[1:]), dtype=np.float32)
        pad *= const_value
        sequence = np.concatenate([pad, sequence, pad], axis=0)

    return np.take(sequence, window, axis=0)


def windows_to_sequence(windows,
                        pad_size,
                        step_size):
    windows = np.array(windows)
    sequence = np.zeros((windows.shape[0],), dtype=np.float32)
    indices = np.arange(1, windows.shape[0]+1)
    indices = sequence_to_windows(indices, pad_size, step_size, True)

    for i in range(windows.shape[0]):
        pred = windows[np.where(indices-1 == i)]
        sequence[i] = pred.mean()
    
    return sequence


def pad(spec, pad_size, axis, const_value):
    padding = np.ones((*spec.shape[:axis], pad_size, *spec.shape[axis+1:]),
                      dtype=np.float32)
    padding *= const_value
    return np.concatenate([padding, spec, padding], axis=axis)


# TODO (test is required)
def preprocess_spec(config, feature='mel', skip=1):
    if feature not in ['spec', 'mel', 'mfcc']:
        raise ValueError(f'invalid feature - {feature}')

    def _preprocess_spec(spec):
        const_value = EPSILON
        if feature in ['spec', 'mel']:
            spec = np.log(spec + EPSILON)
            const_value = LOG_EPSILON

        spec = spec.transpose(1, 0) # to (time, freq)

        windows = sequence_to_windows(spec, 
                                      config.pad_size, config.step_size,
                                      skip, True, const_value)
        if feature == 'mel':
            windows = (windows + 2.7766097) / 5.0815244 # normalize
        return windows
    return _preprocess_spec


# TODO (test is required)
def label_to_window(config, skip=1):
    def _preprocess_label(label):
        label = sequence_to_windows(
            label, config.pad_size, config.step_size, skip, True)
        return label
    return _preprocess_label


# TODO (test is required)
def mask(x, y, max_ratio, axis=0, cval=0.):
    total = x.shape[axis]
    max_mask_size = int(total * max_ratio)
    mask_shape = tuple(1 if i != axis else -1 
                       for i in range(len(x.shape)))

    size = tf.random.uniform([], maxval=max_mask_size, dtype=tf.int32)
    offset = tf.random.uniform([], maxval=total-size, dtype=tf.int32)

    mask = tf.concat((tf.ones(shape=(offset,)),
                      tf.zeros(shape=(size,)),
                      tf.ones(shape=(total-size-offset,))),
                     0)
    '''
    size = tf.random.uniform([], maxval=max_mask_size, dtype=tf.float32)
    mask = tf.random.uniform([total,], dtype=tf.float32)
    mask = mask / tf.reduce_sum(mask) * (total-size)
    '''
    mask = tf.reshape(mask, mask_shape)

    x = x * mask + cval * (1-mask)
    y = y * tf.reduce_mean(mask, axis=2) # test (make no difference)

    return x, y


def mask2(x, y, max_ratio, axis=0, cval=0.):
    total = x.shape[axis]
    max_mask_size = int(total * max_ratio)
    mask_shape = tuple(1 if i != axis else -1 
                       for i in range(len(x.shape)))

    if max_ratio == 0:
        mask = tf.ones_like(x)
    else:
        size = tf.random.uniform([], maxval=max_mask_size, dtype=tf.int32)
        offset = tf.random.uniform([], maxval=total-size, dtype=tf.int32)

        mask = tf.concat((tf.ones(shape=(offset,)),
                          tf.zeros(shape=(size,)),
                          tf.ones(shape=(total-size-offset,))),
                         0)
        mask = tf.reshape(mask, mask_shape)

    masked = x * mask + cval * (1-mask)
    mask = tf.ones_like(x) * mask

    return (masked,), (x, mask)


# TODO (test is required)
def cutmix(x, y, axis=2, batch_size=4096):
    # x.shape = [batch, time, freq]
    '''
    total = x.shape[axis]
    mask_shape = tuple(1 if i != axis else -1 
                       for i in range(len(x.shape)))

    size = tf.random.uniform([], maxval=total, dtype=tf.int32)
    offset = tf.random.uniform([], maxval=total-size, dtype=tf.int32)

    mask = tf.concat((tf.ones(shape=(offset,)),
                      tf.zeros(shape=(size,)),
                      tf.ones(shape=(total-size-offset,))),
                     0)
    mask = tf.reshape(mask, mask_shape)

    lmbda = tf.cast(size / total, dtype=tf.float32)

    indices = tf.random.shuffle(tf.range(batch_size))
    x = x * mask + tf.gather(x, indices, axis=0) * (1-mask)
    y = y * (1-lmbda) + tf.gather(y, indices, axis=0) * lmbda
    '''
    batch, time, freq = x.shape

    # select lambda
    sqrt_lmbda = tf.math.sqrt(tf.random.uniform([], maxval=1.))

    # set size of window and recalculate lambda
    size_t = tf.cast(time * sqrt_lmbda, dtype=tf.int32)
    size_f = tf.cast(freq * sqrt_lmbda, dtype=tf.int32)

    # select window
    offset_t = tf.random.uniform([], maxval=time-size_t, dtype=tf.int32)
    offset_f = tf.random.uniform([], maxval=freq-size_f, dtype=tf.int32)

    windows = tf.ones([size_t, size_f])
    windows = tf.pad(windows, 
                     [[offset_t, time-offset_t-size_t],
                      [offset_f, freq-offset_f-size_f]])
    windows = tf.expand_dims(windows, axis=0)

    # shuffle~
    indices = tf.random.shuffle(tf.range(batch_size))
    x = x * windows + tf.gather(x, indices, axis=0) * (1-windows)

    mean = tf.reduce_mean(windows, axis=2) # mean on freq axis
    y = y * mean + tf.gather(y, indices, axis=0) * (1-mean)

    return x, y


# TODO (test is required)
def add_noise(x, y, stddev=0.1):
    noise = tf.random.normal(x.shape[1:], stddev=stddev)
    x = x + tf.expand_dims(noise, axis=0)
    return x, y


def layer_and_model(layers, model):
    if isinstance(layers, list):
        inputs = layers[0]
        outputs = inputs
        for layer in layers[1:]:
            outputs = layer(outputs)
    else:
        inputs = layers
        outputs = layers
    outputs = model(outputs)
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def extract_model(model, index=-1):
    return tf.keras.models.Model(inputs=model.layers[index].input,
                                 outputs=model.layers[index].output)


if __name__ == '__main__':
    from models import bdnn, Mask
    shape = [16, 16]
    m = bdnn(shape)
    l = tf.keras.layers.Input(shape)
    l_ = Mask()
    m_ = layer_and_model([l, l_], m)
    m_.summary()
    extract_model(m_).summary()

