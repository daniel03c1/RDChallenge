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
            windows = (windows + 1.3856256) / 5.0747867 # normalize
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
def mask(spec, max_ratio, axis=0, method='reduce_mean'):
    total = spec.shape[axis]
    max_mask_size = int(total * max_ratio)
    mask_shape = tuple(1 if i != axis else -1 
                       for i in range(len(spec.shape)))

    size = tf.random.uniform([], maxval=max_mask_size, dtype=tf.int32)
    offset = tf.random.uniform([], maxval=total-size, dtype=tf.int32)

    mask = tf.concat((tf.ones(shape=(offset,)),
                      tf.zeros(shape=(size,)),
                      tf.ones(shape=(total-size-offset,))),
                     0)
    mask = tf.reshape(mask, mask_shape)

    fill_value = getattr(tf, method)(spec)
    return spec * mask + fill_value * (1-mask)

