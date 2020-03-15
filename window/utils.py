import tensorflow as tf
import numpy as np


def sequence_to_windows(sequence, 
                        pad_size, 
                        step_size, 
                        padding=True, 
                        const_value=0):
    '''
    SEQUENCE: (time, ...)

    RETURN: (time, window, ...)
    '''
    assert (pad_size-1) % step_size == 0

    window = np.concatenate([np.arange(-pad_size, -step_size, step_size),
                             np.array([-1, 0, 1]),
                             np.arange(step_size+1, pad_size+1, step_size)],
                            axis=0)
    window += pad_size
    output_len = len(sequence) if padding else len(sequence) - 2*pad_size
    window = window[np.newaxis, :] + np.arange(output_len)[:, np.newaxis]

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

