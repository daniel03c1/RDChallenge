import os
import numpy as np
import tensorflow as tf

EPSILON = 1e-8


'''
LABEL PRE-PROCESSING
'''
def degree_to_class(degrees,
                    resolution=20,
                    min_degree=0,
                    max_degree=180,
                    one_hot=True):
    degrees = np.array(degrees)
    n_classes = int((max_degree-min_degree)/resolution + 2)

    mask = np.logical_and(min_degree <= degrees, degrees <= max_degree)
    classes = mask * (degrees/resolution) + (1-mask) * (n_classes-1)
    classes = classes.astype(np.int32)

    if not one_hot:
        return classes 
    return np.eye(n_classes, dtype=np.float32)[classes]


def class_to_degree(classes,
                    resolution=20,
                    min_degree=0,
                    max_degree=180,
                    non_voice_value=-1):
    classes = np.array(classes)
    mask = classes != (max_degree-min_degree)/resolution + 1
    degrees = mask * (classes*resolution) + (1-mask) * non_voice_value
    return degrees


''' 
UTILS FOR FRAMES AND WINDOWS 
'''
def seq_to_windows(seq, 
                   window, 
                   skip=1,
                   padding=True, 
                   **kwargs):
    '''
    INPUT:
        seq: np.ndarray
        window: array of indices
            ex) [-3, -1, 0, 1, 3]
        skip: int
        padding: bool
        **kwargs: params for np.pad

    OUTPUT:
        windows: [n_windows, window_size, ...]
    '''
    window = np.array(window - np.min(window)).astype(np.int32)
    win_size = max(window) + 1
    windows = window[np.newaxis, :] \
            + np.arange(0, len(seq), skip)[:, np.newaxis]
    if padding:
        seq = np.pad(
            seq,
            [[win_size//2, (win_size-1)//2]] + [[0, 0]]*len(seq.shape[1:]),
            **kwargs)

    return np.take(seq, windows, axis=0)


def windows_to_seq(windows,
                   window,
                   skip=1):
    '''
    INPUT:
        windows: np.ndarray (n_windows, window_size, ...)
        window: array of indices
        skip: int

    OUTPUT:
        seq
    '''
    n_window = windows.shape[0]
    window = np.array(window - np.min(window)).astype(np.int32)
    win_size = max(window)

    seq_len = (n_window-1)*skip + 1
    seq = np.zeros([seq_len, *windows.shape[2:]], dtype=windows.dtype)
    count = np.zeros(seq_len)

    for i, w in enumerate(window):
        indices = np.arange(n_window)*skip - win_size//2 + w
        select = np.logical_and(0 <= indices, indices < seq_len)
        seq[indices[select]] += windows[select, i]
        count[indices[select]] += 1
    
    seq = seq / (count + EPSILON)
    return seq


'''
DATASET
'''
def window_generator(specs, 
                     labels, 
                     window_size, 
                     infinite=True):
    '''
    GENERATES WINDOWED SPECS AND LABELS

    INPUT:
        specs: continuous single spectrogram
        labels: continuous framewise labels
        window_size: 
        infinite: infinite generator or not
    OUTPUT:
        generator
    '''
    def generator():
        max_hop = window_size
        n_frames = len(specs)
        i = 0

        while True:
            i = (i + np.random.randint(1, max_hop+1)) % n_frames

            if i+window_size > n_frames:
                if not infinite:
                    break

                spec = np.concatenate(
                        [specs[i:], specs[:(i+window_size)%n_frames]],
                    axis=0)
                assert spec.shape[0] == window_size
            else:
                spec = specs[i:i+window_size]

            label = labels[(i+window_size//2) % n_frames] # center

            yield (spec, label)

    return generator


