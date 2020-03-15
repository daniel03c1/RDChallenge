import argparse
import numpy as np
import os
import tensorflow as tf
import pickle
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

import models
from utils import *


AUTOTUNE = tf.data.experimental.AUTOTUNE
EPSILON = 1e-8
LOG_EPSILON = np.log(EPSILON)


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='model')
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--model', type=str, default='suggestion')


def mask(spec, max_ratio=.5, axis=0, fill_value=0):
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

    spec = spec * mask + fill_value * (1-mask)
    return tf.cast(spec, dtype=tf.float32)


def window_dataset_from_list(padded_x_list, padded_y_list, 
                             pad_size, step_size, 
                             batch_per_node, train=False):
    def augment(x, y):
        x = mask(x, 0.2, 1, fill_value=(LOG_EPSILON-4.5252)/2.6146)
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((padded_x_list, padded_y_list))
    if train:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat().shuffle(buffer_size=100000) # len(padded_x_list))
    return dataset.batch(batch_per_node, drop_remainder=True).prefetch(AUTOTUNE)


if __name__ == "__main__":
    config = args.parse_args()
    print(config, '\n')

    device_nums = (2, 3) # 0, 1)
    devices = ['/device:GPU:{}'.format(i) for i in device_nums]
    strategy = tf.distribute.MirroredStrategy(devices)

    """ HYPER_PARAMETERS """
    BATCH_SIZE = 4096 // len(devices) # per each GPU
    TOTAL_EPOCH = 100
    VAL_SPLIT = 0.15

    WINDOW_SIZE = 2*(config.pad_size-1)/config.step_size + 3

    """ DATA """
    # 1. IMPORTING TRAINING DATA & PRE-PROCESSING
    PATH = '/datasets/ai_challenge/' # icassp/' # inside a container
    if not os.path.isdir(PATH): # outside... 
        PATH = '/media/data1/datasets/ai_challenge/' # icassp/'
    PATH = os.path.join(PATH, 'TIMIT_NOISEX_extended/TRAIN')

    # x = pickle.load(open(os.path.join(PATH, 'x_8_mel_flat.pickle'), 'rb'))
    # y = pickle.load(open(os.path.join(PATH, 'y_8_vad_fixed.pickle'), 'rb'))
    x = pickle.load(open(os.path.join(PATH, 'snr-5.pickle'), 'rb'))
    y = pickle.load(open(os.path.join(PATH, 'label.pickle'), 'rb'))

    # preprocess 
    def preprocess_spec(spec):
        spec = np.log(spec + EPSILON)
        spec = pad(spec, config.pad_size, 1, LOG_EPSILON)
        spec = (spec - 4.5252) / 2.6146 # normalize
        spec = spec.transpose(1, 0) # , 2) # to (time, freq, chan)
        windows = sequence_to_windows(spec, 
                                      config.pad_size, config.step_size, False)
        return windows

    def preprocess_label(label):
        label = pad(label, config.pad_size, 0, 0)
        label = sequence_to_windows(label,
                                    config.pad_size, config.step_size, False)
        return label

    x = np.concatenate(list(map(preprocess_spec, x)), axis=0)
    if config.model in ['bdnn']: # win2win
        y = np.concatenate(list(map(preprocess_label, y)), axis=0)
    elif config.model in ['dnn']: # win2one
        y = np.concatenate(y, axis=0)
    else:
        raise ValueError('invalid model :{}'.format(config.model))

    # 2. SPLITTING TRAIN AND VALIDATION SET
    val_size = int(len(x) * VAL_SPLIT)

    x, val_x = x[:-val_size], x[-val_size:]
    y, val_y = y[:-val_size], y[-val_size:]

    # 3. TRAINING
    with strategy.scope():
        """ MODEL """
        time, freq = x[0].shape

        model = getattr(models, config.model)(
            input_shape=(WINDOW_SIZE, freq),
            kernel_regularizer=tf.keras.regularizers.l2(1e-5))

        model.compile(optimizer=Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'AUC'])
        model.summary()

        """ TRAINING """
        train_dataset = window_dataset_from_list(
            x, y, config.pad_size, config.step_size, BATCH_SIZE, train=True)
        val_dataset = window_dataset_from_list(
            val_x, val_y, 
            config.pad_size, config.step_size, BATCH_SIZE, train=False)

        callbacks = [
            EarlyStopping(monitor='val_AUC',
                          mode='max',
                          patience=5),
            CSVLogger(config.name + '.log',
                      append=True),
            ModelCheckpoint(config.name+'.h5',
                            monitor='val_AUC',
                            mode='max',
                            save_best_only=True),
            TerminateOnNaN(),
        ]

        model.fit(train_dataset,
                  epochs=TOTAL_EPOCH,
                  validation_data=val_dataset,
                  steps_per_epoch=len(x)//BATCH_SIZE,
                  callbacks=callbacks)

