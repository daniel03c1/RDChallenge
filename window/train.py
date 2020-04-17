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


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='model')
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--model', type=str, default='test2')
args.add_argument('--lr', type=float, default=0.1)
args.add_argument('--gpus', type=str, default='0,1')
args.add_argument('--skip', type=int, default=3)


def mask(spec, max_ratio, axis=0, fill_value=0):
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

    fill_value = tf.reduce_min(spec) # TEST
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
        dataset = dataset.repeat().shuffle(buffer_size=1000000) # len(padded_x_list))
    return dataset.batch(batch_per_node, drop_remainder=True).prefetch(AUTOTUNE)


if __name__ == "__main__":
    config = args.parse_args()
    print(config, '\n')

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    strategy = tf.distribute.MirroredStrategy()

    """ HYPER_PARAMETERS """
    BATCH_SIZE = 4096 # per each GPU
    TOTAL_EPOCH = 100

    WINDOW_SIZE = int(2*(config.pad_size-1)/config.step_size + 3)

    """ DATA """
    # 1. IMPORTING TRAINING DATA
    PATH = '/datasets/ai_challenge/' # inside a docker container
    if not os.path.isdir(PATH): # outside... 
        PATH = '/media/data1/datasets/ai_challenge/'
    PATH = os.path.join(PATH, 'TIMIT_NOISEX_extended/')

    TRAINPATH = os.path.join(PATH, 'TRAIN')
    TESTPATH = os.path.join(PATH, 'TEST')

    x, val_x = [], []
    y, val_y = [], []

    for snr in ['-20', '-15', '-5', '0', '5', '10']:
        x += pickle.load(open(os.path.join(TRAINPATH, f'snr{snr}.pickle'), 'rb'))
        val_x += pickle.load(open(os.path.join(TESTPATH, f'snr{snr}_test.pickle'), 'rb'))

    for i in range(6):
        y += pickle.load(open(os.path.join(TRAINPATH, 'phn.pickle'), 'rb'))
        val_y += pickle.load(open(os.path.join(TESTPATH, 'phn_test.pickle'), 'rb'))

    """ MODEL """
    with strategy.scope(): 
        time, freq = WINDOW_SIZE, x[0].shape[0]

        model = getattr(models, config.model)(
            input_shape=(WINDOW_SIZE, freq),
            kernel_regularizer=tf.keras.regularizers.l2(1e-5))

        model.compile(optimizer=SGD(config.lr, momentum=0.9),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'AUC'])
        model.summary()

    """ DATA """
    # 2. DATA PRE-PROCESSING
    x = np.concatenate(
        list(map(preprocess_spec(config, skip=config.skip), x)), axis=0)
    val_x = np.concatenate(
        list(map(preprocess_spec(config, skip=config.skip), val_x)), axis=0)
    if model.output_shape[-1] != 1: # win2win
        y = np.concatenate(
            list(map(label_to_window(config, skip=config.skip), y)), axis=0)
        val_y = np.concatenate(
            list(map(label_to_window(config, skip=config.skip), val_y)), axis=0)
    else: # win2one
        y = np.concatenate(y, axis=0)
        val_y = np.concatenate(val_y, axis=0)
    print("data pre-processing finished")

    # 2.1. SHUFFLING TRAINING DATA
    perm = np.random.permutation(len(x))
    x = np.take(x, perm, axis=0)
    y = np.take(y, perm, axis=0)
    print("shuffling training data finished")

    """ TRAINING """
    with strategy.scope(): 
        train_dataset = window_dataset_from_list(
            x, y, config.pad_size, config.step_size, BATCH_SIZE, train=True)
        val_dataset = window_dataset_from_list(
            val_x, val_y, 
            config.pad_size, config.step_size, BATCH_SIZE, train=False)

        callbacks = [
            ReduceLROnPlateau(monitor='val_AUC',
                              factor=0.9,
                              patience=1,
                              mode='max',
                              verbose=1,
                              min_lr=1e-5),
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
