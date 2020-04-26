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
args.add_argument('--name', type=str, default='bdnn')
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--model', type=str, default='bdnn')
args.add_argument('--lr', type=float, default=0.1)
args.add_argument('--gpus', type=str, default='2,3')
args.add_argument('--skip', type=int, default=2)
args.add_argument('--noise_aug', action='store_true')
args.add_argument('--voice_aug', action='store_true')
args.add_argument('--aug', action='store_true')


def window_dataset_from_list(padded_x_list, padded_y_list, 
                             pad_size, step_size, 
                             batch_per_node, train=False, aug=True):
    def augment(x, y):
        x = mask(x, 0.2, axis=1, method='reduce_min')
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((padded_x_list, padded_y_list))
    if train:
        dataset = dataset.repeat().shuffle(buffer_size=1000000) 
        if aug:
            dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
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
    TESTPATH = os.path.join(PATH, 'TIMIT_NOISEX_extended/TEST')
    TRAINPATH = os.path.join(PATH, 'TIMIT_noisex3/')

    x, val_x = [], []
    y, val_y = [], []

    snrs = ['-20', '-15', '-5', '0', '5', '10']

    tail = '.pickle' if config.noise_aug else '_no_noise_aug.pickle'
    speed = ['09', '10', '11'] if config.voice_aug else ['10']

    for snr in snrs:
        for s in speed:
            x += pickle.load(
                open(os.path.join(TRAINPATH, f'snr{snr}_{s}{tail}'), 'rb'))
        
        val_x += pickle.load(
            open(os.path.join(TESTPATH, f'snr{snr}.pickle'), 'rb'))

    for i in range(len(snrs)):
        for s in speed:
            y += pickle.load(
                open(os.path.join(TRAINPATH, f'label_{s}.pickle'), 'rb'))

        val_y += pickle.load(open(os.path.join(TESTPATH, 'phn.pickle'), 'rb'))

    # fix mismatch 
    for i in range(len(x)):
        x[i] = x[i][:, :len(y[i])]

    """ MODEL """
    with strategy.scope(): 
        time, freq = WINDOW_SIZE, x[0].shape[0]

        print(config.model)
        model = getattr(models, config.model)(
            input_shape=(WINDOW_SIZE, freq),
            kernel_regularizer=tf.keras.regularizers.l2(1e-5))

        model.compile(optimizer=SGD(config.lr, momentum=0.9),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'AUC'])

    """ DATA """
    # 2. DATA PRE-PROCESSING
    x = np.concatenate(
        list(map(preprocess_spec(config, skip=config.skip), x)), axis=0)
    val_x = np.concatenate(
        list(map(preprocess_spec(config), val_x)), axis=0)
    if model.output_shape[-1] != 1: # win2win
        y = np.concatenate(
            list(map(label_to_window(config, skip=config.skip), y)), axis=0)
        val_y = np.concatenate(
            list(map(label_to_window(config), val_y)), axis=0)
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
            x, y, config.pad_size, config.step_size, BATCH_SIZE, train=True, aug=config.aug)
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
                  steps_per_epoch=len(x)//BATCH_SIZE//len(speed),
                  callbacks=callbacks)

