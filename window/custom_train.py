import argparse
import numpy as np
import os
import tensorflow as tf
import pickle
from functools import partial
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

import models
from utils import *


AUTOTUNE = tf.data.experimental.AUTOTUNE


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='bdnn')
args.add_argument('--pretrain', type=str, default='')
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--model', type=str, default='bdnn')
args.add_argument('--lr', type=float, default=0.1)
args.add_argument('--gpus', type=str, default='2,3')
args.add_argument('--skip', type=int, default=2)
args.add_argument('--aug', type=str, default='min')


def window_dataset_from_list(padded_x_list, padded_y_list, 
                             pad_size, step_size, 
                             batch_per_node, 
                             train=False, 
                             cval=0.):
    dataset = tf.data.Dataset.from_tensor_slices((padded_x_list, padded_y_list))
    
    if train:
        dataset = dataset.repeat().shuffle(buffer_size=1000000) 
        dataset = dataset.batch(batch_per_node)
        dataset = dataset.prefetch(AUTOTUNE)
    else:
        dataset = dataset.batch(batch_per_node, drop_remainder=False)
    return dataset


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
    ORG_PATH = '/datasets/ai_challenge/' # inside a docker container
    if not os.path.isdir(ORG_PATH): # outside... 
        ORG_PATH = '/media/data1/datasets/ai_challenge/'

    x, val_x = [], []
    y, val_y = [], []

    # 1.1 training set
    PATH = os.path.join(ORG_PATH, 'TIMIT_sound_norm')
    for snr in ['-15', '-5', '5', '15']:
        x += pickle.load(
            open(os.path.join(PATH, 
                              # f'train/snr{snr}_10_no_noise_aug.pickle'),
                              f'snr{snr}_10_no_noise_aug.pickle'),
                 'rb'))
        y += pickle.load(
            open(os.path.join(PATH, f'label_10.pickle'), 'rb'))

    # 1.2 validation set
    PATH = os.path.join(ORG_PATH, 'TIMIT_noisex_norm')
    for snr in ['-20', '-10', '0', '10', '20']:
        val_x += pickle.load(
            open(os.path.join(PATH, f'test/snr{snr}.pickle'), 'rb'))

        val_y += pickle.load(open(os.path.join(PATH, 'test/phn.pickle'), 'rb'))

    # 1.3 fix mismatch 
    for i in range(len(x)):
        x[i] = x[i][:, :len(y[i])]

    """ MODEL """
    # 2. model definition
    with strategy.scope(): 
        time, freq = WINDOW_SIZE, x[0].shape[0]
        input_shape = (WINDOW_SIZE, freq)

        print(config.model)
        if config.pretrain != '':
            model = tf.keras.models.load_model(config.pretrain, compile=False)
        else:
            model = getattr(models, config.model)(
                input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        model.summary()

        model.compile(optimizer=SGD(config.lr, momentum=0.9),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'AUC'])

    """ DATA """
    # 3. DATA PRE-PROCESSING
    x = np.concatenate(
        list(map(preprocess_spec(config, skip=config.skip), x)), axis=0)
    val_x = np.concatenate(
        list(map(preprocess_spec(config), val_x)), axis=0)

    # 3.1 sequence to window
    if model.output_shape[-1] != 1: # win2win
        y = np.concatenate(
            list(map(label_to_window(config, skip=config.skip), y)), axis=0)
        val_y = np.concatenate(
            list(map(label_to_window(config), val_y)), axis=0)
    else: # win2one
        y = np.concatenate(y, axis=0)
        val_y = np.concatenate(val_y, axis=0)
    print("data pre-processing finished")

    # 3.2 shuffling
    perm = np.random.permutation(len(x))
    x = np.take(x, perm, axis=0)
    y = np.take(y, perm, axis=0)
    print("shuffling training data finished")

    # 3.3 CVAL
    cval = getattr(x, config.aug)()
    print(f'CVAL: {cval}')

    """ TRAINING """
    with strategy.scope(): 
        # 4. train starts
        train_dataset = window_dataset_from_list(
            x, y, config.pad_size, config.step_size, BATCH_SIZE, 
            train=True, cval=cval) 
        val_dataset = window_dataset_from_list(
            val_x, val_y, config.pad_size, config.step_size, BATCH_SIZE,
            train=False)

        callbacks = [
            CSVLogger(config.name + '.log',
                      append=True),
            ReduceLROnPlateau(monitor='val_AUC',
                              factor=0.9,
                              patience=1,
                              mode='max',
                              verbose=1,
                              min_lr=1e-5),
            EarlyStopping(monitor='val_AUC',
                          mode='max',
                          patience=3),
            ModelCheckpoint(config.name+'.h5',
                            monitor='val_AUC',
                            mode='max',
                            save_best_only=True),
            TerminateOnNaN(),
        ]

        model.fit(x, y, # train_dataset,
                  epochs=TOTAL_EPOCH,
                  validation_data=val_dataset,
                  batch_size=BATCH_SIZE,
                  # steps_per_epoch=len(x)//BATCH_SIZE,
                  callbacks=callbacks)
