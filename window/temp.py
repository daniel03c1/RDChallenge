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
args.add_argument('--names', type=str, default='bdnn_baseline')
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--gpus', type=str, default='2,3')
args.add_argument('--skip', type=int, default=2)


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
    PATH = os.path.join(ORG_PATH, 'TIMIT_noisex_norm')
    for snr in ['-15', '-5', '5', '15']:
        x += pickle.load(
            open(os.path.join(PATH, 
                              f'train/snr{snr}_10_no_noise_aug.pickle'),
                 'rb'))
        y += pickle.load(
            open(os.path.join(PATH, f'train/label_10.pickle'), 'rb'))

    # 1.2 validation set
    PATH = os.path.join(ORG_PATH, 'TIMIT_noisex_norm')
    for snr in ['-20', '-10', '0', '10', '20']:
        val_x += pickle.load(
            open(os.path.join(PATH, f'test/snr{snr}.pickle'), 'rb'))

        val_y += pickle.load(open(os.path.join(PATH, 'test/phn.pickle'), 'rb'))

    # 1.3 fix mismatch 
    for i in range(len(x)):
        x[i] = x[i][:, :len(y[i])]

    """ DATA """
    # 3. DATA PRE-PROCESSING
    x = np.concatenate(
        list(map(preprocess_spec(config, skip=config.skip), x)), axis=0)
    val_x = np.concatenate(
        list(map(preprocess_spec(config), val_x)), axis=0)

    # 3.1 sequence to window
    # window to window
    y = np.concatenate(
        list(map(label_to_window(config, skip=config.skip), y)), axis=0)
    val_y = np.concatenate(
        list(map(label_to_window(config), val_y)), axis=0)
    print("data pre-processing finished")

    # 3.2 shuffling
    perm = np.random.permutation(len(x))
    x = np.take(x, perm, axis=0)
    y = np.take(y, perm, axis=0)
    print("shuffling training data finished")

    """ TRAINING """
    for name in config.names.split(','):
        if name.endswith('.h5'):
            name = name.replace('.h5', '')
        print(name)

        # 4.1 calibrate BN
        model = tf.keras.models.load_model(name+'.h5', compile=False)
        model.compile(optimizer=SGD(0),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'AUC'])

        callbacks = [CSVLogger(name+'.log', append=True)]

        model.fit(x, y, epochs=1,
                  batch_size=BATCH_SIZE,
                  validation_data=(val_x, val_y),
                  shuffle=False,
                  callbacks=callbacks)

