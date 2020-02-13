import argparse
import keras
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

from models import dense_net_based_model, model_two
from utils import time_mask, freq_mask, random_equalizer


AUTOTUNE = tf.data.experimental.AUTOTUNE
EPSILON = 1e-8
LOG_EPSILON = np.log(EPSILON)


args = argparse.ArgumentParser()
args.add_argument('--model_name', type=str, default='model')
args.add_argument('--pretrain', type=str, default='')
args.add_argument('--window_size', type=int, default=31)
args.add_argument('--ratio', type=float, default=1.)
args.add_argument('--type', type=str, default='spec',
                  choices=['spec', 'melspec'])
args.add_argument('--runner', type=str, required=True,
                  choices=['daniel', 'docker'])
args.add_argument('--task', type=str, default='both',
                  choices=('vad', 'both'))


def preprocess_spec(spec, pad=0, melspec=False):
    ''' apply log on magnitude ''' 
    if melspec:
        spec = np.sqrt(spec)
    spec[:, :, (0, 1)] = np.log(spec[:, :, (0, 1)] + EPSILON)
    spec = np.pad(spec, ((0, 0), (pad, pad), (0, 0)), 'constant',
                  constant_values=LOG_EPSILON)
    return spec


def preprocess_label(label, pad=0, n_classes=11):
    assert n_classes in (2, 11)
    if n_classes == 2:
        label = (label >= 0).astype(np.int32)
        label = np.pad(label, pad, 'constant', constant_values=0)
    else:
        ''' No Voice(-1) -> 10, Voice(0:180:20) -> (0:9) '''
        mask = np.equal(label, -1).astype(np.int32)
        label = mask * 10 + (1-mask) * label//20
        label = np.pad(label, pad, 'constant', constant_values=10)
    return label


def window_dataset_from_list(padded_x_list, padded_y_list, 
                             window_size, batch_per_node, train=False):
    ''' from list of variable size data, generate windowed dataset '''
    assert window_size % 2, 'window size must be an even number'

    def gen_from_list(x_list, y_list):
        def _gen():
            while True:
                for x, y in zip(x_list, y_list):
                    start = np.random.randint(len(y)-window_size+1)
                    x = x[:, start:start+window_size]
                    y = y[start:start+window_size]
                    yield x, y
        return _gen

    def augment(x, y):
        x = freq_mask(x, x.shape[0]//8, 3)
        # x = time_mask(x, max_mask_size=window_size//2)
        # x = random_equalizer(x)
        return x, y

    freq, _, chan = padded_x_list[0].shape
    dataset = tf.data.Dataset.from_generator(
        gen_from_list(padded_x_list, padded_y_list),
        (tf.float32, tf.int32),
        (tf.TensorShape([freq, window_size, chan]), tf.TensorShape([window_size])))
    if train:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat().shuffle(buffer_size=len(padded_x_list)*2)
    return dataset.batch(batch_per_node, drop_remainder=True).prefetch(AUTOTUNE)


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    device_nums = (2, 3)
    devices = ['/device:GPU:{}'.format(i) for i in device_nums]
    strategy = tf.distribute.MirroredStrategy(devices)

    """ HYPER_PARAMETERS """
    BATCH_SIZE = 256 // len(devices) # per each GPU
    TOTAL_EPOCH = 500
    VAL_SPLIT = 0.15

    if config.task == 'vad':
        N_CLASSES = 2
    else:
        N_CLASSES = 11

    freq = int(config.ratio * 257)

    """ DATA """
    # 1. IMPORTING TRAINING DATA & PRE-PROCESSING
    if config.runner == 'docker':
        PATH = '/datasets/ai_challenge/icassp/'
    else:
        PATH = '/media/data1/datasets/ai_challenge/icassp/'

    x = 'x_8.pickle' if config.type == 'spec' else 'x_8_mel.pickle'
    x = pickle.load(open(os.path.join(PATH, x), 'rb'))
    y = pickle.load(open(os.path.join(PATH, 'y_8_abs.pickle'), 'rb'))

    # preprocess 
    pad = config.window_size // 2
    x = list(map(lambda x: preprocess_spec(x, pad, config.type=='melspec'), x))
    y = list(map(lambda y: preprocess_label(y, pad, N_CLASSES), y))

    # 2. SPLITTING TRAIN AND VALIDATION SET
    val_size = int(len(x) * VAL_SPLIT)

    x, val_x = x[:-val_size], x[-val_size:]
    y, val_y = y[:-val_size], y[-val_size:]

    with strategy.scope():
        """ MODEL """
        if len(config.pretrain) == 0:
            freq, time, chan = x[0].shape

            '''
            model = dense_net_based_model(
                input_shape=(freq, config.window_size, chan),
                n_classes=N_CLASSES,
                n_layer_per_block=[4, 6, 10, 6],
                growth_rate=12,
                activation='softmax',
                kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            '''
            model = model_two(
                input_shape=(freq, config.window_size, chan),
                n_classes=N_CLASSES,
                n_layer_per_block=[4, 6, 10, 6],
                growth_rate=12,
                kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        else:
            model = tf.keras.models.load_model(config.pretrain, compile=False)

        lr = tf.optimizers.schedules.PiecewiseConstantDecay(
                [750, 75000, 75000],
                [0.1, 0.01, 0.001, 0.0001])
        opt = SGD(lr, momentum=0.9, nesterov=True)
        # opt = Adam()

        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=opt, 
                      loss=loss,
                      metrics=['accuracy'])
        model.summary()

        """ TRAINING """
        train_dataset = window_dataset_from_list(
            x, y, config.window_size, BATCH_SIZE, train=True)
        val_dataset = window_dataset_from_list(
            val_x, val_y, config.window_size, BATCH_SIZE, train=False)

        callbacks = [
            EarlyStopping(monitor='val_accuracy',
                          patience=25),
            CSVLogger(config.model_name + '.log',
                      append=True),
            ModelCheckpoint(config.model_name+'.h5',
                            monitor='val_accuracy',
                            save_best_only=True),
            TerminateOnNaN()
        ]

        model.fit(train_dataset,
                  epochs=TOTAL_EPOCH,
                  validation_data=val_dataset,
                  steps_per_epoch=len(x)//BATCH_SIZE * 10,
                  validation_steps=len(val_x)//BATCH_SIZE * 5,
                  callbacks=callbacks)
        
