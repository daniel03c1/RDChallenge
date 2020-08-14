import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *

from efficientnet.model import EfficientNetB0
from utils import *
from transforms import *
from swa import SWA


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='model_0')

# HYPER_PARAMETERS
args.add_argument('--epochs', type=int, default=250)
args.add_argument('--batch_size', type=int, default=128)
args.add_argument('--win_size', type=int, default=128)
args.add_argument('--n_chan', type=int, default=2)


def framewise_augment(win_size):
    def augment(specs, labels):
        print(specs.shape)
        specs = mask(specs, axis=0, max_mask_size=win_size//4) # time
        specs = mask(specs, axis=1, max_mask_size=32) # freq
        # specs = random_shift(specs, axis=1, width=16)
        specs, labels = random_magphase_flip(specs, labels)
        return specs, labels
    return augment


def make_framewise_dataset(specs, labels, 
                           batch_size,
                           train=False, 
                           win_size=32,
                           n_chan=2, **kwargs):
    dataset = tf.data.Dataset.from_generator(
            window_generator(specs, labels, win_size, **kwargs),
            (tf.float32, tf.int32),
            (tf.TensorShape([win_size, 257, n_chan*2]), tf.TensorShape([11])))
    if train:
        dataset = dataset.map(framewise_augment(win_size),
                              num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=len(specs)//win_size*2)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if train:
        dataset = dataset.map(magphase_mixup(alpha=2.))
    dataset = dataset.map(log_magphase)
    return dataset 


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    WINDOW_SIZE = config.win_size
    N_CHAN = config.n_chan
    N_CLASSES = 11
    VAL_SPLIT = 0.1

    with strategy.scope():
        """ MODEL """
        x = tf.keras.layers.Input(
            shape=(WINDOW_SIZE, 257, N_CHAN*2))
        model = EfficientNetB0(weights=None,
                               input_tensor=x,
                               classes=N_CLASSES, 
                               backend=tf.keras.backend,
                               layers=tf.keras.layers,
                               models=tf.keras.models,
                               utils=tf.keras.utils,
                               )

        opt = Adam()
        model.compile(optimizer=opt, 
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'AUC'])
        model.summary()

        """ DATA """
        # 1. IMPORTING TRAINING DATA & PRE-PROCESSING
        PATH = '/datasets/ai_challenge/interspeech20/'
        x = np.load(os.path.join(PATH, 'cont_train_x.npy'))
        y = np.load(os.path.join(PATH, 'cont_train_y.npy'))
        test_x = np.load(os.path.join(PATH, 'cont_test_x.npy'))
        test_y = np.load(os.path.join(PATH, 'cont_test_y.npy'))
        print('data loading finished')
        
        # pre-process
        y = degree_to_class(y, one_hot=True).astype(np.float32)
        test_y = degree_to_class(test_y, one_hot=True).astype(np.float32)

        # 2. SPLITTING TRAIN AND VALIDATION SET
        train_size = len(x)
        val_size = int(train_size * VAL_SPLIT)

        x, val_x = x[:-val_size], x[-val_size:]
        y, val_y = y[:-val_size], y[-val_size:]
        print(x.shape, y.shape)
        print(val_x.shape, val_y.shape)
        print('data spliting finished')

        """ TRAINING """
        train_dataset = make_framewise_dataset(
            x, y, 
            batch_size=BATCH_SIZE,
            train=True,
            win_size=WINDOW_SIZE,
        )
        val_dataset = make_framewise_dataset(
            val_x, val_y, 
            batch_size=BATCH_SIZE,
            train=False,
            win_size=WINDOW_SIZE,
            infinite=False,
        )

        callbacks = [
            EarlyStopping(monitor='val_auc',
                          mode='max',
                          patience=32),
            CSVLogger(config.name + '.log',
                      append=True),
            ReduceLROnPlateau(monitor='auc',
                              factor=0.9,
                              patience=8,
                              mode='max'),
            SWA(start_epoch=75, swa_freq=2),
            ModelCheckpoint(config.name+'.h5',
                            monitor='val_auc',
                            mode='max',
                            save_best_only=True),
            TerminateOnNaN()
        ]

        model.fit(train_dataset,
                  epochs=TOTAL_EPOCH,
                  validation_data=val_dataset,
                  steps_per_epoch=train_size//BATCH_SIZE//WINDOW_SIZE*2,
                  callbacks=callbacks)

