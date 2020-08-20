import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *

from efficientnet.model import EfficientNetB0
from swa import SWA
from transforms import *
from utils import *


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='model_0')

# HYPER_PARAMETERS
args.add_argument('--epochs', type=int, default=250)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--n_chan', type=int, default=2)
args.add_argument('--lr', type=float, default=0.001)


def augment(specs, labels, time_axis=1, freq_axis=0):
    specs = mask(specs, axis=time_axis, max_mask_size=32) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=24) # freq
    specs = random_shift(specs, axis=freq_axis, width=8)
    specs, labels = random_magphase_flip(specs, labels)
    return specs, labels


def make_dataset(specs, labels, 
                 batch_size,
                 train=False, 
                 **kwargs):
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).cache()
    if train:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat().shuffle(buffer_size=len(x)//4)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if train:
        dataset = dataset.map(magphase_mixup(alpha=1.))
    dataset = dataset.map(minmax_norm_magphase)
    dataset = dataset.map(log_magphase)
    return dataset #.prefetch(AUTOTUNE)


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    N_CLASSES = 11
    VAL_SPLIT = 0.1

    with strategy.scope():
        """ MODEL """
        x = tf.keras.layers.Input(shape=(257, None, 4))
        model = EfficientNetB0(weights=None,
                               input_tensor=x,
                               classes=N_CLASSES, 
                               backend=tf.keras.backend,
                               layers=tf.keras.layers,
                               models=tf.keras.models,
                               utils=tf.keras.utils,
                               )

        opt = Adam(config.lr)
        model.compile(optimizer=opt, 
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'AUC'])
        model.summary()

        """ DATA """
        # 1. IMPORTING TRAINING DATA & PRE-PROCESSING
        PATH = '/datasets/ai_challenge/icassp'
        prefixes = ['gen'] # , 'noise_train']
        x = [np.load(os.path.join(PATH, f'{pre}_x.npy'))
             for pre in prefixes]
        y = [np.load(os.path.join(PATH, f'{pre}_y.npy'))
             for pre in prefixes]
        test_x = [np.load(os.path.join(PATH, '50_x.npy'))]
        test_y = [np.load(os.path.join(PATH, '50_y.npy'))]

        PATH = '/datasets/ai_challenge/interspeech20/'
        x.append(np.load(os.path.join(PATH, 'shuffled_train_x.npy')))
        y.append(np.load(os.path.join(PATH, 'shuffled_train_y.npy')))
        test_x.append(np.load(os.path.join(PATH, 'test_x.npy')))
        test_y.append(np.load(os.path.join(PATH, 'test_y.npy')))
        print('data loading finished')

        max_len = max([_x.shape[2] for _x in x])
        x = [np.pad(_x, ((0, 0), (0, 0), (0, max_len-_x.shape[2]), (0, 0)))
             for _x in x]
        max_len = max([_x.shape[2] for _x in test_x])
        test_x = [np.pad(_x, ((0, 0), (0, 0), (0, max_len-_x.shape[2]), (0, 0)))
                  for _x in test_x]
        print('padding finished')

        xs = np.concatenate(x, axis=0)
        ys = np.concatenate(y, axis=0)
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)
        del x[:], y[:]
        x, y = xs, ys
        print('data concatenating finished')
        
        # pre-process
        y = degree_to_class(y, one_hot=True).astype(np.float32)
        test_y = degree_to_class(test_y, one_hot=True).astype(np.float32)

        # 2. SPLITTING TRAIN AND VALIDATION SET
        train_size = len(x)
        val_size = int(train_size * VAL_SPLIT)
        indices = np.random.permutation(train_size)

        x, val_x = x[indices[:-val_size]], x[indices[-val_size:]]
        y, val_y = y[indices[:-val_size]], y[indices[-val_size:]]
        print('data spliting finished')

        val_x = log_magphase(val_x)
        test_x = log_magphase(test_x)

        """ TRAINING """
        train_dataset = make_dataset(x, y, 
                                     BATCH_SIZE,
                                     train=True)

        callbacks = [
            EarlyStopping(monitor='val_auc',
                          patience=50,
                          mode='max'),
            CSVLogger(config.name + '.log',
                      append=True),
            ReduceLROnPlateau(monitor='auc',
                              factor=0.9,
                              patience=4,
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
                  batch_size=BATCH_SIZE,
                  validation_data=(val_x, val_y),
                  steps_per_epoch=x.shape[0]//BATCH_SIZE,
                  callbacks=callbacks)

        result = model.evaluate(test_x, test_y, verbose=1)
        with open(config.name + '.log', 'a') as f:
            f.write(f'\n{result}')

    # For SWA
    repeat = x.shape[0] // BATCH_SIZE
    for x, y in train_dataset:
        model(x, training=True)
        repeat -= 1
        if repeat <= 0:
            break

    model.evaluate(test_x, test_y, verbose=1)
    model.save(config.name+"_SWA.h5")
