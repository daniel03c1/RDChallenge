import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *

from models import dense_net_based_model
from efficientnet.model import EfficientNetB0
from utils import *
from swa import SWA


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='model_0')
args.add_argument('--gpus', type=str, default='0')


def augment(spec, label):
    spec = freq_mask(spec)
    spec = time_mask(spec)
    # spec = random_roll(spec)
    spec = freq_shift()(spec)
    spec, label = random_flip(spec, label)
    return spec, label


def make_dataset(x, y, n_proc, batch_per_node, train=False, **kwargs):
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).cache()
    if train:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat().shuffle(buffer_size=len(x))
    dataset = dataset.batch(batch_per_node, drop_remainder=True)
    if train:
        dataset = dataset.map(mixup())
    dataset = dataset.map(log)
    return dataset.prefetch(AUTOTUNE)


if __name__ == "__main__":
    config = args.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = 250
    BATCH_SIZE = 32
    N_CLASSES = 11
    VAL_SPLIT = 0.1

    with strategy.scope():
        """ MODEL """
        '''
        freq, time, chan = x.shape[1:]

        model = dense_net_based_model(
            input_shape=(freq, None, chan),
            n_classes=N_CLASSES,
            n_layer_per_block=[6, 12, 24, 16],
            growth_rate=24,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        '''
        x = tf.keras.layers.Input(shape=(257, None, 4))
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
        PATH = '/datasets/ai_challenge/icassp/'
        '''
        prefixes = [
            'train', 'gen', 'new_train', 'new_train', 'new_train2'] # , 'noise_train']

        x = np.concatenate(
            [np.load(os.path.join(PATH, f'{prefix}_x.npy')) for prefix in prefixes],
            axis=0)

        y = np.concatenate(
            [np.load(os.path.join(PATH, f'{prefix}_y.npy')) for prefix in prefixes],
            axis=0)
        '''
        test_x = [
            np.load(os.path.join(PATH, 'final_x.npy')),
            np.load(os.path.join(PATH, '50_x.npy')),
        ]
        test_y = [
            np.load(os.path.join(PATH, 'final_y.npy')),
            np.load(os.path.join(PATH, '50_y.npy')),
        ]

        PATH = '/datasets/ai_challenge/interspeech20/'
        x = np.load(os.path.join(PATH, 'shuffled_train_x.npy'))
        y = np.load(os.path.join(PATH, 'shuffled_train_y.npy'))
        test_x.append(np.load(os.path.join(PATH, 'test_x.npy')))
        test_y.append(np.load(os.path.join(PATH, 'test_y.npy')))

        target = max([f.shape for f in test_x])[1:]
        test_x = [
            np.pad(t_x, 
                [[0, 0]] + [[0, target[i]-t_x.shape[i+1]] for i in range(len(target))]) 
            for t_x in test_x]
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)
        print('data loading finished')
        
        # pre-process
        y = azimuth_to_classes(y, N_CLASSES, smoothing=False)
        test_y = azimuth_to_classes(test_y, N_CLASSES, smoothing=False)

        # 2. SPLITTING TRAIN AND VALIDATION SET
        train_size = x.shape[0] - x.shape[0] % BATCH_SIZE
        perm = np.random.permutation(x.shape[0])[:train_size]
        val_size = BATCH_SIZE * int(train_size / BATCH_SIZE * VAL_SPLIT)

        # x, val_x = x[perm[:-val_size]], x[perm[-val_size:]]
        # y, val_y = y[perm[:-val_size]], y[perm[-val_size:]]
        # importing already shuffled datasets
        # x, val_x = x[:-val_size], x[-val_size:]
        # y, val_y = y[:-val_size], y[-val_size:]
        val_x = test_x
        val_y = test_y
        print(x.shape, y.shape)
        print('data spliting finished')

        """ TRAINING """
        train_dataset = make_dataset(x, y, 
                                     n_proc=strategy.num_replicas_in_sync,
                                     batch_per_node=BATCH_SIZE,
                                     train=True)
        val_dataset = make_dataset(val_x, val_y, 
                                   n_proc=strategy.num_replicas_in_sync,
                                   batch_per_node=BATCH_SIZE,
                                   train=False)

        callbacks = [
            EarlyStopping(monitor='val_accuracy',
                          patience=32),
            CSVLogger(config.name + '.log',
                      append=True),
            ReduceLROnPlateau(monitor='accuracy',
                              factor=0.9,
                              patience=4,
                              mode='max'),
            SWA(start_epoch=75, swa_freq=2),
            ModelCheckpoint(config.name+'.h5',
                            monitor='val_accuracy',
                            save_best_only=True),
            TerminateOnNaN()
        ]

        model.fit(train_dataset,
                  epochs=TOTAL_EPOCH,
                  validation_data=val_dataset,
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
