import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.experimental import CosineDecay, CosineDecayRestarts
import keras

from models import dense_net_based_model
from utils import *
from swa import SWA


args = argparse.ArgumentParser()
args.add_argument('--model_name', type=str, default='model')
args.add_argument('--pretrain', type=str, default='')
args.add_argument('--norm', type=bool, default=False)
args.add_argument('--cls_weights', type=bool, default=False)
args.add_argument('--augment', type=bool, default=True)
args.add_argument('--mask', type=bool, default=True)
args.add_argument('--equalizer', type=bool, default=False)
args.add_argument('--roll', type=bool, default=True)
args.add_argument('--flip', type=bool, default=False)
args.add_argument('--se', type=bool, default=False)
args.add_argument('--ratio', type=float, default=1.)
args.add_argument('--task', type=str, required=True, 
                  choices=('vad', 'both'))
args.add_argument('--type', type=str, required=True, 
                  choices=('spec', 'mel'))


if __name__ == "__main__":
    config = args.parse_args()
    print(config)
    devices = ['/device:GPU:{}'.format(i) for i in (0, 1, 2, 3)]
    strategy = tf.distribute.MirroredStrategy(devices)

    """ HYPER_PARAMETERS """
    BATCH_SIZE = 256 // len(devices) # per each GPU
    TOTAL_EPOCH = 150
    VAL_SPLIT = 0.15

    if config.task == 'vad':
        N_CLASSES = 2
    else:
        N_CLASSES = 11

    freq = int(config.ratio * 257)

    """ DATA """
    # 1. IMPORTING TRAINING DATA & PRE-PROCESSING
    PATH = '/datasets/ai_challenge/icassp/'
    prefixes = [
        'train_', 'gen_', 'new_train_', 'new_train_', 'new_train2_', 'noise_train_']

    x = [prefix+'x.npy' if config.type=='spec' else prefix+'x_mel.npy'
         for prefix in prefixes]
    x = [np.load(os.path.join(PATH, _x)) for _x in x]
    x = np.concatenate(x, axis=0)

    y = [prefix+'y.npy' for prefix in prefixes]
    y = [np.load(os.path.join(PATH, _y)) for _y in y]
    y = np.concatenate(y, axis=0)
    
    # split input data
    x = x[:, :int(config.ratio * 257)]

    # aggregate and normalize
    x = normalize_spec(x, norm=config.norm)
    y = azimuth_to_classes(y, N_CLASSES, smoothing=True)

    # 2. SPLITTING TRAIN AND VALIDATION SET
    train_size = x.shape[0] - x.shape[0] % BATCH_SIZE
    perm = np.random.permutation(x.shape[0])[:train_size]
    val_size = BATCH_SIZE * int(train_size / BATCH_SIZE * VAL_SPLIT)

    x, val_x = x[perm[:-val_size]], x[perm[-val_size:]]
    y, val_y = y[perm[:-val_size]], y[perm[-val_size:]]

    # 3. Class Weights
    if config.cls_weights:
        class_weight = {i: (1/N_CLASSES) / np.mean(y==i) 
                        for i in range(N_CLASSES)}
    else:
        class_weight = None

    """ MODEL """
    with strategy.scope():
        if len(config.pretrain) == 0:
            freq, time, chan = x.shape[1:]

            model = dense_net_based_model(
                input_shape=(freq, None, chan),
                n_classes=N_CLASSES,
                n_layer_per_block=[4, 6, 10, 6],
                growth_rate=12,
                activation='softmax',
                se=config.se,
                kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        else:
            model = tf.keras.models.load_model(config.pretrain, compile=False)

        lr = tf.optimizers.schedules.PiecewiseConstantDecay(
                [750, 75000, 75000],
                [0.1, 0.01, 0.001, 0.0001])
        opt = SGD(lr, momentum=0.9, nesterov=True)

        model.compile(optimizer=opt, 
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'AUC'])
        model.summary()

    """ TRAINING """
    with strategy.scope():
        train_dataset = make_dataset(x, y, 
                                     n_proc=strategy.num_replicas_in_sync,
                                     batch_per_node=BATCH_SIZE,
                                     train=config.augment,
                                     mask=config.mask,
                                     equalizer=config.equalizer,
                                     roll=config.roll,
                                     flip=config.flip)
        val_dataset = make_dataset(val_x, val_y, 
                                   n_proc=strategy.num_replicas_in_sync,
                                   batch_per_node=BATCH_SIZE,
                                   train=False)

        callbacks = [
            # EarlyStopping(monitor='val_accuracy',
            #               patience=50 if config.norm else 25),
            CSVLogger(config.model_name + '.log',
                      append=True),
            SWA(start_epoch=80, swa_freq=2),
            ModelCheckpoint(config.model_name+'.h5',
                            monitor='val_accuracy',
                            save_best_only=True),
            TerminateOnNaN()
        ]

        model.fit(train_dataset,
                  epochs=TOTAL_EPOCH,
                  validation_data=val_dataset,
                  steps_per_epoch=x.shape[0]//BATCH_SIZE,
                  callbacks=callbacks,
                  class_weight=class_weight)

        model.evaluate(val_dataset, verbose=1)

        # For SWA
        repeat = x.shape[0] // BATCH_SIZE
        for x, y in train_dataset:
            model(x, training=True)
            repeat -= 1
            if repeat <= 0:
                break

        model.evaluate(val_dataset, verbose=1)
        model.save(config.model_name+"_SWA.h5")
