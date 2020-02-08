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


args = argparse.ArgumentParser()
args.add_argument('--model_name', type=str, default='model')
args.add_argument('--pretrain', type=str, default='')
args.add_argument('--norm', type=bool, default=False)
args.add_argument('--cls_weights', type=bool, default=False)
args.add_argument('--augment', type=bool, default=True)
args.add_argument('--mask', type=bool, default=True)
args.add_argument('--equalizer', type=bool, default=True)
args.add_argument('--roll', type=bool, default=True)
args.add_argument('--flip', type=bool, default=False)
args.add_argument('--se', type=bool, default=False)
args.add_argument('--ratio', type=float, default=1.)
args.add_argument('--task', type=str, required=True, 
                  choices=('vad', 'both'))


if __name__ == "__main__":
    config = args.parse_args()
    print(config)
    devices = ['/device:GPU:{}'.format(i) for i in (0, 1, 2, 3)]
    strategy = tf.distribute.MirroredStrategy(devices)

    """ HYPER_PARAMETERS """
    BATCH_SIZE = 64 # per each GPU
    TOTAL_EPOCH = 750
    VAL_SPLIT = 0.15

    if config.task == 'vad':
        N_CLASSES = 2
    else:
        N_CLASSES = 11

    freq = int(config.ratio * 257)

    """ DATA """
    # 1. IMPORTING TRAINING DATA & PRE-PROCESSING
    PATH = '/datasets/ai_challenge/icassp/'
    ORG_CNT = 5000
    GEN_CNT = 8192

    x = []
    y = []

    # 1.1. original data
    x.append(np.load(os.path.join(PATH, 'train_x.npy'))[:ORG_CNT])
    y.append(np.load(os.path.join(PATH, 'train_y.npy'))[:ORG_CNT])

    x.append(np.load(os.path.join(PATH, 'noise_only_x.npy'))[:66])
    y.append(np.load(os.path.join(PATH, 'noise_only_y.npy'))[:66])

    # 1.2. generated data (with different noises)
    x.append(np.load(os.path.join(PATH, 'gen_x.npy'))[:GEN_CNT])
    y.append(np.load(os.path.join(PATH, 'gen_y.npy'))[:GEN_CNT])

    # 1.3. new 1000 samples
    x.append(np.load(os.path.join(PATH, 'new_train_x.npy')))
    y.append(np.load(os.path.join(PATH, 'new_train_y.npy')))

    x.append(np.load(os.path.join(PATH, 'new_train_x2.npy')))
    y.append(np.load(os.path.join(PATH, 'new_train_y2.npy')))

    x = np.concatenate(x, axis=0)
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

        if config.pretrain:
            init_lr = 5e-3
        else:
            init_lr = 1e-2
        # lr = CosineDecayRestarts(init_lr, 2, t_mul=1.5, m_mul=0.5, alpha=0.01)
        lr = tf.optimizers.schedules.PiecewiseConstantDecay(
                [750, 75000, 75000],
                [0.1, 0.01, 0.001, 0.0001])
        # opt = SGD(lr, momentum=0.9, nesterov=True)
        '''
        lr = tf.optimizers.schedules.PiecewiseConstantDecay(
                [5000, 10000],
                [0.01, 0.001, 0.0005])
        '''
        opt = Adam(lr)

        model.compile(optimizer=opt, 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
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
            EarlyStopping(monitor='val_accuracy',
                          patience=50 if config.norm else 35),
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
                  steps_per_epoch=x.shape[0]//BATCH_SIZE,
                  callbacks=callbacks,
                  class_weight=class_weight)
