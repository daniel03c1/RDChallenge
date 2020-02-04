import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from multiprocessing import Pool
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
args.add_argument('--se', type=bool, default=False)
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

    """ MODEL """
    with strategy.scope():
        if len(config.pretrain) == 0:
            model = dense_net_based_model(
                input_shape=(257, None, 4),
                n_classes=N_CLASSES,
                n_layer_per_block=[4, 6, 10, 6],
                growth_rate=12,
                activation='softmax',
                se=config.se,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        else:
            model = tf.keras.models.load_model(config.pretrain, compile=False)

        if config.pretrain:
            init_lr = 5e-3
        else:
            init_lr = 1e-2
        opt = Adam(CosineDecayRestarts(init_lr, 2, t_mul=1.5, m_mul=0.5, 
                                       alpha=0.01),
                   clipnorm=0.1)

        '''
        def loss_fn(label_smoothing):
            def _loss(y_true, y_pred):
                return categorical_crossentropy(y_true, y_pred, 
                                                label_smoothing=label_smoothing)
            return _loss
        '''
        
        model.compile(optimizer=opt, 
                      loss='categorical_crossentropy', # loss=loss_fn(0.1),
                      metrics=['accuracy'])
        model.summary()


    """ DATA """
    # 1. IMPORTING TRAINING DATA & PRE-PROCESSING
    PATH = '/datasets/ai_challenge/icassp/'
    ORG_CNT = 3000 # 5000
    GEN_CNT = 5120

    # 1.1. original data
    x = np.load(os.path.join(PATH, 'train_x.npy'))[:ORG_CNT]
    y = np.load(os.path.join(PATH, 'train_y.npy'))[:ORG_CNT]

    noise_x = np.load(os.path.join(PATH, 'noise_only_x.npy'))[:66]
    x = np.concatenate([x, noise_x, noise_x[:, :, :, (1, 0, 3, 2)]], axis=0)
    noise_y = np.load(os.path.join(PATH, 'noise_only_y.npy'))[:66]
    y = np.concatenate([y, noise_y, noise_y], axis=0)
    _x, _y = x, y

    # 1.2. generated data (with different noises)
    x = np.load(os.path.join(PATH, 'gen_x.npy'))[GEN_CNT:]
    y = np.load(os.path.join(PATH, 'gen_y.npy'))[GEN_CNT:]

    x = np.concatenate([_x, x], axis=0)
    y = np.concatenate([_y, y], axis=0)

    # 1.3. 50 given samples
    _x = np.load(os.path.join(PATH, '50_x.npy'))
    _x = np.pad(_x, ((0, 0), (0, 0), (0, x.shape[2]-_x.shape[2]), (0, 0)),
                'constant')
    _y = np.load(os.path.join(PATH, '50_y.npy'))

    x = np.concatenate([_x, x], axis=0)
    y = np.concatenate([_y, y], axis=0)
    
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
        train_dataset = make_dataset(x, y, 
                                     n_proc=strategy.num_replicas_in_sync,
                                     batch_per_node=BATCH_SIZE,
                                     train=config.augment,
                                     mask=config.mask,
                                     equalizer=config.equalizer,
                                     roll=config.roll)
        val_dataset = make_dataset(val_x, val_y, 
                                   n_proc=strategy.num_replicas_in_sync,
                                   batch_per_node=BATCH_SIZE,
                                   train=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                             patience=50 if config.norm else 25),
            tf.keras.callbacks.CSVLogger(config.model_name + '.log',
                                         append=True),
            tf.keras.callbacks.ModelCheckpoint(config.model_name+'.h5',
                                               monitor='val_accuracy',
                                               save_best_only=True),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        model.fit(train_dataset,
                  epochs=TOTAL_EPOCH,
                  validation_data=val_dataset,
                  steps_per_epoch=x.shape[0]//BATCH_SIZE,
                  callbacks=callbacks,
                  class_weight=class_weight)
