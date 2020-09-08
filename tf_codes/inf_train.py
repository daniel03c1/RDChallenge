import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from functools import partial

import efficientnet.model as model
from swa import SWA
from transforms import *
from utils import *


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='infinite_B0')
args.add_argument('--model', type=str, default='EfficientNetB0')
args.add_argument('--optimizer', type=str, default='adam',
                                 choices=['adam', 'sgd', 'rmsprop'])
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--lr_factor', type=float, default=0.9)
args.add_argument('--lr_patience', type=int, default=2)
args.add_argument('--pretrain', type=bool, default=False)
args.add_argument('--cls_weights', type=bool, default=False)

# HYPER_PARAMETERS
args.add_argument('--epochs', type=int, default=200)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--n_chan', type=int, default=2)
args.add_argument('--steps_per_epoch', type=int, default=100)

# AUGMENTATION
args.add_argument('--alpha', type=float, default=0.75)
args.add_argument('--l2', type=float, default=0)


def augment(specs, labels, time_axis=1, freq_axis=0):
    specs = mask(specs, axis=time_axis, max_mask_size=64) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=32) # freq
    specs = random_shift(specs, axis=freq_axis, width=8)
    specs, labels = random_magphase_flip(specs, labels)
    return specs, labels


def to_generator(dataset):
    def _gen():
        if isinstance(dataset, tuple):
            for z in zip(*dataset):
                yield z
        else:
            for data in dataset:
                yield data
    return _gen


def make_dataset(background, voice, label,
                 batch_size,
                 alpha=1,
                 **kwargs):
    freq, _, chan = background[0].shape
    b_dataset = tf.data.Dataset.from_generator(
        to_generator(background),
        tf.float32,
        tf.TensorShape([freq, None, chan]))
    b_dataset = b_dataset.repeat().shuffle(len(background))

    v_dataset = tf.data.Dataset.from_generator(
        to_generator((voice, label)),
        (tf.float32, tf.int32),
        (tf.TensorShape([freq, None, chan]), tf.TensorShape([])))
    v_dataset = v_dataset.repeat().shuffle(len(voice))

    dataset = tf.data.Dataset.zip((b_dataset, v_dataset))
    dataset = dataset.map(partial(merge_complex_specs,
                                  n_frame=300,
                                  prob=0.9,
                                  min_voice_ratio=2/3,
                                  speed_std=0.1))
    dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.map(interbinary(magphase_mixup(alpha=alpha, feat='complex')))
    dataset = dataset.map(minmax_norm_magphase)
    dataset = dataset.map(log_magphase)
    return dataset


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'
    N_CLASSES = 11

    with strategy.scope():
        """ MODEL """
        x = tf.keras.layers.Input(shape=(257, None, 4))
        model = getattr(model, config.model)(
            weights=None,
            input_tensor=x,
            classes=N_CLASSES, 
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils,
        )

        if config.optimizer == 'adam':
            opt = Adam(config.lr) 
        elif config.optimizer == 'sgd':
            opt = SGD(config.lr, momentum=0.9)
        else:
            opt = RMSprop(config.lr, momentum=0.9)

        if config.l2 > 0:
            model = apply_kernel_regularizer(
                model, tf.keras.regularizers.l2(config.l2))
        model.compile(optimizer=opt, 
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'AUC'])
        # model.summary()
        
        if config.pretrain:
            model.load_weights(NAME)
            print('loadded pretrained model')

        """ DATA """
        # TRAINING DATA
        PATH = '/codes/generate_wavs'
        backgrounds = pickle.load(
            open(os.path.join(PATH, 'drone_complex_norm100_specs.pickle'), 'rb'))
        voices = pickle.load(
            open(os.path.join(PATH, 'voice_complex_norm100_specs.pickle'), 'rb'))
        labels = np.load(os.path.join(PATH, 'voice_labels.npy'))

        # TESTING DATA
        PATH = '/datasets/ai_challenge/icassp'
        test_x = [np.load(os.path.join(PATH, '50_x.npy'))]
        test_y = [np.load(os.path.join(PATH, '50_y.npy'))]

        PATH = '/datasets/ai_challenge/interspeech20/'
        test_x.append(np.load(os.path.join(PATH, 'test_x.npy')))
        test_y.append(np.load(os.path.join(PATH, 'test_y.npy')))
        print('data loading finished')

        max_len = max([_x.shape[2] for _x in test_x])
        test_x = [np.pad(_x, ((0, 0), (0, 0), (0, max_len-_x.shape[2]), (0, 0)))
                  for _x in test_x]
        print('padding finished')

        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)
        print('data concatenating finished')
        
        test_x = minmax_norm_magphase(test_x)
        test_x = log_magphase(test_x)
        test_y = degree_to_class(test_y, one_hot=True).astype(np.float32)
        print('preprocessing test dataset finished')

        """ TRAINING """
        train_dataset = make_dataset(backgrounds, voices, labels,
                                     BATCH_SIZE,
                                     alpha=config.alpha)

        if config.cls_weights:
            class_weight = {
                i : (1/N_CLASSES) / np.mean(np.argmax(y, axis=-1)==i)
                for i in range(N_CLASSES)}
            print(class_weight)
        else:
            class_weight = None

        callbacks = [
            EarlyStopping(monitor='val_auc',
                          patience=50,
                          mode='max'),
            CSVLogger(NAME.replace('.h5', '.log'),
                      append=True),
            ReduceLROnPlateau(monitor='auc',
                              factor=config.lr_factor,
                              patience=config.lr_patience,
                              mode='max'),
            SWA(start_epoch=TOTAL_EPOCH//2, swa_freq=2),
            ModelCheckpoint(NAME,
                            monitor='val_auc',
                            mode='max',
                            save_best_only=True),
            TensorBoard(log_dir=f'./logs/{NAME.replace(".h5", "")}',
                        histogram_freq=0,
                        profile_batch=2),
            TerminateOnNaN()
        ]

        model.fit(train_dataset,
                  epochs=TOTAL_EPOCH,
                  batch_size=BATCH_SIZE,
                  validation_data=(test_x, test_y),
                  steps_per_epoch=config.steps_per_epoch,
                  class_weight=class_weight,
                  callbacks=callbacks)

        result = model.evaluate(test_x, test_y, verbose=1)
        with open(NAME.replace('.h5', '.log'), 'a') as f:
            f.write(f'\n{result}\n')

    # SWA
    repeat = x.shape[0] // BATCH_SIZE
    for x, y in train_dataset:
        model(x, training=True)
        repeat -= 1
        if repeat <= 0:
            break

    result = model.evaluate(test_x, test_y, verbose=1)
    with open(NAME.replace('.h5', '.log'), 'a') as f:
        f.write(f'\n{result}\n')
    model.save(NAME.replace('.h5', "_SWA.h5"))
