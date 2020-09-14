import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *

import efficientnet.model as model
from swa import SWA
from transforms import *
from utils import *


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='model_0')
args.add_argument('--model', type=str, default='EfficientNetB0')
args.add_argument('--optimizer', type=str, default='adam',
                                 choices=['adam', 'sgd', 'rmsprop'])
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--lr_factor', type=float, default=0.9)
args.add_argument('--lr_patience', type=int, default=2)
args.add_argument('--pretrain', type=bool, default=False)
args.add_argument('--cls_weights', type=bool, default=False)

# HYPER_PARAMETERS
args.add_argument('--epochs', type=int, default=250)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--n_chan', type=int, default=2)

# AUGMENTATION
args.add_argument('--alpha', type=float, default=0.75)
args.add_argument('--l2', type=float, default=0)


def augment(specs, labels, time_axis=1, freq_axis=0):
    specs = mask(specs, axis=time_axis, max_mask_size=32) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=24) # freq
    specs = random_shift(specs, axis=freq_axis, width=8)
    specs, labels = random_magphase_flip(specs, labels)
    return specs, labels


def make_dataset(specs, labels, 
                 batch_size,
                 alpha=1,
                 **kwargs):
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).cache()

    dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat().shuffle(buffer_size=len(x)//2)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.map(interbinary(mixup(alpha=alpha)))
    dataset = dataset.map(minmax_norm_magphase)
    dataset = dataset.map(log_magphase)
    return dataset


def cosine_decay_with_warmup(warm_steps, lr, total_steps):
    cosine_decay = tf.keras.experimental.CosineDecay(
        lr, total_steps - warm_steps, alpha=lr*1e-4)

    def scheduler(step):
        if step < warm_steps:
            return lr * (step / warm_steps)
        return cosine_decay(step - warm_steps)

    return scheduler


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'
    N_CLASSES = 11
    VAL_SPLIT = 0.05 # 0.1

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

        config.lr = cosine_decay_with_warmup(config.steps_per_epoch * 5,
                                             config.lr,
                                             config.steps_per_epoch * config.steps)
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
        model.summary()
        
        if config.pretrain:
            model.load_weights(NAME)
            print('loadded pretrained model')

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

        val_x = minmax_norm_magphase(val_x)
        val_x = log_magphase(val_x)
        test_x = minmax_norm_magphase(test_x)
        test_x = log_magphase(test_x)

        # FOR TESTING
        max_len = max([_x.shape[2] for _x in [val_x, test_x]])
        val_x = [np.pad(_x, ((0, 0), (0, 0), (0, max_len-_x.shape[2]), (0, 0)))
                  for _x in [val_x, test_x]]
        val_x = np.concatenate(val_x, axis=0)
        val_y = np.concatenate([val_y, test_y], axis=0)

        """ TRAINING """
        train_dataset = make_dataset(x, y, 
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
            # EarlyStopping(monitor='val_auc',
            #               patience=50,
            #               mode='max'),
            CSVLogger(NAME.replace('.h5', '.log'),
                      append=True),
            # ReduceLROnPlateau(monitor='auc',
            #                   factor=config.lr_factor,
            #                   patience=config.lr_patience,
            #                   mode='max'),
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
                  validation_data=(val_x, val_y),
                  steps_per_epoch=x.shape[0]//BATCH_SIZE,
                  class_weight=class_weight,
                  callbacks=callbacks)

        result = model.evaluate(test_x, test_y, verbose=1)
        with open(NAME.replace('.h5', '.log'), 'a') as f:
            f.write(f'\n{result}\n')

    # For SWA
    repeat = x.shape[0] // BATCH_SIZE
    for x, y in train_dataset:
        model(x, training=True)
        repeat -= 1
        if repeat <= 0:
            break

    model.evaluate(test_x, test_y, verbose=1)
    model.save(NAME.replace('.h5', "_SWA.h5"))
