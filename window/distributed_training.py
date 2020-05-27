import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
import time
from functools import partial
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *

import models
from utils import *


AUTOTUNE = tf.data.experimental.AUTOTUNE


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, default='bdnn')
args.add_argument('--pretrain', type=str, default='')
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--model', type=str, default='bdnn')
args.add_argument('--lr', type=float, default=0.1)
args.add_argument('--gpus', type=str, default='2,3')
args.add_argument('--skip', type=int, default=2)
args.add_argument('--aug', type=str, default='min')


def make_callable(value):
    def _callable(inputs):
        return value
    return _callable


class Trainer:
    def __init__(self, 
                 model, 
                 optimizer, 
                 strategy, 
                 batch_size=256):
        self.model = model
        self.optimizer = optimizer
        self.strategy = strategy
        self.batch_size = batch_size

        # Loss
        self.loss = tf.keras.losses.binary_crossentropy

        # Metrics
        self.train_metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC()]
        self.test_metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC()]

        self.train_loss_metric = tf.keras.metrics.Sum()
        self.test_loss_metric = tf.keras.metrics.Sum()

        # experimental
        self.meta_opt = tf.keras.optimizers.SGD(0.01, momentum=0.9)
        self.noise = tf.ones([1, 80])

    @tf.function
    def distributed_train_epoch(self, dataset, testset, steps, interval):
        total_loss = 0.
        num_train_batches = 0.

        for i in range(steps):
            train = []
            for train_batch in train_set:
                train.append(train_batch)
                if len(train) >= interval:
                    break
            for val in testset:
                val = val
                break

            per_replica_loss = self.strategy.experimental_run_v2(
                self.train_step, args=(train, val))
            total_loss += strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            num_train_batches += 1

        return total_loss, num_train_batches

    def train_step(self, train_inputs, test_inputs):
        # import pdb; pdb.set_trace()
        test_image, test_label = test_inputs

        with tf.GradientTape() as tape, tf.GradientTape(persistent=True) as m_tape:
            tape.watch(self.noise)
            m_tape.watch(self.model.trainable_variables)

            # train
            # for image, label in train_inputs:
            image, label = train_inputs[0]
                # with tf.GradientTape(persistent=True) as m_tape:
            noise = tf.random.normal(image.shape[1:])
            noise = self.noise * noise
            image = image + tf.expand_dims(noise, axis=0)
            predictions = self.model(image, training=True)

            loss = tf.reduce_sum(self.loss(label, predictions)) \
                    / self.batch_size
            loss += (sum(self.model.losses) \
                    / self.strategy.num_replicas_in_sync)

            t_gradients = m_tape.gradient(loss, 
                                          self.model.trainable_variables)
            # self.optimizer.apply_gradients(
            #         zip(t_gradients, self.model.trainable_variables))
            for g, v in zip(t_gradients, self.model.trainable_variables):
                # tf.compat.v1.assign_sub(v, g * 0.1, use_locking=True)
                v -= g* 0.1

            # val
            predictions = self.model(test_image, training=False)
            val_loss = tf.reduce_sum(self.loss(test_label, predictions)) \
                    / self.batch_size
      
        # gradients = tape.gradient(val_loss, self.noise)
        # import pdb; pdb.set_trace()
        # self.meta_opt.apply_gradients((gradients, self.noise))# zip(gradients, self.noise))        

        for metric in self.train_metrics:
            metric(label, predictions)
        self.train_loss_metric(loss)
        return loss

    @tf.function
    def distributed_test_epoch(self, dataset):
        num_test_batches = 0.

        for minibatch in dataset:
            strategy.experimental_run_v2(self.test_step, args=(minibatch,))
            num_test_batches += 1
        return self.test_loss_metric.result(), num_test_batches

    def test_step(self, inputs):
        image, label = inputs
        predictions = self.model(image, training=False)
        unscaled_test_loss = self.loss(label, predictions) \
                             + sum(self.model.losses)

        for metric in self.test_metrics:
            metric(label, predictions)
        self.test_loss_metric(unscaled_test_loss)

    def fit(self, train_set, val_set, test_set, steps, interval, lr_scheduler=0.01):
        if not callable(lr_scheduler):
            lr_scheduler = make_callable(lr_scheduler)

        timer = time.time()
        for step in range(steps//interval//1000):
            self.optimizer.learning_rate = lr_scheduler(steps)

            train_loss, num_train_batches = self.distributed_train_epoch(
                train_set, val_set, 1000, interval)
            test_loss, num_test_batches = self.distributed_test_epoch(test_set)

            print('Epoch: {}, Train Loss: {}, Train Accuracy: {}, '
                  'Test Loss: {}, Test Accuracy {}, {} seconds'.format(
                      epoch,
                      train_loss / num_train_batches,
                      self.train_metrics[0].result(),
                      test_loss / num_test_batches,
                      self.test_metrics[0].result(),
                      time.time() - timer))
            if len(self.train_metrics) > 1:
                print([m.result() for m in self.train_metrics[1:]])

            if len(self.test_metrics) > 1:
                print([m.result() for m in self.test_metrics[1:]])

            timer = time.time()

            if epoch != epochs - 1: # End of Epochs
                for metric in self.train_metrics:
                    metric.reset_states()
                for metric in self.test_metrics:
                    metric.reset_states()

        result = [train_total_loss / num_train_batches]
        for metric in self.train_metrics:
            result.append(metric.result().numpy())
        result.append(test_total_loss / num_test_batches)
        for metric in self.test_metrics:
            result.append(metric.result().numpy())

        return result


def prepare_dataset(dataset, is_train, batch_size, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    if repeat:
        dataset = dataset.repeat()
    if is_train:
        dataset = dataset.shuffle(buffer_size=50000)
    return dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)


if __name__ == '__main__':
    config = args.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    strategy = tf.distribute.MirroredStrategy()
    print(config)

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
    with strategy.scope(): 
        PATH = os.path.join(ORG_PATH, 'TIMIT_sound_norm')
        for snr in ['-15']: # , '-5', '5', '15']:
            x += pickle.load(
                open(os.path.join(PATH, 
                                  # f'train/snr{snr}_10_no_noise_aug.pickle'),
                                  f'snr{snr}_10_no_noise_aug.pickle'),
                     'rb'))
            y += pickle.load(
                open(os.path.join(PATH, f'label_10.pickle'), 'rb'))

        # 1.2 validation set
        PATH = os.path.join(ORG_PATH, 'TIMIT_noisex_norm')
        for snr in ['-20']: # , '-10', '0', '10', '20']:
            val_x += pickle.load(
                open(os.path.join(PATH, f'test/snr{snr}.pickle'), 'rb'))

            val_y += pickle.load(open(os.path.join(PATH, 'test/phn.pickle'), 'rb'))

        # 1.3 fix mismatch 
        for i in range(len(x)):
            x[i] = x[i][:, :len(y[i])]

        """ MODEL """
        # 2. model definition
        input_shape = (WINDOW_SIZE, x[0].shape[0])

        print(config.model)
        if config.pretrain != '':
            model = tf.keras.models.load_model(config.pretrain, compile=False)
        else:
            model = getattr(models, config.model)(
                input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        model.summary()
        optimizer = SGD(0.1, momentum=0.9)

        """ DATA """
        # 3. DATA PRE-PROCESSING
        x = np.concatenate(
            list(map(preprocess_spec(config, skip=config.skip), x)), axis=0)
        val_x = np.concatenate(
            list(map(preprocess_spec(config), val_x)), axis=0)

        # 3.1 sequence to window
        if model.output_shape[-1] != 1: # win2win
            y = np.concatenate(
                list(map(label_to_window(config, skip=config.skip), y)), axis=0)
            val_y = np.concatenate(
                list(map(label_to_window(config), val_y)), axis=0)
        else: # win2one
            y = np.concatenate(y, axis=0)
            val_y = np.concatenate(val_y, axis=0)
        print("data pre-processing finished")

        # 3.2 shuffling
        perm = np.random.permutation(len(x))
        x = np.take(x, perm, axis=0)
        y = np.take(y, perm, axis=0)
        print("shuffling training data finished")

        # 3.3 CVAL
        cval = getattr(x, config.aug)()
        print(f'CVAL: {cval}')

        train_set = prepare_dataset((x, y), True, BATCH_SIZE, repeat=True)
        val_set = prepare_dataset((val_x, val_y), True, BATCH_SIZE, repeat=True)
        test_set = prepare_dataset((val_x, val_y), False, BATCH_SIZE)

        trainer = Trainer(model, optimizer, strategy, BATCH_SIZE)
        trainer.fit(train_set, val_set, test_set, 100000, 32, lr_scheduler=0.1)
    model.save('dist_model.h5')
