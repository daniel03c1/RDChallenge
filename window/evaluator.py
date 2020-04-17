import argparse
import glob
import numpy as np
import os
import pickle
import tensorflow as tf
from tqdm import tqdm
from multiprocessing import Pool
from tqdm import tqdm
from utils import *
from data_utils import *


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--pad_size', type=int, default=19)
args.add_argument('--step_size', type=int, default=9)
args.add_argument('--snrs', type=str, default='none')
args.add_argument('--batch_size', type=int, default=8192)
args.add_argument('--gpus', type=str, default='0')


if __name__ == '__main__':
    config = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

    # 1. Loading a saved model
    model = tf.keras.models.load_model(config.name, compile=False)

    # 2. IMPORTING TRAINING DATA
    PATH = '/datasets/ai_challenge/' # inside a docker container
    if not os.path.isdir(PATH): # outside... 
        PATH = '/media/data1/datasets/ai_challenge/'
    path = os.path.join(PATH, 'TIMIT_NOISEX_extended/TEST')
    x, y = [], []

    snrs = config.snrs
    if snrs == 'none':
        snrs = '-20, -15, -5, 0, 5, 10'
    for snr in map(int, snrs.split(',')):
        x += pickle.load(open(os.path.join(path, f'snr{snr}_test.pickle'), 
                             'rb'))
        y += pickle.load(open(os.path.join(path, 'phn_test.pickle'), 'rb'))
    print("Loading x, y is complete")

    x = list(map(preprocess_spec(config), x))
    y = np.concatenate(y, axis=0)
    print("Preprocessing x, y is complete")

    # 3. Predictions
    predictions = [] 
    for windows in tqdm(x):
        pred = model.predict(windows, batch_size=config.batch_size, verbose=0)
        if model.output_shape[-1] != 1:
            pred = windows_to_sequence(pred, config.pad_size, config.step_size)
        predictions.append(pred)

    predictions = np.concatenate(predictions, axis=0)
    predictions = np.squeeze(predictions)

    acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
    acc.update_state(y, predictions)
    auc = tf.keras.metrics.AUC()
    auc.update_state(y, predictions)

    print('Test Accuracy: {}'.format(acc.result().numpy()))
    print('Test AUC: {}'.format(auc.result().numpy()))

