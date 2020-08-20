import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import *
from utils import *
from transforms import log_magphase

from efficientnet.model import EfficientNetB0

# disable GPU
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

args = argparse.ArgumentParser()
args.add_argument('--saved_model', type=str, required=True)
args.add_argument('--window_size', type=int, default=128)
args.add_argument('--freq_first', type=bool, default=False)
args.add_argument('--dataset', type=str, default='challenge',
                  choices=['challenge', 'our'])


if __name__ == '__main__':
    config = args.parse_args()

    SAVED_MODEL_PATH = config.saved_model 
    N_CLASSES = 11

    # 1. Loading a saved model
    if config.freq_first:
        shape = (257, None, 4)
    else:
        shape = (None, 257, 4)
    x = tf.keras.layers.Input(shape=shape)
    model = EfficientNetB0(weights=None,
                           input_tensor=x,
                           classes=N_CLASSES, 
                           backend=tf.keras.backend,
                           layers=tf.keras.layers,
                           models=tf.keras.models,
                           utils=tf.keras.utils,
                           )
    model.load_weights(SAVED_MODEL_PATH)

    PATH = '/media/data1/datasets/ai_challenge/icassp/'

    # 2. loading evaluation dataset
    if config.dataset == 'challenge':
        eval_x = np.load(os.path.join(PATH, 'final_x.npy'))
        eval_y = np.load(os.path.join(PATH, 'final_y.npy'))
    else:
        eval_x = np.concatenate(
            [np.load(os.path.join(PATH, 'test_x.npy')),
             np.load(os.path.join(PATH, 'noise_test_x.npy'))],
            axis=0)
        eval_y = np.concatenate(
            [np.load(os.path.join(PATH, 'test_y.npy')),
             np.load(os.path.join(PATH, 'noise_test_y.npy'))],
            axis=0)

    eval_x = log_magphase(eval_x) 
    eval_y = degree_to_class(eval_y, one_hot=False)

    window = np.arange(config.window_size)

    # 3. predict
    for x, y in zip(eval_x, eval_y):
        x = np.transpose(x, (1, 0, 2)) # to time, freq, chan
        x = seq_to_windows(x, window=window)
        # (n_win, win, freq, chan) -> (n_win, freq, win, chan)
        if config.freq_first:
            x = np.transpose(x, (0, 2, 1, 3))
        pred_y = model.predict(x)
        mask = pred_y.max(axis=-1) > 0.5
        pred_y[mask, :-1] = 0
        pred_y = pred_y.argmax(axis=-1)
        print(y, pred_y)
        pred_y = pred_y[pred_y != N_CLASSES-1]
        import pdb;pdb.set_trace()

    n_classes = pred_y.shape[-1]
    pred_y = np.argmax(pred_y, axis=1)
    pred_y = class_to_azimuth(pred_y)
    pred_y = azimuth_to_classes(pred_y, N_CLASSES, one_hot=False)

    print("GROUND TRUTH\n", eval_y)
    print("PREDICTIONS\n", pred_y)

    print("Accuracy:", Accuracy()(eval_y, pred_y).numpy())
    if N_CLASSES == 2:
        print("Precision:", Precision()(eval_y, pred_y).numpy())
        print("Recall:", Recall()(eval_y, pred_y).numpy())
    else:
        print("SCORE:", 
              score(class_to_azimuth(eval_y),
                    class_to_azimuth(pred_y)).numpy())
    print(confusion_matrix(eval_y, pred_y))
