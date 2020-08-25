import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import *
from utils import *
from transforms import *
from metrics import score

import efficientnet.model as model

# disable GPU
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB0')
args.add_argument('--norm', type=bool, default=True)
args.add_argument('--verbose', type=bool, default=False)
args.add_argument('--dataset', type=str, default='challenge',
                  choices=['challenge', 'our'])


if __name__ == '__main__':
    config = args.parse_args()

    N_CLASSES = 11

    # 1. Loading a saved model
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
    model.load_weights(config.name)

    if config.verbose:
        model.summary()

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

    n_chan = eval_x.shape[-1] // 2
    if config.norm:
        eval_x = minmax_norm_magphase(eval_x)
    eval_x = log_magphase(eval_x)
    eval_y = degree_to_class(eval_y, one_hot=False)

    # 3. predict
    pred_y = model.predict(eval_x)
    if config.verbose:
        print(pred_y[:5])
        print(np.max(pred_y, axis=1))

    n_classes = pred_y.shape[-1]
    pred_y = np.argmax(pred_y, axis=-1)

    print("GROUND TRUTH\n", eval_y)
    print("PREDICTIONS\n", pred_y)

    print("Accuracy:", Accuracy()(eval_y, pred_y).numpy())
    print("SCORE:", 
          score(class_to_degree(eval_y),
                class_to_degree(pred_y)).numpy())
    print(confusion_matrix(eval_y, pred_y))
