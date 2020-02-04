import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import *
from utils import *


# disable GPU
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

args = argparse.ArgumentParser()
args.add_argument('--saved_model', type=str, required=True)
args.add_argument('--norm', type=bool, default=False)
args.add_argument('--verbose', type=bool, default=False)
args.add_argument('--task', type=str, default='vad',
                  choices=('vad', 'both'))
args.add_argument('--dataset', type=str, default='challenge',
                  choices=['challenge', 'our'])


if __name__ == '__main__':
    config = args.parse_args()

    SAVED_MODEL_PATH = config.saved_model 

    if config.task == 'vad':
        N_CLASSES = 2
    else:
        N_CLASSES = 11

    # 1. Loading a saved model
    model = tf.keras.models.load_model(SAVED_MODEL_PATH, compile=False)
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
             np.load(os.path.join(PATH, 'noise_only_x.npy'))[66:]],
            axis=0)
        eval_y = np.concatenate(
            [np.load(os.path.join(PATH, 'test_y.npy')),
             np.load(os.path.join(PATH, 'noise_only_y.npy'))[66:]],
            axis=0)
    eval_x = normalize_spec(eval_x, norm=config.norm)
    eval_y = azimuth_to_classes(eval_y, N_CLASSES, one_hot=False)

    # 3. predict
    pred_y = model.predict(eval_x)
    if config.verbose:
        print(pred_y[:5])
        print(np.max(pred_y, axis=1))

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
