import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def vis(pattern='*.log', train=True, val=True):
    if isinstance(pattern, list):
        model_names = pattern
    else:
        model_names = sorted(glob.glob(pattern))
    print(model_names)
    n_models = len(model_names)
    fig, axes = plt.subplots(1, 3)

    for i in range(n_models):
        model_name = model_names[i]
        print(model_name)
        table = pd.read_csv(model_name)
        if train:
            axes[0].plot(table['loss'], label=model_name+'_train_loss')
            axes[1].plot(table['accuracy'], label=model_name+'_train_acc')
            axes[2].plot(table['AUC'], label=model_name+'_train_AUC')
        if val:
            axes[0].plot(table['val_loss'], label=model_name+'_val_loss')
            axes[1].plot(table['val_accuracy'], label=model_name+'_val_acc')
            axes[2].plot(table['val_AUC'], label=model_name+'_val_AUC')

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plt.show()


def visualize_logs(default_path=None, category='val_accuracy'):
    files = [f for f in os.listdir(default_path) if f.endswith('.log')]

    for f in files:
        d = pd.read_csv(f)
        plt.plot(d[category], label=f)
    plt.legend()
    plt.show()


"""
if __name__ == "__main__":
    PATH = '/media/volume1/ai_challenge/data2019/'
    MAX_NUM = 200
    t = np.load(PATH+'test_x.npy')[:MAX_NUM]
    tn = np.load(PATH+'noise_only_x.npy')[:MAX_NUM]
    t2 = np.load(PATH+'train_x2.npy')[:MAX_NUM]
    tn2 = np.load(PATH+'noise_only_x2.npy')[:MAX_NUM]
    c = np.load(PATH+'final_x.npy')
    cy = np.load(PATH+'final_y.npy')
    cn = c[cy == -1][:MAX_NUM]
    c = c[cy != -1][:MAX_NUM]

    '''
    data = (t, tn, t2, tn2, c, cn)
    x = np.concatenate(data, axis=0)
    y = np.zeros((x.shape[0],))
    offset = 0
    for i in range(len(data)):
        y[offset:] = i
        offset += len(data[i])
    print("Total {} samples".format(offset))

    t = t_sne(x[:, :, :60, :2].reshape((offset, -1)), perplexity=50)
    plt_t_sne(t, y, 't_sne_per150_1e4_t_t2_c_-60.png')
    '''

    violin_plot(list(map(lambda x: normalize_spec(x[:, :, :60]), [t, tn, c, cn, t2, tn2])),
                ['ORG_Voice', 'ORG_No_Voice', 'CHL_Voice', 'CHL_No_Voice',
                 'NEW_Voice', 'NEW_No_Voice'],
                'violinplot_part_-60.png')
"""
