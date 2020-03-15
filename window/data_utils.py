import csv
import glob
import numpy as np
import os
import pickle
import shutil
import torch
import torch.nn.functional as F
import torchaudio
from multiprocessing import Pool
from tqdm import tqdm


def phn_to_label(phn_filename,
                 padding=0,
                 sync=True):
    '''
    phn_filename : STR -> file name (ends with PHN)
    padding : padding for the label (for both sides)
    sync : fitting the size of the label with the one of the real audio
    '''
    with open(phn_filename, 'r') as o:
        lines = list(map(lambda x: x.split(), o.readlines()))

    label = np.ones((int(lines[-1][1]),))
    for line in lines:
        if line[-1] == 'h#':
            label[int(line[0]):int(line[1])] = 0

    if padding > 0:
        pad = np.zeros((padding,))
        label = np.concatenate([pad, label, pad], axis=0)

    if sync:
        real_audio, _ = torchaudio.load(phn_filename.replace('.PHN',
                                                             '.wav'))
        label = np.concatenate([label, np.zeros((real_audio.shape[1]-len(label),))],
                               axis=0)

        assert len(label) == real_audio.shape[1]

    label = label.astype(np.uint8)
    np.save(phn_filename.replace('.PHN', ''), label)

    with open(phn_filename.replace('.PHN', '.csv'), 'w') as f:
        csv.writer(f).writerow(label)


def labels_to_pickle(folder, pickle_name):
    if not pickle_name.endswith('.pickle'):
        pickle_name += '.pickle'
    fnames = sorted(glob.glob(os.path.join(folder, '*.npy')))
    with Pool() as p:
        npys = p.map(raw_to_frames, tqdm(fnames))
    pickle.dump(npys, open(pickle_name, 'wb'))


def raw_to_frames(fname):
    raw = np.load(fname)
    raw = torch.Tensor(raw[None, None, :])
    frames = F.avg_pool1d(raw, 512, 256, ceil_mode=True)
    frames = torch.squeeze(frames).numpy()
    frames = np.greater(frames, 0.5).astype(np.float32)
    return frames


def wavs_to_pickle(folder, pickle_name):
    if not pickle_name.endswith('.pickle'):
        pickle_name += '.pickle'
    fnames = sorted(glob.glob(os.path.join(folder, '*.wav')))
    with Pool() as p:
        npys = p.map(wav_to_mel, tqdm(fnames))
    pickle.dump(npys, open(pickle_name, 'wb'))


def wav_to_mel(wav_name):
    wav, _ = torchaudio.load(wav_name)
    mel = torchaudio.transforms.MelSpectrogram(n_fft=512, n_mels=80)(wav)
    mel = torch.squeeze(mel)
    mel = mel[:, 1:].numpy() # remove first frame
    return mel


def fit_x_to_y(x_list, y_list):
    assert len(x_list) == len(y_list)

    def x_len(index):
        return x_list[i].shape[1]

    def y_len(index):
        return y_list[i].shape[0]

    for i in range(len(x_list)):
        diff = x_len(i) - y_len(i)
        if diff not in [0, 1]:
            temp = x_list[i]
            x_list[i] = x_list[i+1]
            x_list[i+1] = temp

        if x_len(i) == y_len(i) + 1: 
            x_list[i] = x_list[i][:, :-1]
            

if __name__ == '__main__':
    '''
    PATH = '/home/daniel/TIMIT_extended'
    for root, dirs, files in os.walk(PATH):
        for name in files:
            if name.endswith('.PHN'):
                phn_to_label(os.path.join(root, name), padding=32000)

    for folder in ['snr-5', 'snr0', 'snr5', 'snr10']:
        path = '/media/data1/datasets/ai_challenge/' \
                'TIMIT_NOISEX_extended/TEST/' + folder

        wavs_to_pickle(path, folder)

    path = '/media/data1/datasets/ai_challenge/' \
           'TIMIT_NOISEX_extended/TEST/label'
    labels_to_pickle(path, 'label')
    '''
    import os
    import pickle
    os.chdir('../../')
    x = pickle.load(open('snr0.pickle', 'rb'))
    y = pickle.load(open('label.pickle', 'rb'))

    fit_x_to_y(x, y)

    for i in range(len(x)):
        if x[i].shape[1] != y[i].shape[0]:
            print(i, x[i].shape, y[i].shape)
    print()

