import csv
import glob
import numpy as np
import os
import pickle
import shutil
import time
import torch
import torch.nn.functional as F
import torchaudio
from multiprocessing import Pool
from tqdm import tqdm


# preprocessing labels
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


def labels_to_pickle(folder, pickle_name=None):
    if pickle_name is None:
        pickle_name = folder
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


def frames_to_raw(win_size, step_size):
    def _frames_to_raw(fname):
        frames = np.load(fname)
        # raw = np.zeros((len(frames)-1)*step_size + win_size, dtype=np.float32)
        raw = np.zeros((len(frames))*step_size + win_size, dtype=np.float32)
        for i in range(len(frames)):
            raw[step_size*i: step_size*i+win_size] += frames[i]
        raw = np.greater_equal(raw, win_size/step_size/2)
        raw = raw.astype(np.float32)
        return raw
    return _frames_to_raw
       

# preprocessing wav files
def wavs_to_pickle(folder, pickle_name=None, feature_type='mel', n_procs=None):
    start = time.time()
    if feature_type == 'spec':
        transform = wav_to_spec
    elif feature_type == 'mel':
        transform = wav_to_mel
    elif feature_type == 'mfcc':
        transform = wav_to_mfcc
    else:
        raise ValueError(f'invalid feature type :{feature_type}')

    if pickle_name is None:
        pickle_name = folder
        if feature_type != 'mel':
            pickle_name += f'_{feature_type}'
    if not pickle_name.endswith('.pickle'):
        pickle_name += '.pickle'
    fnames = glob.glob(os.path.join(folder, '*.wav'))
    fnames.sort(key=lambda x: x[:x.rfind('_')])
    with Pool(n_procs) as p:
        npys = p.map(transform, tqdm(fnames))
    pickle.dump(npys, open(pickle_name, 'wb'))
    print(f'took {time.time() - start:.3f} secs')


def wav_to_spec(wav_name):
    wav, _ = torchaudio.load(wav_name)
    n_fft = 512
    spec = torchaudio.functional.spectrogram(
        waveform=wav, pad=0, window=torch.hann_window(n_fft), 
        n_fft=n_fft, hop_length=n_fft//2, win_length=n_fft, power=2, normalized=False)
    spec = torch.squeeze(spec)
    spec = spec[:, 1:].numpy() # remove first frame
    return spec


def wav_to_mel(wav_name):
    wav, _ = torchaudio.load(wav_name)
    mel = torchaudio.transforms.MelSpectrogram(n_fft=512, n_mels=80)(wav)
    mel = torch.squeeze(mel)
    mel = mel[:, 1:].numpy() # remove first frame
    return mel


def wav_to_mfcc(wav_name):
    wav, _ = torchaudio.load(wav_name)
    mfcc = torchaudio.transforms.MFCC(melkwargs={'n_fft':512})(wav)
    mfcc = torch.squeeze(mfcc)
    mfcc = mfcc[:, 1:].numpy() # remove first frame
    return mfcc


if __name__ == '__main__':
    '''
    temp = np.load('temp.npy')
    print(temp)
    win_size, step_size = 8, 4
    frames = frames_to_raw(win_size, step_size)('temp.npy')
    print(frames)
    frames = torch.Tensor(frames[None, None, :])
    frames = F.avg_pool1d(frames, win_size, step_size, ceil_mode=True)
    frames = torch.squeeze(frames).numpy()
    frames = np.greater(frames, 0.5).astype(np.float32)
    print(frames)
    '''
    import tqdm
    import os
    from multiprocessing import Pool
    
    os.chdir('../../rvads')
    fs = sorted(os.listdir())
    f2r = frames_to_raw(400, 160)

    def foo(fname):
        np.save(fname.replace('rvad', 'rvad-ext'), f2r(fname))

    with Pool() as p:
        p.map(foo, tqdm.tqdm(fs))

