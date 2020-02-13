import numpy as np
import os
import pickle
import torch
import torchaudio
from tqdm import tqdm


def from_wav_to_dataset(path, name=None, pickled=False):
    files = sorted(os.listdir(path))
    dataset = []
    max_len = 0
    name = name if name is not None else 'dataset'

    stft = torchaudio.transforms.Spectrogram(512, power=None)

    for f in tqdm(files):
        if not f.endswith('.wav'):
            continue
        data, sample_rate = torchaudio.load(os.path.join(path, f))
        data = torchaudio.compliance.kaldi.resample_waveform(data,
                                                             sample_rate,
                                                             16000)
        data = stft(data)
        data = torch.cat(torchaudio.functional.magphase(data))
        data = data.numpy().transpose(1, 2, 0) # freq, time, chan
        dataset.append(data)

        if data.shape[1] > max_len:
            max_len = data.shape[1]

    if pickled:
        pickle.dump(dataset, open(name+'.pickle', 'wb'))
    else:
        def pad(x, max_len):
            return np.pad(x, ((0, 0), (0, max_len - x.shape[1]), (0, 0)), 'constant')
        dataset = np.stack(tuple(map(lambda x: pad(x, max_len), dataset)))
        np.save(name+'.npy', dataset)


def wavs2mel(path, amp_to_DB=False, name=None, pickled=True):
    files = sorted(os.listdir(path))
    dataset = []
    max_len = 0
    name = name if name is not None else 'wavs'

    mel = torchaudio.transforms.MelSpectrogram(n_fft=512, n_mels=80)
    if amp_to_DB:
        amp_to_DB = torchaudio.transforms.AmplitudeToDB('power', top_db=80.)

    for f in tqdm(files):
        if not f.endswith('.wav'):
            continue
        data, sample_rate = torchaudio.load(os.path.join(path, f))
        data = torchaudio.compliance.kaldi.resample_waveform(data,
                                                             sample_rate,
                                                             16000)
        data = mel(data)
        if amp_to_DB:
            data = amp_to_DB(data)
        data = data.numpy().transpose(1, 2, 0) # freq, time, chan
        dataset.append(data)

        if data.shape[1] > max_len:
            max_len = data.shape[1]

    if pickled:
        pickle.dump(dataset, open(name+'.pickle', 'wb'))
    else:
        def pad(x, max_len):
            return np.pad(x, ((0, 0), (0, max_len - x.shape[1]), (0, 0)), 'constant')
        dataset = np.stack(tuple(map(lambda x: pad(x, max_len), dataset)))
        np.save(name+'.npy', dataset)


def from_clean_to_labels(path, name=None):
    files = sorted(os.listdir(path))
    labels = []
    name = name if name is not None else 'label'

    for f in tqdm(files):
        if not f.endswith('.wav'):
            continue
        data, sample_rate = torchaudio.load(os.path.join(path, f))
        data = torchaudio.compliance.kaldi.resample_waveform(data,
                                                             sample_rate,
                                                             16000)
        data = data.mean(axis=0, keepdims=True)
        data = torch.nn.functional.max_pool1d(data[None, :],
                                              512, 256,
                                              ceil_mode=True)
        data = np.abs(data.numpy()[0, 0])
        label = data > data.max() * 0.1
        label *= data > 0.
        labels.append(label.astype(np.int16))

    pickle.dump(labels, open(name+'.pickle', 'wb'))


def left_right(label):
    ''' 
    이건 from_clean_to_labels 함수를 거치면 음성 없는 부분은
    없게 나오는데, 음성 시작부터 끝까지 음성 있는걸로 매워주는 역할이다.
    '''
    arr = ''.join(map(str, label))
    label[arr.find('1'):arr.rfind('1')+1] = 1
    return label


def spec2melspec(spec):
    '''
    assume spec has a shape of (freq, time, chan * 2)
    and spec[:, :, :chan] is mag spec and spec[:, :, chan:] is phase spec
    '''
    freq, time, chan2 = spec.shape
    spec = np.power(spec[:, :, :chan2//2], 2)
    spec = spec.transpose(2, 0, 1)
    spec = torchaudio.transforms.MelScale(n_mels=80)(torch.Tensor(spec))
    spec = spec.numpy().transpose(1, 2, 0)
    return spec


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = np.load('/media/data1/datasets/ai_challenge/icassp/final_x.npy')
    print('loaded')
    spec = data[0]
    # plt.plot(spec[:, :, 0])
    # plt.show()
    melspec = spec2melspec(spec)
    plt.plot(melspec[:, :, 0])
    plt.show()
    
