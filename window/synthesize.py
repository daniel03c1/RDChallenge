import os


FANT_DIR = '/codes/fant/filter_add_noise'


def raw2wav(raw, wav=None, sr=16000, bit=16, channel=1):
    assert raw.endswith('.raw')
    if wav is None:
        wav = raw.replace('.raw', '.wav')
    os.system(f'sox -r {sr} -e signed-integer -c {channel} '
              f'-b {bit} {raw} {wav}')


def wav2raw(wav, raw=None, sr=16000, bit=16, channel=1):
    assert wav.endswith('.wav')
    if raw is None:
        raw = wav.replace('.wav', '.raw')
    os.system(f'sox -r {sr} -e signed-integer -c {channel} '
              f'-b {bit} --endian little {wav} {raw}')


def generate_fant(in_list, out_list, noise, 
                  snr, filter_mode='p341', norm=True):
    norm = '-l -20 ' if norm else ''
    os.system(f'{FANT_DIR} -u {norm}-i {in_list} -o {out_list} '
              f'-n {noise} -f {filter_mode} -s {snr}')


def extract_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def synthesize(clean_raws, noise, snr, norm=True):
    _noise = extract_name(noise)
    in_list = f'{_noise}_{snr}_in.list'
    out_list = f'{_noise}_{snr}_out.list'

    with open(in_list, 'wb') as i:
        i.write('\n'.join(clean_raws).encode())

    with open(out_list, 'wb') as o:
        o.write('\n'.join(
            map(lambda x: f'{extract_name(x)}_{_noise}.raw', clean_raws)).encode())
    
    generate_fant(in_list, out_list, noise, snr, 'p341', norm)

    os.remove(in_list)
    os.remove(out_list)


def synth2(clean_noise_snr):
    return synthesize(*clean_noise_snr)


if __name__ == "__main__":
    '''
    import numpy as np
    from glob import glob
    from multiprocessing import Pool
    from tqdm import tqdm

    SNRS = [-10, 0, 10] # -15, -5, 5, 15]
    SPEEDS = ['10'] # '09', '10', '11']
    NOISE_AUG = [True] # True, False]

    # NOISES = glob(os.path.join('/codes/noisex/train', '*.raw'))
    NOISES = glob(os.path.join('/codes/aurora2', '*.raw'))
    AUDIO_PATH = '/datasets/ai_challenge/LibriSpeech'

    os.chdir('/codes')

    for noise_aug in NOISE_AUG:
        # Select Noises
        if noise_aug:
            noises = NOISES
            tail = ''
        if not noise_aug:
            noises = [noise for noise in NOISES 
                      if noise.endswith('0.raw')]
            tail = '_no_noise_aug'
        n_noises = len(noises)

        for speed in SPEEDS:
            # Select Audios
            CLEAN_PATH = os.path.join(
                AUDIO_PATH,
                # f'train_raw_{speed}')
                'train-clean-100-raw')
            clean = glob(os.path.join(CLEAN_PATH, '*.raw'))
            clean = np.array(clean)

            for snr in SNRS:
                # Synthesize
                alloc = np.random.randint(n_noises, size=len(clean))
                clean_alloc = [clean[np.where(alloc == i)[0]]
                               for i in range(n_noises)]

                with Pool(10) as p:
                    p.map(synth2, 
                          list(zip(clean_alloc, noises, np.repeat(snr, n_noises))))
                    p.map(raw2wav, tqdm(glob('*.raw')))

                folder = f'snr{snr}_{speed}{tail}'
                os.mkdir(folder)
                os.system(f'mv *.wav {folder}')
                os.system('rm *.raw')
    '''
    '''
    os.system(f'sox -r 48k -e signed-integer -c 2 '
              f'-b 16 --endian little audio_000001.wav test.raw rate 16k')
    raw2wav('test.raw', channel=2)
    '''

    os.chdir('/datasets/ai_challenge/interspeech20/train/_output_wavs_clean')
    '''
    for i in range(100):
        os.system(f'sox -r 48k -e signed-integer -c 2 -b 16 --endian little '
                  f'noise_{i+4901:06d}.wav noise_{i+4901:06d}.raw rate 16k')
    
    '''
    print()
    for i in range(100):
        synthesize([f'clean_{i+4901:06d}.raw'], 
                   f'../_output_wavs_noise/noise_{i+4901:06d}.raw', 
                   0, norm=False)
        print(f'{i+1}\r', end='')
    print()

