import os
from pathlib import Path

import h5py
import librosa

import numpy as np
import pandas as pd
from tqdm import tqdm


def resample(ebird_code: str, filename: str, target_sr: int,
             audio_dir: str):

    try:
        y, sr = librosa.load(
            path=os.path.join(audio_dir, ebird_code, filename),
            sr=target_sr,
            mono=True,
            res_type='kaiser_fast'
        )

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        s_db = librosa.power_to_db(mel_spectrogram, ref=np.max).astype(np.float32)

        print(len(mel_spectrogram))
        print(y.shape)
        print(s_db.shape)

    except Exception as exception:
        with open(f'bad_files.txt', mode='a') as file:
            file.write(os.path.join(audio_dir, ebird_code, filename) + '\n')


if __name__ == '__main__':
    TRAIN_AUDIO_DIR = Path('/home/vadbeg/Data/birdsong/birdsong-recognition/train_audio')
    TRAIN_RESAMPLED_AUDIO_DIR = Path(
        '/home/vadbeg/Data/birdsong/birdsong-recognition/train_resampled_audio'
    )

    TARGET_SR = 32000
    NUM_THREAD = 12

    train = pd.read_csv('/home/vadbeg/Data/birdsong/birdsong-recognition/train.csv')

    train_audio_info = train[['ebird_code', 'filename']].values.tolist()
    dataset_length = len(train_audio_info)

    for idx, (ebird_code, file_name) in enumerate(tqdm(train_audio_info)):

        resample(ebird_code=ebird_code,
                 filename=file_name,
                 target_sr=TARGET_SR,
                 audio_dir=TRAIN_AUDIO_DIR)

