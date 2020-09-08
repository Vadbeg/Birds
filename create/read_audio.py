import os
from pathlib import Path

import h5py
import librosa

import numpy as np
import pandas as pd
from tqdm import tqdm


def resample(ebird_code: str, filename: str, target_sr: int,
             audio_dir: str,  dataset):

    try:
        y, sr = librosa.load(
            path=os.path.join(audio_dir, ebird_code, filename),
            sr=target_sr,
            mono=True,
            res_type='kaiser_fast'
        )

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        s_db = librosa.power_to_db(mel_spectrogram, ref=np.max).astype(np.float32)

        curr_shape = s_db.shape
        dataset_shape = dataset.shape

        new_shape = (dataset_shape[0], dataset_shape[1] + curr_shape[1])

        dataset.resize(size=new_shape)

        dataset[:, -curr_shape[1]:] = s_db

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

    with h5py.File('/home/vadbeg/Data/birdsong/bird_spectrogram/testfile2.hdf5', mode='w') as file:
        codes = list()

        for idx, (ebird_code, file_name) in enumerate(tqdm(train_audio_info)):
            if ebird_code not in file:

                codes.append(ebird_code)
                dataset = file.create_dataset(ebird_code,
                                              shape=(128, 0),
                                              maxshape=(None, None),
                                              chunks=True,
                                              dtype=np.float32)
            else:
                dataset = file[ebird_code]

            resample(ebird_code=ebird_code,
                     filename=file_name,
                     target_sr=TARGET_SR,
                     audio_dir=TRAIN_AUDIO_DIR,
                     dataset=dataset)

        print(len(codes))

    # with h5py.File('/home_hard/Data/Birds/testfile.hdf5', mode='r') as file:
    #     for name, data in file.items():
    #         print(name)
    #         print(data.shape)
    #         print(sum(sum(data[:, 0: 10_000])))

