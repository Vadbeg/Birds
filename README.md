Librosa is pretty slow, so it is better to transform data using it, and save spectrograms in another data format. For example [HDF5](https://www.h5py.org):   
> It lets you store huge amounts of numerical data, and easily manipulate that data from NumPy. For example, you can slice into multi-terabyte datasets stored on disk, as if they were real NumPy arrays. Thousands of datasets can be stored in a single file, categorized and tagged however you want.

Script for data transforming:
```
import os
from pathlib import Path

import h5py
import librosa

import numpy as np
import pandas as pd
from tqdm import tqdm


def resample(ebird_code: str, filename: str, target_sr: int,
             audio_dir: str,  dataset) -> None:
    """
    Reads mp3 file and save it into HDF5 binary data format

    :source: https://www.h5py.org
    :param ebird_code: birds name code
    :param filename: name of the file with birdsong mp3
    :param target_sr: sampling rate we wan't to have
    :param audio_dir: directory of all mp3 birdsong file
    :param dataset: HDF5 dataset
    """

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
    TRAIN_AUDIO_DIR = Path('/**/birdsong-recognition/train_audio')
    TRAIN_RESAMPLED_AUDIO_DIR = Path('/**/birdsong-recognition/train_resampled_audio')

    TARGET_SR = 32000
    NUM_THREAD = 12

    train = pd.read_csv('/**/birdsong-recognition/train.csv')

    train_audio_info = train[['ebird_code', 'filename']].values.tolist()
    dataset_length = len(train_audio_info)

    with h5py.File('/**/birdsong/bird_spectrogram/testfile.hdf5', mode='w') as file:
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

```

What does this script do?


```
Basically, it creates some kind of dictionary, where the key is the bird code (ex: aldfly), 
and the value is concatenated spectrogram of every birdsound of given class. It creates the 
problem: we can't distinct spectrograms of the same class from each other. It can be easely 
solved by saving every birdsong starting and ending index.
```

What is the size of resulted dataset?

`38,7 GB`

How long does it take to create it?

`~3 hours (AMD Ryzen 2600X)`

How to load it?

```
file = h5py.File(data_config.dataset_path, mode='r')
sound_array = file[label][:, 0:256]
```

If you find a bug, please, let me know in the comments.

P.S. Huge thank you to `Vladimir Sydorskyi` for the great idea.