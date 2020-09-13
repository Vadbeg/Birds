"""Module with evaluAtion dataset for Birds sounds recognition"""

import os
import glob
from typing import Tuple, List, Union

import librosa
import numpy as np
from cv2 import cv2

import torch
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2


class TestDataset(Dataset):
    def __init__(self, folder: str,
                 size: Tuple[int, int] = (224, 224),
                 width: int = 512,
                 melspectrogram_parameters={}):
        self.folder = folder
        self.image_paths = self.__get_all_image_paths__()

        self.size = size
        self.width = width
        self.melspectrogram_parameters = melspectrogram_parameters

        self.min_value = -80

    def __len__(self):
        return len(self.image_paths)

    def __get_all_image_paths__(self) -> List[str]:
        all_image_paths = glob.glob(os.path.join(self.folder, '**', '*.mp3'), recursive=True)

        return all_image_paths

    def __normalize__(self, numpy_array: np.ndarray) -> np.ndarray:
        numpy_array = numpy_array + np.abs(self.min_value)
        numpy_array = numpy_array / np.abs(self.min_value)

        return numpy_array

    def __resize__(self, numpy_array: np.ndarray) -> np.ndarray:
        numpy_array = cv2.resize(numpy_array, dsize=self.size[::-1])

        return numpy_array

    @staticmethod
    def __get_label_from_path__(path: str):
        label = path.split(os.sep)[-2]

        return label

    @staticmethod
    def __to_tensor__(data: Union[int, float, np.ndarray, List]) -> torch.tensor:
        if isinstance(data, int) or len(data.shape) <= 2:
            tensor = torch.tensor(data=data)
        else:
            tensor = ToTensorV2()(image=data)['image']

        return tensor

    def __getitem__(self, idx: int):
        SR = 32000

        curr_image_path = self.image_paths[idx]
        curr_image_label = self.__get_label_from_path__(path=curr_image_path)

        y, sr = librosa.load(
            path=curr_image_path,
            sr=SR,
            mono=True,
            res_type='kaiser_fast'
        )

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        sound_array = librosa.power_to_db(mel_spectrogram, ref=np.max).astype(np.float32)

        curr_sound_width = sound_array.shape[1]

        if curr_sound_width > self.width:
            start_idx = np.random.randint(low=0, high=curr_sound_width - self.width)
            end_idx = start_idx + self.width

            sound_array = sound_array[:, start_idx:end_idx]

        elif curr_sound_width < self.width:
            n_times = self.width // curr_sound_width

            sound_array = np.repeat(sound_array, repeats=n_times, axis=1)
            sound_array = sound_array[:, :self.width]

        sound_array = self.__normalize__(numpy_array=sound_array)
        sound_array = self.__resize__(numpy_array=sound_array)
        sound_array_tensor = self.__to_tensor__(data=sound_array)

        sound_array_tensor = sound_array_tensor.unsqueeze(0).unsqueeze(0)

        return sound_array_tensor, curr_image_label
        # return sound_array, curr_image_label



