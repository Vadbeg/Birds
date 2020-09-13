import os
import glob
import time
from typing import Tuple, List, Union

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2

from modules.model.model import ClassificationModel


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
            sr=None,
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


if __name__ == '__main__':
    weights_path = '/home/vadbeg/Projects/Kaggle/' \
                   'Birds/efficientnet-b0efficientnet-b0_ckpt_epoch_10.ckpt'
    # PATH = '/home/vadbeg/Projects/Kaggle/Birds/model.pt'

    model_config = {
        "model_name": "efficientnet-b0",
        "n_classes": 264
    }

    model = ClassificationModel(**model_config)

    lightning_model_checkpoint = torch.load(weights_path)
    model_state_dict = lightning_model_checkpoint['state_dict']

    model.load_state_dict(model_state_dict)

    # torch.save(model.state_dict(), PATH)
    # exit()

    DATA_FOLDER = '/home/vadbeg/Data/birdsong/birdsong-recognition/train_audio'
    test_dataset = TestDataset(folder=DATA_FOLDER, width=2048, size=(128, 512))

    labels_list = sorted(os.listdir(DATA_FOLDER))
    print(f'Labels list: {labels_list}')

    top = 0

    for _ in tqdm(range(100)):
        IDX = np.random.randint(low=0, high=len(test_dataset))

        item = test_dataset[IDX]

        # plt.imshow(item[0])
        # plt.title(item[1])
        # plt.show()
        # continue

        result = model(item[0])
        result = result.detach().cpu().numpy().flatten()
        result = np.argmax(result)

        # print(result)
        # print(labels_list[int(result)])
        # print(item[1])
        # print(f'-' * 15)

        if item[1] == labels_list[int(result)]:
            top += 1


        # time.sleep(1)

    print(f'Accuracy: {top / 100}')

