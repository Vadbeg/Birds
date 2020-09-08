"""Module with dataset for Birds sounds recognition"""

from typing import Union, List

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset

from modules.config import DataConfig


class BirdsDataset(Dataset):
    def __init__(self, h5_file_path: str,
                 width: int, n_classes: int,
                 start_edge: float = 0,
                 end_edge: float = 1):
        """
        Dataset for bird sounds classification.

        :param h5_file: h5py file
        """

        self.h5_file = None
        self.h5_file_path = h5_file_path

        self.labels = None
        self.n_classes = n_classes

        self.start_edge = start_edge
        self.end_edge = end_edge

        self.min_value = -80

        assert 0 <= self.start_edge <= 1, 'Start edge must be greater or ' \
                                          'equal 0 and lower or equal 1'
        assert self.start_edge <= self.end_edge <= 1, 'End edge must be greater or equal start_edge ' \
                                                      'and lower or equal 1'

        self.width = width

    @staticmethod
    def __get_label_idx__(idx: int) -> int:
        label_idx = idx // 100
        label_idx = int(label_idx) if label_idx >= 0 else 0

        return label_idx

    def __get_bounds__(self, total_width, sound_lower_idx):
        """
        Finds bound of audiotrack sample

        :param total_width:
        :param sound_lower_idx:
        :return:
        """

        lower_bound = int((total_width * sound_lower_idx) / 100)
        upper_bound = int((total_width * (sound_lower_idx + 1)) / 100)

        width_of_one_sector = upper_bound - lower_bound

        if width_of_one_sector > self.width:
            lower_bound = np.random.randint(lower_bound, upper_bound - self.width)
            upper_bound = lower_bound + self.width

        elif width_of_one_sector < self.width:
            if upper_bound - self.width < 0:
                lower_bound = 0
                upper_bound = lower_bound + self.width

                if upper_bound > total_width:
                    raise ValueError(f'Bad width value: {self.width}. It is bigger then whole width of soundtrack.')

            else:
                lower_bound = upper_bound - self.width
                upper_bound = upper_bound

        return lower_bound, upper_bound

    @staticmethod
    def __to_tensor__(data: Union[int, float, np.ndarray, List]) -> torch.tensor:
        tensor = torch.tensor(data=data)

        return tensor

    def __normalize__(self, numpy_array: np.ndarray) -> np.ndarray:
        numpy_array = numpy_array + np.abs(self.min_value)
        numpy_array = numpy_array / np.abs(self.min_value)

        return numpy_array

    def __get_ohe_label__(self, label_idx):
        label = [0] * self.n_classes
        label[label_idx] = 1

        return label

    def __getitem__(self, idx: int):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, mode='r')
            self.labels = sorted(list(self.h5_file.keys()))

        label_idx = self.__get_label_idx__(idx=idx)
        sound_lower_idx = idx % 100

        label = self.labels[label_idx]
        sound_array_shape = self.h5_file[label].shape

        start_bound = int(self.start_edge * sound_array_shape[1])
        end_bound = int(self.end_edge * sound_array_shape[1])

        total_width = int(end_bound - start_bound)

        lower_bound, upper_bound = self.__get_bounds__(total_width=total_width, sound_lower_idx=sound_lower_idx)

        sound_array = self.h5_file[label][:, start_bound + lower_bound: start_bound + upper_bound]
        sound_array = self.__normalize__(numpy_array=sound_array)

        tensor_sound_array = self.__to_tensor__(data=sound_array)
        tensor_sound_array = tensor_sound_array.unsqueeze(0)

        # label = self.__get_ohe_label__(label_idx=label_idx)
        tensor_label = self.__to_tensor__(data=label_idx)

        return tensor_sound_array, tensor_label
        # return sound_array

    def __len__(self):
        length = self.n_classes * 100

        return length


if __name__ == '__main__':
    data_config = DataConfig()

    file = h5py.File(data_config.dataset_path, mode='r')
    birds_dataset = BirdsDataset(h5_file_path=data_config.dataset_path,
                                 width=256, n_classes=data_config.n_classes,
                                 start_edge=0,
                                 end_edge=1)

    temp = birds_dataset[299]
    print(temp[0])
    print(temp[1])
    exit()


    while True:
        IDX = np.random.randint(0, len(birds_dataset))
        temp = birds_dataset[IDX]

        plt.imshow(temp)
        plt.show()

    min_value = float('inf')
    max_value = -float('inf')
    for idx, el in enumerate(tqdm(birds_dataset)):
        if min_value > torch.min(el[0]):
            min_value = torch.min(el[0])

        if max_value < torch.max(el[0]):
            max_value = torch.max(el[0])

    print(f'Min value: {min_value}')
    print(f'Max value: {max_value}')




