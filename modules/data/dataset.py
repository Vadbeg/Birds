"""Module with dataset for Birds sounds recognition"""

from typing import Union, List, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cv2 import cv2

import torch
import albumentations as albu
from torch.utils.data.dataset import Dataset
from albumentations.pytorch.transforms import ToTensorV2

from modules.config import DataConfig
from modules.data.utils import get_imagenet_normalizer, get_train_transforms, get_valid_transforms


class BirdsDataset(Dataset):
    def __init__(self, h5_file_path: str,
                 width: int, n_classes: int,
                 start_edge: float = 0,
                 end_edge: float = 1,
                 size: Tuple[int, int] = (224, 224),
                 transforms: albu.Compose = None):
        """
        Dataset for bird sounds classification.

        :param h5_file_path: path to hdf5 file
        :param width: width of the sound we wan't to read
        :param n_classes: number of classes in dataset
        :param start_edge: starting edge, used when we need to check part of the dataset
        :param end_edge: ending edge, used when we need to check part of the dataset
        :param size: size of image (spectrogram) we want to send to CNN
        :param transforms: transforms for audio
        """

        self.h5_file = None
        self.h5_file_path = h5_file_path

        self.labels = None
        self.n_classes = n_classes

        self.start_edge = start_edge
        self.end_edge = end_edge

        self.min_value = -80
        self.size = size

        self.transforms = transforms

        assert 0 <= self.start_edge <= 1, 'Start edge must be greater or ' \
                                          'equal 0 and lower or equal 1'
        assert self.start_edge <= self.end_edge <= 1, 'End edge must be greater or equal start_edge ' \
                                                      'and lower or equal 1'

        self.width = width

        self.normalizer = get_imagenet_normalizer()

    @staticmethod
    def __get_label_idx__(idx: int) -> int:
        """
        Finds index of the label

        :param idx: idx of sector
        :return: index of label
        """

        label_idx = idx // 100
        label_idx = int(label_idx) if label_idx >= 0 else 0

        return label_idx

    def __get_bounds__(self, total_width, sound_lower_idx) -> Tuple[int, int]:
        """
        Finds bound of audiotrack sample

        :param total_width: total width of audio track for given specie
        :param sound_lower_idx: lower index of sector we wan't to cutout
        :return: lower_bound, upper_bound
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
        """
        Transform numpy arrays, integers, etc. to pytoch data

        :param data: numpy arrays, integers, etc. to transform
        :return: tensor
        """

        if isinstance(data, int) or len(data.shape) <= 2:
            tensor = torch.tensor(data=data)
        else:
            tensor = ToTensorV2()(image=data)['image']

        return tensor

    def __normalize__(self, numpy_array: np.ndarray) -> np.ndarray:
        """
        Normalizes numpy array (MinMax)

        :param numpy_array: array to normalize
        :return: resulted array
        """

        numpy_array = numpy_array + np.abs(self.min_value)
        numpy_array = numpy_array / np.abs(self.min_value)

        return numpy_array

    def __resize__(self, numpy_array: np.ndarray) -> np.ndarray:
        """
        Resizes numpy array

        :param numpy_array: array to resize
        :return: resized array
        """

        numpy_array = cv2.resize(numpy_array, dsize=self.size[::-1])

        return numpy_array

    def __get_ohe_label__(self, label_idx) -> List[int]:
        """
        Transforms index of label to OHE label

        :param label_idx: index of label
        :return: OHE label
        """

        label = [0] * self.n_classes
        label[label_idx] = 1

        return label

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Reads item by index from hdf5 file

        :param idx: index of item we want to read
        :return: melspectrogram as tensor, index of label as tensor
        """

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

        if self.transforms:
            sound_array = self.transforms(image=sound_array)['image']

        sound_array = self.__normalize__(numpy_array=sound_array)
        sound_array = self.__resize__(numpy_array=sound_array)

        tensor_sound_array = self.__to_tensor__(data=sound_array)
        tensor_sound_array = tensor_sound_array.unsqueeze(0)

        tensor_label = self.__to_tensor__(data=label_idx)

        return tensor_sound_array, tensor_label

        # print(f'Sound array mean: {sound_array.mean()}')
        # print(f'Sound array max: {sound_array.max()}')
        # print(f'Sound array min: {sound_array.min()}')
        # print(f'-' * 15)
        #
        # return sound_array, label

    def __len__(self) -> int:
        """
        Returns length of dataset

        :return: length of dataset
        """

        length = self.n_classes * 100

        return length


if __name__ == '__main__':
    data_config = DataConfig()

    file = h5py.File(data_config.dataset_path, mode='r')

    transforms = get_train_transforms()

    birds_dataset = BirdsDataset(h5_file_path=data_config.dataset_path,
                                 width=2048, n_classes=data_config.n_classes,
                                 start_edge=0,
                                 end_edge=1,
                                 transforms=transforms, size=(128, 512))

    temp = birds_dataset[299]
    #
    print(birds_dataset.labels)
    exit()

    # print(temp[0].size())
    # print(temp[1])
    # exit()
    # print(temp[0])
    # print(temp[1])
    # exit()

    while True:
        IDX = np.random.randint(0, (len(birds_dataset) / 264) * 3)
        temp = birds_dataset[IDX]

        plt.imshow(temp[0])
        plt.title(temp[1])
        plt.show()
    # exit()

    min_value = float('inf')
    max_value = -float('inf')
    for idx, el in enumerate(tqdm(birds_dataset)):
        if min_value > torch.min(el[0]):
            min_value = torch.min(el[0])

        if max_value < torch.max(el[0]):
            max_value = torch.max(el[0])

    print(f'Min value: {min_value}')
    print(f'Max value: {max_value}')




