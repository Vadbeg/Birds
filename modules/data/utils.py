"""Module with utils for creating dataset"""

from typing import Tuple, Callable

import numpy as np
from albumentations import Normalize, Compose, Resize

from modules.data.augmentations.audio_augmentations import TimeShifting, SpeedTuning


def get_imagenet_normalizer() -> Callable:
    """
    Creates imagenet normalizing function

    :return: normalizing function
    """

    norm_function = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    def normalizer(image: np.ndarray) -> np.ndarray:
        image_norm = norm_function(image=image, mask=None)['image']

        return image_norm

    return normalizer


def get_train_transforms() -> Callable:
    """
    Creates train transforms

    :return: train transforms
    """

    train_transforms = Compose([
        TimeShifting(max_shift_value=0.5, p=0.5),
        SpeedTuning(p=0.5)
    ])

    return train_transforms


def get_valid_transforms(size: Tuple[int, int]) -> Callable:
    """
    Creates validation transforms

    :param size: size of image we want to input in CNN
    :return: validation transforms
    """

    valid_transforms = Compose([
        Resize(height=size[0], width=size[1])
    ])

    return valid_transforms


def get_resize_transforms(size: Tuple[int, int]):
    """
    Return resize function

    :param size: size of image we want to input in CNN
    :return: resize function
    """

    resize_transforms = Compose([
        Resize(height=size[0], width=size[1])
    ])

    return resize_transforms
