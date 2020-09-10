"""Module with help functions for dataset"""

from typing import Tuple

import numpy as np
from albumentations import Normalize, Compose, Resize


def get_imagenet_normalizer():
    norm_function = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    def normalizer(image: np.ndarray) -> np.ndarray:
        image_norm = norm_function(image=image, mask=None)['image']

        return image_norm

    return normalizer


def get_train_transforms(size: Tuple[int, int]):
    train_transforms = Compose([
        Resize(height=size[0], width=size[1])
    ])

    return train_transforms


def get_valid_transforms(size: Tuple[int, int]):
    valid_transforms = Compose([
        Resize(height=size[0], width=size[1])
    ])

    return valid_transforms
