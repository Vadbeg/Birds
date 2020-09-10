from typing import Tuple

from modules.data.dataset import BirdsDataset
from modules.data.utils import get_train_transforms, get_valid_transforms


def get_train_val_dataset(file_path: str,
                          n_classes: int,
                          width: int = 256,
                          size: Tuple[int, int] = (256, 256),
                          valid_percent: float = 0.3) -> Tuple[BirdsDataset, BirdsDataset]:
    assert 0 <= valid_percent <= 1, 'Start edge must be greater ' \
                                    'or equal 0 and lower or equal 1'

    train_end_edge = 1 - valid_percent

    train_transforms = get_train_transforms(size=size)
    valid_transforms = get_valid_transforms(size=size)

    birds_dataset_train = BirdsDataset(h5_file_path=file_path,
                                       width=width, start_edge=0.0,
                                       end_edge=train_end_edge,
                                       n_classes=n_classes,
                                       transforms=train_transforms)
    birds_dataset_valid = BirdsDataset(h5_file_path=file_path,
                                       width=width, start_edge=train_end_edge,
                                       end_edge=1.0,
                                       n_classes=n_classes,
                                       transforms=valid_transforms)

    return birds_dataset_train, birds_dataset_valid

