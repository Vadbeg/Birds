from typing import Tuple

from modules.data.dataset import BirdsDataset

from albumentations import Normalize, Compose


def get_train_val_dataset(file_path: str,
                          n_classes: int,
                          valid_percent: float = 0.3) -> Tuple[BirdsDataset, BirdsDataset]:
    assert 0 <= valid_percent <= 1, 'Start edge must be greater ' \
                                    'or equal 0 and lower or equal 1'

    train_end_edge = 1 - valid_percent

    birds_dataset_train = BirdsDataset(h5_file_path=file_path,
                                       width=256, start_edge=0.0,
                                       end_edge=train_end_edge,
                                       n_classes=n_classes)
    birds_dataset_valid = BirdsDataset(h5_file_path=file_path,
                                       width=256, start_edge=train_end_edge,
                                       end_edge=1.0,
                                       n_classes=n_classes)

    return birds_dataset_train, birds_dataset_valid


# def get_train_transforms():
#     train_transforms = Compose([
#         Normalize(std=)
#     ])

