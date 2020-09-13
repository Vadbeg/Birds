"""Module with configs"""

from pathlib import Path


class DataConfig:
    """Config for data loading"""

    dataset_path = Path('/home/vadbeg/Data/birdsong/birds.hdf5')

    n_classes = 264


class ModelConfig:
    """Config for model saving"""

    weights_folder = Path('weights')
