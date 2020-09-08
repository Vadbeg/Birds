from pathlib import Path


class DataConfig:
    dataset_path = Path('/home/vadbeg/Data/birdsong/birds.hdf5')

    n_classes = 264


class ModelConfig:
    weights_folder = Path('weights')
