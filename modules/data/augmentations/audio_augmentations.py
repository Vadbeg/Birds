import random

import numpy as np

from modules.data.augmentations.base_augmentation import AudioTransform


class TimeShifting(AudioTransform):
    """Perform time shifting on images"""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__()

    def apply(self, data, **params):
        pass



