import random
from typing import Optional, Tuple

import numpy as np
from cv2 import cv2

from modules.data.augmentations.base_augmentation import AudioTransform


class TimeShifting(AudioTransform):
    """Perform time shifting on spectrogtrams"""

    def __init__(self, max_shift_value: float,
                 border_mode=cv2.BORDER_WRAP,
                 always_apply: bool = False, p: float = 0.5):
        """
        Spectrogram can be shifted forward and backward.
        Shift value needs to be in range [0, 1]. It will be randomly choosed in range [0, max_shift_value]

        :param max_shift_value: maximal shifting value in percents
        :param border_mode: cv2 border mode
        :param always_apply: if True, than augmentation is applied always
        :param p: probability of applying
        """

        super().__init__(always_apply, p)

        self.max_shift_value = max_shift_value

        self.border_mode = border_mode

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Applies augmentation on real data

        :param image: spectrogram
        :param params: additional params for use of albumentations
        :return: shifted spectrogram
        """

        image_width = image.shape[1]

        max_shift_value_real = int(image_width * self.max_shift_value)
        shift_value_real = np.random.randint(0, max_shift_value_real)

        warp_matrix = np.array([[1, 0, shift_value_real],
                                [0, 1, 0]], dtype=np.float32)

        data_augmentated = cv2.warpAffine(image, M=warp_matrix,
                                          dsize=image.shape[:2][::-1],
                                          borderMode=self.border_mode)

        return data_augmentated

    def get_transform_init_args_names(self):
        return ['shift_value', 'border_mode']


class SpeedTuning(AudioTransform):
    """Perform time shifting on spectrogtrams"""

    def __init__(self, speed_rate_range: Optional[Tuple[float, float]] = None,
                 always_apply: bool = False, p: float = 0.5):
        """
        Spectrogram can be resized on speed speed_rate (stretched).
        For best result Shift value needs to be in range [0.5, 1.5].
        It will be randomly choosed in range [0.6, 1.3] if not provided

        :param speed_rate: spectrogram will be stretched on factor taken from this range
        :param always_apply: if True, than augmentation is applied always
        :param p: probability of applying
        """

        super().__init__(always_apply, p)

        if speed_rate_range:
            self.speed_rate_range = speed_rate_range
        else:
            self.speed_rate_range = (0.6, 1.3)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Applies augmentation on real data

        :param data: spectrogram
        :param params: additional params for use of albumentations
        :return: shifted spectrogram
        """

        image_width = image.shape[1]

        speed_rate = np.random.uniform(*self.speed_rate_range)
        audio_speed_tune = cv2.resize(image, (int(image_width * speed_rate), image.shape[0]))

        audio_speed_tune_width = audio_speed_tune.shape[1]

        if audio_speed_tune_width < image_width:
            pad_length = image_width - audio_speed_tune_width
            audio_speed_tune = np.r_[np.random.uniform(low=-80,
                                                       high=-79,
                                                       size=(int(pad_length / 2), image.shape[0])),
                                     audio_speed_tune.transpose(1, 0),
                                     np.random.uniform(low=-80,
                                                       high=-79,
                                                       size=(int(pad_length / 2), image.shape[0]))]

            audio_speed_tune = audio_speed_tune.transpose(1, 0)
        elif audio_speed_tune_width > image_width:
            cut_len = audio_speed_tune_width - image_width

            start_idx = np.random.randint(0, cut_len)

            audio_speed_tune = audio_speed_tune[:, start_idx: start_idx + image_width]

        return audio_speed_tune

    def get_transform_init_args_names(self):
        return ['shift_value', 'border_mode']


