import random

import h5py
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import albumentations.augmentations.functional as F
from albumentations.core.transforms_interface import BasicTransform
# from albumentations.augmentations.transforms import ShiftScaleRotate

from modules.config import DataConfig
from modules.data.augmentations.base_augmentation import AudioTransform



class TimeShifting(AudioTransform):
    """Perform time shifting on images"""

    def __init__(self, delta_x: int, delta_y: int,
                 border_mode=cv2.BORDER_REFLECT,
                 always_apply: bool = False, p: float = 0.5):
        super(TimeShifting, self).__init__(always_apply)

        self.delta_x = delta_x
        self.delta_y = delta_y

        print(f'WTGS')

        self.border_mode = border_mode

    def apply(self, data, **params):
        """


        :param data:
        :param delta_x:
        :param delta_y:
        :param border_mode:
        :param params:
        :return:
        """

        warp_matrix = np.array([[1, 0, self.delta_x],
                                [0, 1, self.delta_y]], dtype=np.float32)

        data_augmentated = cv2.warpAffine(data, M=warp_matrix,
                                          dsize=data.shape[:2][::-1],
                                          borderMode=self.border_mode)
        print(f'WTF')

        return data_augmentated


class ShiftScaleRotate(BasicTransform):
    """Randomly apply affine transforms: translate, scale and rotate the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Default: (-0.1, 0.1).
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(ShiftScaleRotate, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        print(f'Sleep')

        return F.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode, self.value)


if __name__ == '__main__':
    data_config = DataConfig()

    print(data_config.dataset_path)

    delta_x = 15
    delta_y = 0
    warp_matrix = np.array([[1, 0, delta_x],
                            [0, 1, delta_y]], dtype=np.float32)
    # print(warp_matrix)

    shift_function = TimeShifting(p=1.0, delta_x=125, delta_y=120)
    shift_function = ShiftScaleRotate(p=1.0)
    h5_file = h5py.File(data_config.dataset_path, mode='r')

    while True:
        label = random.sample(h5_file.keys(), k=1)[0]

        data = h5_file[label][:, 0:256]

        data_augmentated = shift_function(image=data)['image']

        print(data.shape[:2])

        fig, axs = plt.subplots(1, 2)
        axs = axs.flatten()

        axs[0].imshow(data)
        axs[0].set_title(f'No augs')

        axs[1].imshow(data_augmentated)
        axs[1].set_title(f'With augs')

        plt.show()
