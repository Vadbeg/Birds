"""
Module with base class for augmentations

Idea was taken from kaggle, but rebuilded for spectrogram purposes

:source: https://www.kaggle.com/tanulsingh077/audio-albumentations-transform-your-audio
"""

from albumentations.core.transforms_interface import BasicTransform


class AudioTransform(BasicTransform):
    """Base class for melspectrogram transforms"""

    @property
    def targets(self):
        return {'data': self.apply}

    def update_params(self, params, **kwargs):
        if hasattr(self, 'interpolation'):
            params['interpolation'] = self.interpolation

        if hasattr(self, 'fill_value'):
            params['fill_value'] = self.fill_value

        return params

