"""Module with model for Birdcall classification"""

import torch

from efficientnet_pytorch import EfficientNet


class ClassificationModel(torch.nn.Module):
    def __init__(self, n_classes: int, model_name: str = 'efficientnet-b0'):
        """
        Model for cirdcall classification

        :param n_classes: number of output nodes (number of classes)
        :param model_name: name of the model can be 'efficientnet-bX', where X is in [0, 1, 2, 3, 4, 5, 6, 7]
        """

        super().__init__()

        self.n_classes = n_classes
        self.model_name = model_name

        self.model = self.__build_model__()

    def __build_model__(self):
        """
        Build model

        :return: model
        """

        model = EfficientNet.from_pretrained(self.model_name,
                                             in_channels=1,
                                             num_classes=self.n_classes)

        return model

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Performs forward pass

        :param x: input tensor
        :return: model result
        """

        x = self.model(x)

        return x

