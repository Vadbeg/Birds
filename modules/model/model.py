import torch

from torchvision.models import resnet18, resnext50_32x4d


class ClassificationModel(torch.nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.n_classes = n_classes

        self.model = self.__build_model__()

    def __build_model__(self):
        model = resnet18(pretrained=True)

        model.conv1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=model.conv1.out_channels,
                                      kernel_size=model.conv1.kernel_size,
                                      stride=model.conv1.stride,
                                      padding=model.conv1.padding,
                                      bias=model.conv1.bias)

        sequential = torch.nn.Sequential(
            torch.nn.Linear(in_features=model.fc.in_features,
                            out_features=self.n_classes),
            torch.nn.Softmax()
        )

        model.fc = sequential

        return model

    def forward(self, x: torch.tensor):
        x = self.model(x)

        return x

