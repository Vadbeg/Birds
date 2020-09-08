from typing import Union, List, Dict

import numpy as np
import h5py

import torch
import torch.nn.functional as F
import torchsummary
from torch.utils.data import DataLoader

from torchvision.models import resnet18, resnext50_32x4d
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.data.utils import get_train_val_dataset
from modules.config import DataConfig


class ClassificationModel(LightningModule):
    def __init__(self, n_classes, batch_size: int, file_path: str, hparams: Dict):
        super().__init__()

        self.file_path = file_path
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.model = self.__build_model__()

        self.hparams = hparams

    def __get_accuracy__(self, y_pred, y):
        y_pred = torch.argmax(y_pred, dim=-1).squeeze()

        accuracy = torch.true_divide(torch.sum((y_pred == y)), len(y_pred))

        return accuracy

    def __build_model__(self):
        model = resnext50_32x4d(pretrained=True)

        model.conv1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=model.conv1.out_channels,
                                      kernel_size=model.conv1.kernel_size,
                                      stride=model.conv1.stride,
                                      padding=model.conv1.padding,
                                      bias=model.conv1.bias)

        sequential = torch.nn.Sequential(
            torch.nn.Linear(in_features=model.fc.in_features,
                            out_features=self.n_classes),
            # torch.nn.Sigmoid()
        )

        model.fc = sequential

        return model

    def forward(self, x: torch.tensor):
        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x.float())

        accuracy = self.__get_accuracy__(y_pred=y_pred, y=y)

        loss = F.cross_entropy(y_pred.float(), y)

        tensorboard_logs = {'loss': loss, 'acc': accuracy}

        res = {'loss': loss,
               'acc': accuracy,
               'log': tensorboard_logs}

        return res

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        tensorboard_logs = {'loss': avg_loss, 'acc': avg_loss}

        res = {'loss': avg_loss,
               'acc': avg_acc,
               'log': tensorboard_logs}

        return res

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x.float())

        accuracy = self.__get_accuracy__(y_pred=y_pred, y=y)

        loss = F.cross_entropy(y_pred.float(), y)

        tensorboard_logs = {'val_loss': loss, 'val_acc': accuracy}

        res = {'val_loss': loss,
               'val_acc': accuracy,
               'log': tensorboard_logs}

        return res

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

        res = {'val_loss': avg_loss,
               'val_acc': avg_acc,
               'log': tensorboard_logs}

        return res

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])


        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=0.1, patience=3,
                                                                    verbose=True),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def prepare_data(self) -> None:
        birds_dataset_train, birds_dataset_valid = get_train_val_dataset(
            file_path=self.file_path,
            valid_percent=0.3,
            n_classes=self.n_classes
        )

        self.train_dataset = birds_dataset_train
        self.valid_dataset = birds_dataset_valid

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=12,
                                      shuffle=True)

        return train_dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        val_dataloader = DataLoader(self.valid_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=12,
                                    shuffle=True)

        return val_dataloader


if __name__ == '__main__':
    data_config = DataConfig()

    hparams = {
        'learning_rate': 0.001
    }

    model = ClassificationModel(n_classes=2, file_path=data_config.dataset_path, batch_size=4, hparams=hparams)

    torchsummary.summary(model.model.cuda(), input_size=(1, 256, 256))


