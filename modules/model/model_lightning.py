"""Module with pytorch lightning model"""

import warnings
from typing import Union, List, Tuple, Dict

import torch
import torch.nn.functional as F
import torchsummary
from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.classification import f1_score
from efficientnet_pytorch import EfficientNet

from modules.data.help_functions import get_train_val_dataset
from modules.config import DataConfig

warnings.filterwarnings('ignore')


class ClassificationModel(LightningModule):
    def __init__(self, n_classes, batch_size: int,
                 file_path: str, hparams: Dict,
                 model_name: str, width: int, size: Tuple[int, int]):
        """
        Pytorch lightning model for birdcall classification

        :param n_classes: number of output nodes (number of classes)
        :param batch_size: batch size
        :param file_path: path to the hdf5 dataset
        :param hparams: additional params for CNN
        :param model_name: name of the model can be 'efficientnet-bX', where X is in [0, 1, 2, 3, 4, 5, 6, 7]
        :param width: width of sector we want to cutout from audiotrack
        :param size: size of the image we want to input to CNN
        """

        super().__init__()

        self.file_path = file_path
        self.n_classes = n_classes
        self.batch_size = batch_size

        self.model_name = model_name

        self.model = self.__build_model__()

        self.width = width
        self.size = size

        self.hparams = hparams

    @staticmethod
    def __get_accuracy__(y_pred: torch.tensor, y: torch.tensor) -> torch.tensor:
        """
        Calculates accuracy on batch

        :param y_pred: prediction tensor
        :param y: true tensor
        :return: accuracy
        """

        y_pred = torch.argmax(y_pred, dim=-1).squeeze()

        accuracy = torch.true_divide(torch.sum((y_pred == y)), len(y_pred))

        return accuracy

    def __get_f1_score__(self, y_pred: torch.tensor, y: torch.tensor) -> torch.tensor:
        """
        Calculates f1 score on batch

        :param y_pred: prediction tensor
        :param y: true tensor
        :return: f1 score
        """

        y_pred = torch.argmax(y_pred, dim=-1).squeeze()

        f1_score_num = f1_score(pred=y_pred, target=y, reduction='sum', num_classes=self.n_classes) / len(y)

        return f1_score_num

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

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> Dict:
        """
        Performs training step

        :param batch: one batch
        :param batch_idx: index of this batch
        :return: dictionary with stats
        """

        x, y = batch

        y_pred = self(x.float())

        accuracy = self.__get_accuracy__(y_pred=y_pred, y=y)
        f1_score = self.__get_f1_score__(y_pred=y_pred, y=y)

        loss = F.cross_entropy(y_pred.float(), y)

        tensorboard_logs = {'loss': loss,
                            'acc': accuracy,
                            'f1_score': f1_score}

        res = {'loss': loss,
               'acc': accuracy,
               'f1_score': f1_score,
               'log': tensorboard_logs}

        return res

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        """
        Calculates statistics on training epoch end

        :param outputs: statistics for every training step
        :return: dictionary with averaged stats
        """

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_f1_score = torch.stack([x['f1_score'] for x in outputs]).mean()

        tensorboard_logs = {'loss': avg_loss,
                            'acc': avg_loss,
                            'f1_score': avg_f1_score}

        res = {'loss': avg_loss,
               'acc': avg_acc,
               'f1_score': avg_f1_score,
               'log': tensorboard_logs}

        return res

    def validation_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> Dict:
        """
        Performs validation step

        :param batch: one batch
        :param batch_idx: index of this batch
        :return: dictionary with stats
        """

        x, y = batch

        y_pred = self(x.float())

        accuracy = self.__get_accuracy__(y_pred=y_pred, y=y)
        f1_score = self.__get_f1_score__(y_pred=y_pred, y=y)

        loss = F.cross_entropy(y_pred.float(), y)

        tensorboard_logs = {'val_loss': loss,
                            'val_acc': accuracy,
                            'val_f1_score': f1_score}

        res = {'val_loss': loss,
               'val_acc': accuracy,
               'val_f1_score': f1_score,
               'log': tensorboard_logs}

        return res

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        """
        Calculates statistics on validation epoch end

        :param outputs: statistics for every validation step
        :return: dictionary with averaged stats
        """

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_f1_score = torch.stack([x['val_f1_score'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss,
                            'val_acc': avg_acc,
                            'val_f1_score': avg_f1_score}

        res = {'val_loss': avg_loss,
               'val_acc': avg_acc,
               'val_f1_score': avg_f1_score,
               'log': tensorboard_logs}

        return res

    def configure_optimizers(self) -> Tuple[List, List]:
        """
        Configures optimizers and schedulers

        :return: [optimizers], [schedulers]
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=10, eta_min=self.hparams['learning_rate'] / 1000)

        scheduler = {
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def prepare_data(self) -> None:
        """
        Prepares dataset for training and validation
        """

        birds_dataset_train, birds_dataset_valid = get_train_val_dataset(
            file_path=self.file_path,
            valid_percent=0.3,
            width=self.width,
            size=self.size,
            n_classes=self.n_classes
        )

        self.train_dataset = birds_dataset_train
        self.valid_dataset = birds_dataset_valid

    def train_dataloader(self) -> DataLoader:
        """
        Creates train dataloader

        :return: train dataloader
        """

        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=12,
                                      shuffle=True, )

        return train_dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Creates validation dataloader

        :return: validation dataloader
        """

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

    model = ClassificationModel(n_classes=2,
                                file_path=data_config.dataset_path,
                                batch_size=4,
                                hparams=hparams,
                                model_name='efficientnet-b0',
                                width=512,
                                size=(224, 224))

    torchsummary.summary(model.model.cuda(), input_size=(1, 256, 256))


