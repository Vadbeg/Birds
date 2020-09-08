import h5py

import torchsummary
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger

from modules.config import DataConfig, ModelConfig
from modules.model.model_lightning import ClassificationModel


if __name__ == '__main__':
    data_config = DataConfig()
    model_config = ModelConfig()

    hparams = {
        'learning_rate': 0.01,
        'n_classes': data_config.n_classes,
        'max_epochs': 150,
        'batch_size': 35
    }

    model = ClassificationModel(n_classes=data_config.n_classes,
                                file_path=data_config.dataset_path,
                                batch_size=hparams['batch_size'],
                                hparams=hparams)

    checkpoing_callback = ModelCheckpoint(
        filepath=model_config.weights_folder,
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='first'
    )

    neptune_logger = NeptuneLogger(
        api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIs'
                'ImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa'
                '2V5IjoiMTIyODQyZGUtNTdiMS00MDBlLWEzZmYtMzU0N2Q4MDViMjQ0In0=',
        project_name='vadbeg/birds',
        experiment_name='resnet18, sigmoid, bceloss',
        params=hparams,
        tags=['pytorch-lightning', 'birds']
    )

    trainer = Trainer(gpus=1, num_nodes=1,
                      checkpoint_callback=checkpoing_callback,
                      max_epochs=hparams['max_epochs'],
                      logger=neptune_logger)

    trainer.fit(model=model)

    trainer = Trainer()

