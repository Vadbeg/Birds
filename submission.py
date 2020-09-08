import torch

from modules.model.model import ClassificationModel


if __name__ == '__main__':
    model = ClassificationModel(n_classes=264)

    lightning_model_checkpoint = torch.load('firstfirst_ckpt_epoch_14.ckpt')
    model_state_dict = lightning_model_checkpoint['state_dict']

    model.load_state_dict(model_state_dict)

