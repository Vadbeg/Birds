# Cornell Birdcall Identification competition

This is the code for [Cornell Birdcall Identification](https://www.kaggle.com/c/birdsong-recognition/overview) challenge hosted on Kaggle

## Data

[Librosa](https://github.com/librosa/librosa) library is pretty slow for reading and transforming audio. 
So, I read data using librosa and saved it as [HDF5](https://www.h5py.org) file. 
More about that you can read [here](https://www.kaggle.com/c/birdsong-recognition/discussion/181456).

Script for transforming `.mp3` to hdf5: `create/read_and_transform_audio.py`

## Augmentations 

Augmentations are useful for better models generalization. 
I've used [albumentations](https://github.com/albumentations-team/albumentations) 
library and [this](https://www.kaggle.com/tanulsingh077/audio-albumentations-transform-your-audio) Kaggle notebook to build augmentations for spectrograms transforming.

Code for this part tou can find here: `modules/data/augmentations`

## Model

I've used CNN for image classification for this task. 
Family of [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) models is the SOTA for image classification now, so I chose it. 
Also I've used [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to build training pipeline.

Model part: `modules/model`

## Built With

* [PyTorch](https://docs.djangoproject.com/en/2.2/) - Neural networks framework used
* [PyTorch Lightning](https://docs.djangoproject.com/en/2.2/) - For training pipeline
* [Albumentations](https://github.com/albumentations-team/albumentations) - Fot spectrogram augmentaions

## Authors

* **Vadim Titko** aka *Vadbeg* - [GitHub](https://github.com/Vadbeg/PythonHomework/commits?author=Vadbeg) 
| [LinkedIn](https://www.linkedin.com/in/vadim-titko-89ab16149/)