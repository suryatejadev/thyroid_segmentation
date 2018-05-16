# Thyroid Nodule Segmentation

This repository contains code and models to segment thyroid nodules in ultrasound images.
Dataset used: [Open-CAS Ultrasound Dataset](http://opencas.webarchiv.kit.edu/?q=node/29)

## Installation

The main code is written as a Python package named 'tnseg'. After cloning this
repository to your machine, install with:

```bash
cd cloned/path
pip install .
```

You should then be able to use the package in Python:

```python
import matplotlib.pyplot as plt
from tnseg import dataset, models, loss, opts, evaluate
```

## Running models

Scripts for model training and evaluation are located under /scripts/.

```bash
python -u scripts/train.py config_files/defaults.config
```

On running the model, the outputs are saved in the outputs/ folder, in a folder named with the experiment name (this should be specified in the config file). The outputs include the following:
1. weights/ : Weights saved during the training.
2. results/ : The error and accuracy plots, validation dice coefficients
3. predictions/ : Predicted annotation maps of all the validation folders

Note: In this project, the dataset contains 16 folders. Due to the limited nature of the dataset, we trained 8 models (14 train and 2 validation), and obtained the validation dice coefficients of all the folders.

Note: this package is written with the Tensorflow backend in mind -- (batch,
height, width, channels) ordered is assumed and is not portable to Theano.

## Models

The implemented models are:
1. UNet
2. Window UNet
3. Dilated UNet
4. Dilated Densenet


