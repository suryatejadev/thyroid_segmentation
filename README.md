# Thyroid Nodule Segmentation

This repository contains code and models to segment thyroid nodules in ultrasound images.

For the problem description, models and results, please see the blog post
[here](https://suryatejadev.github.io/thyroid-segmentation/).

## Installation

The main code is written as a Python package named `tnseg'. After cloning this
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

Note: this package is written with the Tensorflow backend in mind -- (batch,
height, width, channels) ordered is assumed and is not portable to Theano.
