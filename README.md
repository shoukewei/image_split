# Descripstats

The **'image_split'** package provides a convenient tool for splitting image data into separate training, validation, and testing sets. With this package, you can easily manage the distribution of your image dataset for machine learning tasks. The package offers a simple function, **'image_split'**, that takes input and output directories, along with desired split ratios and an optional seed for randomization. The package is designed to streamline the process of data preparation, making it easier to organize and manage image datasets for model training and evaluation.

Developed by Shouke Wei, Ph.D. from Deepsim Academy, Deepsim Intelligence Technology Inc. (c) 2023

## Install the package
```python
pip install image_split
```

## import the package
```python
from image_split import train_val_test_split
```
then use the `train_val_test_split()` directly. Or 
```python
import image_split as sp
```
then use `sp.train_val_test_split()`

## Document
An example: https://github.com/shoukewei/descripstats/blob/main/docs/example.ipynb


