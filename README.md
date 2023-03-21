# Dogs-vs-Cats-Image-Classifier

## Introduction

This is a an image classification model which leverages convolutional neural networks to classify images as either containing dogs or cats.
This is a training project to help me learn machine learning.


<img src="https://github.com/richardgan36/Dogs-vs-Cats-Image-Classifier/blob/main/screenshots/two_cats.jpg" width=50% height=50%>


## Installation

Python 3.10

### Required External Libraries

* Tensorflow
* Keras
* Numpy

### Training and Testing Dataset

The training and testing images can be downloaded from [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data).
Be sure to update the directory name in dogsVcats_train.py.


## Limitations

An image must contain at least one cat or dog, but not both. Multiple instances of the same animal are fine. The dataset from [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data) already satisfies this condition.

## Usage

Use dogsVcats_train.py to train the image classification model. Pass the filepath of your training images to the global variable `FILEPATH`, then run the program.


