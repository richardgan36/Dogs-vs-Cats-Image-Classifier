# Dogs-vs-Cats-Image-Classifier

![GitHub](https://img.shields.io/github/license/richardgan36/Dogs-vs-Cats-Image-Classifier)

## Introduction

This is a an image classification model which leverages convolutional neural networks to classify images as either containing dogs or cats.
This is a training project to help me learn machine learning.


<img src="https://github.com/richardgan36/Dogs-vs-Cats-Image-Classifier/blob/main/screenshots/two_cats.jpg" width=50% height=50%>


## Installation

Python 3.10

### Required External Libraries

* Tensorflow 2.9.2
* Keras 2.9.0
* Numpy 1.22.3

### Training and Testing Dataset

The training and testing images can be downloaded from [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data).
Be sure to update the directory name in dogsVcats_train.py.


## Limitations

An image must contain at least one cat or dog, but not both. Multiple instances of the same animal are fine. The dataset from [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data) already satisfies this condition.

## Usage

Use dogsVcats_train.py to train the image classification model. Pass the filepath of your training images to the variable `FILEPATH` and the name of the file that the model should be saved as to the variable `MODEL_NAME`, then run the program.
To test a model on a single image, use dogsVcats_predict.py. Load a model, pass in the filepath of the image, then run the program.


