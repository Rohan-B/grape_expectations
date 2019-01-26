# Grape Expectations

## Introduction
Uses a Convolutional Neural Net to determine the type of wine based on a review.
Wine reviews found on [kaggle](https://www.kaggle.com/zynicide/wine-reviews).

## Usage
To train the model
```
./main.py -dataset=X
```
* With X=0, the model will train on classifying the wine as red or white.
* With X=1, the model will train on classifying the wine by type (e.g Pinot Noir, Cabarnet).

To predict
```
./main.py -predict=Z -snapshot=S
```
* Z should be a description of a wine review passed in as a string
* S should be the location of the model used to predict the classification passed in as a string

## Final Report
* [PDF](Final_Paper.pdf)

## References
* [CNN Text Classification for Pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch)
