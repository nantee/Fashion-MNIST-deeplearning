
# **Fashion MNIST Deep learning**
Train a deep Convolutional neural network to classify Zalando’s article images with keras. In general, the color is probably irrelevant when distinquishing different types of clothes since they vary substantially in many clothingtypes. Shape however is relevant and therefor a grayscale is set in the beginning eliminating any problem the RGB dimensions might add to the CNN's featuremapping. In some cases emblems on a t-shirt for example might add unnecessary information, no effort was put in regarding removal of theses emblems, but it could definitely be considered.


## Dependencies
* tensorflow
* keras
* pandas
* numpy
* mathplotlib

## Example 
15 example modelpredictions displaying the CNN's performance with title coloring (green = correct, red = non-correct).
<img src="assets/Fashion_15_30.png" width="850" height="850" />

## MNIST for beginners in tensorflow:
https://www.tensorflow.org/tutorials/

## Download Fashion MNIST data:
https://www.kaggle.com/zalando-research/fashionmnist


