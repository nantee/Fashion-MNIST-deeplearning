
# **Fashion MNIST Deep learning**
Train a deep Convolutional neural network to classify Zalando’s article images with keras. 

## Dependencies
* tensorflow
* keras
* pandas
* numpy
* mathplotlib

## How to

There's a mainscript `main.py` performing step 1 and 2.
1) Read csv files with a 60000/10000 split (Train and test) to pandas dataframes. Name class-numbers to their corresponding clothing type. Transform dataframes to numpyarrays and reshape them to 28x28 arrays and normalize the values varying between 0 to 255. 

2) Specify batch size, number of classes and epochs. Convert categorical vectors to binary class matrix enabling matrix-operations within the network. Define a sequential model and add convolutional, pooling and fully connected layers. Hyperparameters are intuitively predefined and relevant loss function and evaluation metrices are set. The model history, categorical and probability predictions are saved and visual analysis on the models performance are evaluated in a defined function. 

The goal of the visualization is to see what decision the model takes, what are the probabilities which lead to it and what are the true label of the image.


### MNIST for beginners in tensorflow:
https://www.tensorflow.org/tutorials/

### Download Fashion MNIST data:
https://www.kaggle.com/zalando-research/fashionmnist


