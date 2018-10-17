from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn import metrics

## data dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('C:/Sogeti_Projekt/Python_deeplearning_IDLE/Fashion-MNIST_data/fashion-mnist_train.csv')
test = pd.read_csv('C:/Sogeti_Projekt/Python_deeplearning_IDLE/Fashion-MNIST_data/fashion-mnist_test.csv')


len(train) / ( len(train) + len(test))
1- len(train) / ( len(train) + len(test))


train.head(5)
test.head(5)
