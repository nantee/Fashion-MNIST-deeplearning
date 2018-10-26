from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

## data dependencies
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
import time
import keras

## Check for tensorflow-gpu (running on NVIDIA Quadro M1000M)
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

## Reading in data

train = pd.read_csv('C:/Sogeti_Projekt/Python_deeplearning_IDLE/Fashion-MNIST_data/fashion-mnist_train.csv')
test = pd.read_csv('C:/Sogeti_Projekt/Python_deeplearning_IDLE/Fashion-MNIST_data/fashion-mnist_test.csv')

print('Training data have a proportion of {}'.format(len(train) / ( len(train) + len(test)))) 
print('Test data have a proportion of {}'.format(1- len(train) / ( len(train) + len(test))))

x_train = train.drop('label', axis=1)
y_train_long = train['label']

x_test = test.drop('label', axis=1)
y_test_long = test['label']

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle']


## Input image dimensions
img_rows, img_cols = 28, 28

x_train = np.array(x_train)
x_test = np.array(x_test)


## Transform data from (60000, 784) to (60000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

## Plot any sequence of clothes
def VisualizeImages(start, end):
    for i, j in zip(x_train[start:end,:,:,0], y_train[start:end]):
        plt.imshow(i)
        plt.title('Truth: {0}'.format(labels[j]))
        plt.show()


## Normalization (pixel values varies from 0 to 255)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


## Model performance
batch_size = 128
num_classes = 10
epochs = 8

y_train = keras.utils.to_categorical(y_train_long, num_classes)
y_test = keras.utils.to_categorical(y_test_long, num_classes)



def CNN_model():
    model = Sequential()
    
    model.add( Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape = input_shape) )
    model.add( Conv2D(62, (3, 3), activation='relu') )
    model.add( MaxPooling2D(pool_size=(2, 2)) )
    model.add( Dropout(0.1) )
    model.add( Flatten() )
    model.add( Dense(142, activation='relu' ))
    model.add( Dropout(0.1) )
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss = keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.Adam(),
                metrics=['accuracy'])
                
    history = model.fit(x_train, y_train,
            batch_size = batch_size,
            epochs = epochs,
            verbose = 1,
            validation_data=(x_test, y_test))
            
    score = model.evaluate(x_test, y_test, verbose=0)
    predicted_class = model.predict_classes(x_test, batch_size = batch_size, verbose=0)
    predicted_proba = model.predict_proba(x_test, batch_size = batch_size, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(history.history.keys())
    
    # summarize history for accuracy
    plt.plot(history.history['acc'], color = 'blue', marker = 'o')
    plt.plot(history.history['val_acc'], color = 'red', marker = 'o')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'], color = 'blue', marker = 'o')
    plt.plot(history.history['val_loss'], color = 'red', marker = 'o')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    print('%s, %s, %s and %s are returned' % ('Score', 'Predicted_class', 'Predicted_proba', 'History') )
    return(score, predicted_class, predicted_proba, history)


score, predicted_class, predicted_proba, history = CNN_model()


## Visualize predictions, truth and probabilities
def VisualizePredictions(start, end):
    for i in range(start, end):
        print("X = {0}, Predicted class = {1}".format(x_test[i], labels[predicted_class[i]]))
    for i in range(start, end):
        ## In case of correct classfication, plot with green maintitle
            plt.imshow(x_test[i,:,:, 0])
            plt.title('True class: {0} \n Predicted class: {1} \n with probability {2:.4f}'.format(labels[predicted_class[i]], labels[y_test_long[i]], predicted_proba[i][predicted_class[i]]),
             color = 'green' if labels[predicted_class[i]] == labels[y_test_long[i]] else 'red', size = 12)
            plt.show()

