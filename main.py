from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.python.keras.datasets import fashion_mnist

## Data dependencies
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

def read_fashion_mnist(x_train, y_train, x_test, y_test):
    ## Reading in data    
    print('Training data have a proportion of {}'.format(len(x_train) / ( len(x_train) + len(x_test)))) 
    print('Test data have a proportion of {}'.format(1- len(x_train) / ( len(x_train) + len(x_test))))
    
    ## Input image dimensions
    img_rows, img_cols = 28, 28
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    
    ## Transform data from (60000, 784) to (60000, 28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
    ## Normalization (pixel values varies from 0 to 255)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    ## Model performance
    batch_size = 128
    num_classes = 10
    epochs = 8
    
    ## Convert classvectors to binary class matrix
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test

x_train, y_train, x_test, y_test = read_fashion_mnist(x_train, y_train, x_test, y_test)


def CNN_model():
    ## Define model
    model = Sequential()
    
    model.add( Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape = input_shape) )
    model.add( Conv2D(62, (3, 3), activation='relu') )
    model.add( MaxPooling2D(pool_size=(2, 2)) )
    model.add( Dropout(0.5) )
    model.add( Flatten() )
    model.add( Dense(142, activation='relu' ))
    model.add( Dropout(0.5) )
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
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle']
    for i in range(start, end):
            ## In case of correct classfication, plot with green maintitle else red.
            print("X = {0}, Predicted class = {1} with probability {2:.4f}".format(x_test[i], labels[predicted_class[i]], predicted_proba[i][predicted_class[i]]))
            plt.imshow(x_test[i,:,:, 0])
            plt.title('True class: {0} \n Predicted class: {1} \n Probability: {2:.4f}'.format(labels[predicted_class[i]], labels[y_test_long[i]], predicted_proba[i][predicted_class[i]]),
             color = 'green' if labels[predicted_class[i]] == labels[y_test_long[i]] else 'red', size = 12)
            plt.show()

