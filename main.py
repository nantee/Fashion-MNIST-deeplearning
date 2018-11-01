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


class fashion_mnist_classifier:
       
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def read_fashion_mnist(self):
        ## Reading in data    
        print('Training data have a proportion of {}'.format(len(self.x_train) / ( len(self.x_train) + len(self.x_test)))) 
        print('Test data have a proportion of {}'.format(1- len(self.x_train) / ( len(self.x_train) + len(self.x_test))))
        
        ## Input image dimensions
        img_rows, img_cols = 28, 28
        
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        
        ## Transform data from (N, 784) to (N, 28, 28, 1)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, 1)
        self.input_shape = (img_rows, img_cols, 1)
        
        ## Normalization (pixel values varies from 0 to 255)
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255
        
        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')
        
        self.num_classes = len(np.unique(self.y_train))
        
        ## Convert classvectors to binary class matrix
        self.y_train_mat = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test_mat = keras.utils.to_categorical(self.y_test, self.num_classes)
        #return self.x_train, self.x_test, self.y_train, self.y_test
    
    def CNN_model(self, batch_size, epochs):
        ## Define model
        model = Sequential()
        
        model.add( Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape = self.input_shape) )
        model.add( Conv2D(62, (3, 3), activation='relu') )
        model.add( MaxPooling2D(pool_size=(2, 2)) )
        model.add( Dropout(0.5) )
        model.add( Flatten() )
        model.add( Dense(142, activation='relu' ))
        model.add( Dropout(0.5) )
        model.add(Dense(self.num_classes, activation='softmax'))
        
        model.compile(loss = keras.losses.categorical_crossentropy,
                    optimizer = keras.optimizers.Adam(),
                    metrics=['accuracy'])
        
        history = model.fit(self.x_train, self.y_train_mat,
                batch_size = batch_size,
                epochs = epochs,
                verbose = 1,
                validation_data=(self.x_test, self.y_test_mat))
        
        self.score = model.evaluate(self.x_test, self.y_test_mat, verbose=0)
        self.predicted_class = model.predict_classes(self.x_test, batch_size = batch_size, verbose=0)
        self.predicted_proba = model.predict_proba(self.x_test, batch_size = batch_size, verbose=0)
        
        print('Test loss:', self.score[0])
        print('Test accuracy:', self.score[1])
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
        return self.score, self.predicted_class, self.predicted_proba, history
    
    ## Visualize predictions, truth and probabilities
    def VisualizePredictions(self, start, end, output_path):
        labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle']
        fig = plt.figure(figsize=(8, 8))
        for i, j in zip(range(start, end), range(0, 20)):
            ## In case of correct classfication, plot with green maintitle else red.
            print("X = {0}, Predicted class = {1} with probability {2:.4f}".format(self.x_test[i], labels[self.predicted_class[i]], self.predicted_proba[i][self.predicted_class[i]]))
            fig.add_subplot(4, 5, j + 1)
            plt.imshow(self.x_test[i,:,:, 0])
            plt.title('True class: {0} \n Predicted class: {1} \n Probability: {2:.4f}'.format(labels[self.predicted_class[i]], labels[pd.DataFrame(self.y_test)[0][i]], self.predicted_proba[i][self.predicted_class[i]]),
            color='green' if labels[self.predicted_class[i]] == labels[pd.DataFrame(self.y_test)[0][i]] else 'red', size=8)
        plt.tight_layout()
        plt.savefig(output_path + 'Fashion_{0}_{1}.png'.format(start, end))
        plt.close()



fashion = fashion_mnist_classifier(x_train, y_train, x_test, y_test)
fashion.read_fashion_mnist()
fashion.CNN_model(128, 1)

for i in range(0, 200, 20):
    fashion.VisualizePredictions(i, i + 20, output_path)

