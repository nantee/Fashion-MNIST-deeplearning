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
import glob
import moviepy.editor as mpy



class FashionMnistClassifier:
    
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
    
        ## Reading in data    
        print('Training data have a proportion of {}'.format(len(self.x_train) / ( len(self.x_train) + len(self.x_test)))) 
        print('Test data have a proportion of {}'.format(1- len(self.x_train) / ( len(self.x_train) + len(self.x_test))))
        
        ## Input image dimensions
        img_rows, img_cols = 28, 28
        
        ## Transform data from (N, 28,28) to (N, 28, 28, 1) and normalize
        self.x_train = np.expand_dims( (self.x_train - 127.5 ) /127.5, axis = -1)
        self.x_test = np.expand_dims( (self.x_test - 127.5 ) /127.5, axis = -1)
        self.input_shape = (img_rows, img_cols, 1)
        
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
    def Visualize_predictions(self, start, end, output_path):
        labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle']
        fig = plt.figure(figsize=(12, 12))
        for i, j in zip(range(start, end), range(0, 20)):
            ## In case of correct classfication, plot with green maintitle else red.
            print("X = {0}, Predicted class = {1} with probability {2:.4f}".format(self.x_test[i], labels[self.predicted_class[i]], self.predicted_proba[i][self.predicted_class[i]]))
            fig.add_subplot(3, 5, j + 1)
            plt.imshow(self.x_test[i,:,:, 0], cmap = 'gray')
            plt.title('True class: {0} \n Predicted class: {1} \n Probability: {2:.4f}'.format(labels[pd.DataFrame(self.y_test)[0][i]], labels[self.predicted_class[i]], self.predicted_proba[i][self.predicted_class[i]]),
            color='green' if labels[self.predicted_class[i]] == labels[pd.DataFrame(self.y_test)[0][i]] else 'red', size=10)
        plt.savefig(output_path + 'Fashion_{0}_{1}.png'.format(start, end))
        plt.close()
    
    def Gif_maker(self, dir_path):
        img_start = 0
        img_end = 150
        for i in range(img_start, img_end, 15):
            self.Visualize_predictions(i, i + 15, dir_path)
        
        gif_name = 'Graph-metric'
        fps = 0.50
        file_list = glob.glob('*.png') # Get all the pngs in the current directory
        list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.png')[0])) # Sort the images by #, this may need to be tweaked for your use case
        clip = mpy.ImageSequenceClip(file_list, fps=fps)
        clip.write_gif('{}.gif'.format(gif_name), fps=fps)
        
       
