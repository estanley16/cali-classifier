#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:16:06 2020

@author: emmastanley
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.models import load_model 
import matplotlib.pyplot as plt
import pickle 


#open the data files 
X=pickle.load(open('X.pickle', 'rb'))
y=pickle.load(open('y.pickle', 'rb'))

#normalize pixel values 
X=X/255.0

#build the model 
model= Sequential()

#start with 3 convolutional layers
model.add(Conv2D(32, (3,3), input_shape=X.shape[:1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#make 2 hidden layers 
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))

#output layer with 3 neurons for 3 classes
model.add(Dense(3))
model.add(Activation('softmax'))

#compile the model with basic parameters
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

