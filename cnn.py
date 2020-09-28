#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:16:06 2020

@author: emmastanley
"""

#%% 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.models import load_model 
import matplotlib.pyplot as plt
import pickle 
#%%

#open the data files 
X=pickle.load(open('X.pickle', 'rb'))
y=pickle.load(open('y.pickle', 'rb'))


X=X/255.0#normalize pixel values 
y=np.asarray(y) #convert y from list to numpy array 

input_shape=X.shape[1:] 

#build the model 
model= Sequential()

#convolutional layers
model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.8))

# model.add(Conv2D(32, (3,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.8))

#hidden layers 
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#output layer with 3 neurons for 3 classes
model.add(Dense(3))
model.add(Activation('softmax'))

#compile the model with basic parameters
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%

#train the model
history=model.fit(X, y, batch_size=32, epochs=30, validation_split=0.2)
 
#%%
#save the model to json 
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
    
model.save_weights('model.h5')
print('saved model in local directory!')

model.save('CNN.model')

#%%

#show accuracy history during training
print(history.history.keys())
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
