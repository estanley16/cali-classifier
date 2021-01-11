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

X_sample = X[(np.random.randint(0,(X.shape[0]))),:] #check out a random image from the set 
plt.imshow(X_sample, cmap='gray')
#plt.close()

train_split=int(0.7*(len(X)))
val_split=int(0.2*len(X))
test_split=int(0.1*(len(X)))

X_train = X[:train_split]
y_train = y[:train_split]

X_val = X[train_split:train_split+val_split]
y_val = y[train_split:train_split+val_split]

X_test = X[-test_split:]
y_test = y[-test_split:]

#%%

input_shape=X.shape[1:] 

#build the model 
model= Sequential()

#convolutional layers
model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))

#output layer with 3 neurons for 3 classes
model.add(Dense(3, activation='softmax'))

model.summary()
#%%
#compile the model with basic parameters
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%

#train the model
history=model.fit(X_train, y_train, batch_size=32, epochs=20, shuffle=True, validation_data=(X_val, y_val))
 
#%%
#save the model to json 
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
    
model.save_weights('model.h5')
print('saved model in local directory!')

model.save('CNN.model')

#%%

#show accuracy and loss history during training
print(history.history.keys())

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



#%%
#Evaluate the mode on test data
results = model.evaluate(X_test, y_test)
print('Test loss, test accuracy:', results)

predictions = model.predict(X_test[:3])
for i in range(predictions.shape[1]):
    test_img=X_test[i,:,:,:]
    plt.imshow(test_img, cmap='gray')
    plt.show()
    
    if np.argmax(predictions[i])==0:
        print('happy')
    if np.argmax(predictions[i])==1:
        print('sleepy')
    elif np.argmax(predictions[i])==2:
        print('neutral')
    
    if np.argmax(predictions[i])==y_test[i]:
        print('Correctly classified')
    else:
        print('Incorecctly classified')