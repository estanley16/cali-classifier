#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:02:09 2020

@author: emmastanley
"""
 

import numpy as np
import os 
import matplotlib.pyplot as plt
import cv2
import random 
import pickle


file_list=[]
class_list=[]

directory='/Users/emmastanley/Documents/Machine Learning/cali_classifier/data'

#a list of categories for the classifier
categories=['happy', 'sleepy', 'neutral']
            
imgsize=100

#check for images in folder
for category in categories: 
    path=os.path.join(directory, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        

training_data=[]


def create_training_data():
    for category in categories:
        path=os.path.join(directory, category)
        class_num=categories.index(category)
        for img in os.listdir(path):
            try: 
                img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array, (imgsize, imgsize))
                training_data.append([new_array, class_num])
            except Exception as e: 
                pass
            
            
create_training_data()
random.shuffle(training_data)

X=[] #features
y=[] #labels

for features, label in training_data: 
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, imgsize, imgsize, 1)


#pickle X and y 
pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close() 


