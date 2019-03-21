# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:37:08 2019

@author: rohit
"""
import numpy as np
import pandas as pd
import os
import keras
import matplotlib.pyplot as plot
from keras.applications import MobileNet
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.optimizers import Adam


    
    
#model architecture
base_model=MobileNet(weights='imagenet',include_top=False,input_shape=[224,224,3]) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(200,activation='softmax')(x) #final layer with softmax activation


#model 
model=Model(inputs=base_model.input,outputs=preds)


for i,layer in enumerate(model.layers):
    print(i,layer.name)

for layer in model.layers:
    layer.trainable=False

#training using ImagedataGenerator in Keras

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory('C:/Users/rohit/Downloads/tiny-imagenet-200/tiny-imagenet-200/train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=5)
    
