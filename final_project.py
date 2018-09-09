
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import sys
sys.path.append('/home/stan/Desktop/pulled_code/deep_visual_keras')
from guided_backprop import GuidedBackprop
from utils import *
from keras.applications.vgg16 import VGG16

import numpy as np

import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

import json
import random

import os
import pickle
import cv2

batch_size = 64
num_classes = 4
epochs = 12

# input image dimensions
img_rows, img_cols = 50, 50

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

params_file  = open('params.json', 'r')
params = json.load(params_file)
params_file.close()
preprocess_params = params['preprocess']

BASE_FOLDER = preprocess_params['BASE']
FIST = preprocess_params['FIST']
HAND = preprocess_params['HAND']
# NONE = preprocess_params['NONE']
ONE = preprocess_params['ONE']
PEACE = preprocess_params['PEACE']
ALL_IMAGES_PATHS = [FIST, HAND, ONE, PEACE] #change here if you want to add new gesture

random.seed(0)

images = []     # this will contain list of all images preprocessed
n_types = preprocess_params['n_types']
X_train = None
y_train = None
X_test = None
y_test = None
X_validation = None
y_validation = None
path_train = None
path_test = None
path_validation = None

tX_train = []
tX_test = []
tX_validation = []
ty_train = []
ty_test = []
ty_validate = []
tpath_train = []
tpath_test = []
tpath_validate = []

def loadAllImages(path, curr_type):
    file_y = open(path + 'y.txt', 'r')
    y_to_append = curr_type
    file_y.close()
    temp_X = []
    temp_y = []
    temp_path = []
    for f in os.listdir(path):
        if(f != 'y.txt'):
            img = cv2.imread(path + str(f), cv2.COLOR_BGR2GRAY).astype(float)
            img -= np.mean(img)
            
            temp_X.append(img)
            temp_y.append(y_to_append)
            temp_path.append(path + str(f))

    return temp_X, temp_y, temp_path

def shuffleAndSplit(X, y, path):
    l = len(X)
    l_train = int(preprocess_params['train'] * l)
    l_test = int(preprocess_params['test'] * l)
    upper = l_train + l_test
    return X[0:l_train], X[l_train:upper], X[upper:], y[0:l_train], \
        y[l_train:upper], y[upper:], path[0:l_train], path[l_train:upper], path[upper:]

for curr_type in range(n_types):
    print(curr_type)
    local_X, local_y, local_path = loadAllImages(ALL_IMAGES_PATHS[curr_type], curr_type)
    
    lX_train, lX_test, lX_validate, ly_train, ly_test, \
        ly_validate, lpath_train, lpath_test, lpath_validate = \
        shuffleAndSplit(local_X, local_y, local_path)
    
    #adds to main array
    [tX_train.append(x) for x in lX_train]
    [tX_test.append(x) for x in lX_test]
    [tX_validation.append(x) for x in lX_validate]
    [ty_train.append(x) for x in ly_train]
    [ty_test.append(x) for x in ly_test]
    [ty_validate.append(x) for x in ly_validate]
    [tpath_train.append(x) for x in lpath_train]
    [tpath_test.append(x) for x in lpath_test]
    [tpath_validate.append(x) for x in lpath_validate]


#convert into numpy array
X_train = np.dstack(tX_train)
X_test = np.dstack(tX_test)
X_validation = np.dstack(tX_validation)
y_train = np.dstack(ty_train)
y_test = np.dstack(ty_test)
y_validation = np.dstack(ty_validate)
path_train = np.dstack(tpath_train)
path_test = np.dstack(tpath_test)
path_validation = np.dstack(tpath_validate)

#rotate the axis
X_train = np.rollaxis(X_train, -1)
X_test = np.rollaxis(X_test, -1)
X_validation = np.rollaxis(X_validation, -1)
y_train = np.rollaxis(y_train, -1)
y_test = np.rollaxis(y_test, -1)
y_validation = np.rollaxis(y_validation, -1)
path_train = np.rollaxis(path_train, -1)
path_test = np.rollaxis(path_test, -1)
path_validation = np.rollaxis(path_validation, -1)

train = np.random.permutation(len(X_train))

X_train, y_train, path_train = X_train[train],\
    y_train[train], path_train[train]
    
test = np.random.permutation(len(X_test))

X_test, y_test, path_test = X_test[test],\
    y_test[test], path_test[test]

validation = np.random.permutation(len(X_validation))

X_validation, y_validation, path_validation = \
    X_validation[validation], y_validation[validation], path_validation[validation]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_validation.shape)
print('Validation labels shape: ', y_validation.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

y_train = np.squeeze(y_train)
y_validation = np.squeeze(y_validation)
y_test = np.squeeze(y_test)


print(np.shape(X_train))
print(np.shape(y_train))

print(np.shape(X_test))
print(np.shape(y_test))

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_validation = X_validation.reshape(X_validation.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_validation = X_validation.astype('float32')

X_train /= 255
X_test /= 255
X_validation /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_validation.shape[0], 'val samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(32, 32),
                 activation='relu',
                 input_shape=input_shape))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_validation, y_validation), callbacks=[tbCallBack])
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

g_backpropagation = GuidedBackprop(model)

current_img = X_train[10]
current_img_rs = np.reshape(current_img, (50, 50))
plt.imsave("/home/stan/Desktop/pulled_code/baby_result_og.jpg", current_img_rs)
current_mask = g_backpropagation.get_mask(current_img)
# print(current_mask)
show_image(current_mask)
