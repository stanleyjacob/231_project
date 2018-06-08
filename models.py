
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048):

        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        metrics = ['accuracy']

        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()

        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)


    def lstm(self):
        model = Sequential()
        model.add(LSTM(256, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model

    def mlp(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(2048))
        model.add(Dropout(0.2))
        model.add(Dense(2048))
        model.add(Dropout(0.2))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model

    def threedconvolution(self):
        model = Sequential()
        model.add(Conv3D(64, (4,4,4), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(128, (4,4,4), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(256, (4,4,4), activation='relu'))
        model.add(Conv3D(256, (4,4,4), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(512, (4,4,4), activation='relu'))
        model.add(Conv3D(512, (4,4,4), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(2048))
        model.add(Dropout(0.25))
        model.add(Dense(2048))
        model.add(Dropout(0.25))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
