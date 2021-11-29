import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPool2D, BatchNormalization, Dense, Flatten

class SudokuNet:
    @staticmethod
    def build( height, width, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        model.add(Conv2D(60, 5, input_shape=input_shape, padding='same', activation='relu'))
        model.add(Conv2D(60, 5, padding='same', activation='relu'))
        model.add(MaxPool2D(2))
        model.add(Conv2D(30, 3, padding='same', activation='relu'))
        model.add(Conv2D(30, 3, padding='same', activation='relu'))
        model.add(MaxPool2D(2, strides=2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        return model
