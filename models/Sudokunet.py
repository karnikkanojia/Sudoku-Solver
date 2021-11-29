import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPool2D, BatchNormalization, Dense, Flatten

class SudokuNet:
    @staticmethod
    def build( height, width, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        model.add(Conv2D(32, (5, 5), padding="same", input_shape=input_shape, activation='relu'))
		model.add(MaxPool2D(pool_size=(2, 2)))
		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
		model.add(MaxPool2D(pool_size=(2, 2)))
		# first set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(64), activation='relu')
		model.add(Dropout(0.5))
		# second set of FC => RELU layers
		model.add(Dense(64), activation='relu')
		model.add(Dropout(0.5))
		# softmax classifier
		model.add(Dense(classes), activation='softmax')
		# return the constructed network architecture
		return model

        return model
