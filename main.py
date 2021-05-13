import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

#The goal of this CNN is to classify hand pictures showing fingers
#up or down, we want to be able to tell how many fingers are up on a hand

def	manual_train_test_split(train_dataset, test_dataset):
	"""
		Train test split method
	"""
	x_train = np.array(train_dataset["train_set_x"][:])
	y_train = np.array(train_dataset["train_set_y"][:])
	x_test = np.array(test_dataset["test_set_x"][:])
	y_test = np.array(test_dataset["test_set_y"][:])
	return x_train, y_train, x_test, y_test

def	reshape_y(y_train, y_test, mode = 1):
	"""
		Reshape train and test labels
		-> mode = 1 is for 'classic' reshape
		-> mode = 2 is for converting an integer
			corresponding to the digit labeled to one image
			to a verctor (len 5) with 0 and one 1 that corresponds
			to the digit,
			so if we have [0, 0, 0, 0, 1] it means that the 
			label corresponds to a 5
	"""
	if mode == 1:
		y_train = y_train.reshape((1, y_train.shape[0]))
		y_test = y_test.reshape((1, y_test.shape[0]))
	else:
		y_train = np.eye(6)[y_train.reshape(-1)]
		y_test = np.eye(6)[y_test.reshape(-1)]
	return y_train, y_test

def	normalize_data(x_train, x_test):
	"""
		Data normalization
		-> by 255 because we want the rbg activation pixel range
		to fit between 0.0 and 1.0
	"""
	normalized_x_train = x_train.astype('float32') / 255
	normalized_x_test = x_test.astype('float32') / 255
	return normalized_x_train, normalized_x_test

def	build_model():
	model = Sequential()
	# Convolutional layers, just applying filters onto the picture to extract
	# important features
	# We mulity every pixel with a weight defined inside the kernel_size matrix
	# We make the matrix multiply its weights with every block (same size as the matrix)
	# of pixels from the image

	# 32 is the number of filters that we use for the image
	# kernel_size is the filter size (a 2d matrix) 
	# activation function to activate neuron before passing the value to the next layer
	# input shape is pretty explicit I think
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))

	# Pooling layer takes squares from our image and extract the max value from the square
	# The size is defined by : square_root(number_of_filters) / 2
	# -> pool_size : 32**0.5 / 2 = 2,
	# and since its a matrix, pool_size equals to (2, 2)
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Flatten layer goal is the transform to 1d the image data instead of 3d,
	# we are keeping the same values but we just transform it to a vector
	model.add(Flatten())

	# Dense layer contains 6 neurons fully connected
	# -> Fully connected to the output layer
	# We use softmax for the output so we cant get probabilistic outputs
	model.add(Dense(6, activation='softmax'))
	model.summary()

	# The cost function helps to see wrong our model is, the lower the cost the better
	# Stochastic gradiant descent used for optimization
	# Accurary helps to see how the model is performing, the closer to 1 the better
	# -> 1 = 100% precision
	model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
	return model

if __name__ == "__main__":
	# Extracting train and test datasets
	train_dataset = h5py.File('dataset/data/train_signs.h5', 'r')
	test_dataset = h5py.File('dataset/data/test_signs.h5', 'r')

	# Splitting the datasets to get x_train (training data), y_train (training labels),
	# x_test (test data), y_test (test labels)
	x_train, y_train, x_test, y_test = manual_train_test_split(train_dataset, test_dataset)

	# Getting existing classification possibilities
	classes = np.array(test_dataset["list_classes"][:])

	# Reshaping labels
	y_train, y_test = reshape_y(y_train, y_test, 1)
	y_train, y_test = reshape_y(y_train, y_test, 2)
	
	input_shape = x_train[0].shape

	# Data normalization
	x_train, x_test = normalize_data(x_train, x_test)

	model = build_model()
	model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=2)
	model.save_weights("model_weights.h5")
