import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
import warnings
warnings.simplefilter (action = 'ignore', category = FutureWarning)

#Loads the dataset
dataset = np.load("D:\dataset.npy", allow_pickle = True)

#These lists are used to pass the inputs and labels to train the network
input = []
labels = []

#Converting the dataset to a format which the CNN can accept
for i in dataset:

    datapoint = i ['x']
    datapoint = np.resize (datapoint, (1, 230, 90))
    input.append (datapoint)
    labels.append (i ['y'])

input = np.array(input)
labels = np.array (labels)

#Shuffling the data to avoid ordering biases
input, labels = shuffle (input, labels)


#The cutoff point between the training and validation sets
#The first 80% of the data are taken for the training set and
#the rest for the validation set
cutoff_point = int (labels.shape [0] * 0.8)

#The training sets
input_train =  input [:cutoff_point]
labels_train = labels [:cutoff_point]

#These four lines reshape the labels array representing the lables
#of the traininng set so that each label is a one dimensional
#numpy array to fit with the shape of the final dense layer of the network

labels_train_reshaped = []

for label in labels_train:
    labels_train_reshaped.append(label.reshape(labels_train [0].shape [0] * labels_train [0].shape[1]))
labels_train_reshaped = np.array (labels_train_reshaped)

#The validation sets
input_validation =  input [cutoff_point:]
labels_validation = labels [cutoff_point:]

#These four lines reshape the labels array representing the lables
#of the validation set so that each label is a one dimensional
#numpy array to fit with the shape of the final dense layer of the network
labels_validation_reshaped = []

for label in labels_validation:
    labels_validation_reshaped.append(label.reshape(labels_train [0].shape [0] * labels_train [0].shape[1]))

labels_validation_reshaped = np.array (labels_validation_reshaped)


#The architecture of the neural network
#This first model is without pooling layers, which
#might work since the images we will pass through it
#have rather small dimensions

print (input_train.shape)
model = Sequential ([
    Conv2D(filters = 32, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same', input_shape = (1, 230, 90)),
    Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
    Flatten(),
    Dense (units = labels_train [0].shape [0] * labels_train [0].shape[1], activation='sigmoid')])

#A summary of the model architecture
print (model.summary())

#Compiling the model
model.compile (optimizer = Adam(learning_rate = 0.001), loss='mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError()])

#Fitting the model on the data and outputing the progress on each epoch
model.fit (input_train, labels_train_reshaped, validation_data = (input_validation, labels_validation_reshaped), epochs = 5, verbose = 2)