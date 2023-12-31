'''Copyright 2023 Daniel Peev <danipeev1@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
import warnings
warnings.simplefilter (action = 'ignore', category = FutureWarning)

#Loading the arrays representing the inputs and the labels
input = np.load("D:\TsarevitsiInputs.npy")
labels = np.load("D:\TsarevitsiLabels.npy")

#Shuffling the data to avoid ordering biases
input, labels = shuffle (input, labels)

#The cutoff point between the training and validation sets
#The first 80% of the data are taken for the training set and
#the rest for the validation set
cutoff_point = int (labels.shape [0] * 0.8)

#The training sets
input_train =  input [:cutoff_point]
labels_train = labels [:cutoff_point]

#Reshaping the input so as to fit the neural network dimensions
input_train_reshaped = []

for ip in input_train:
    input_train_reshaped.append(ip.reshape (input_train [0].shape[1], input_train [0].shape[2], input_train [0].shape[0]))

input_train_reshaped = np.array(input_train_reshaped)

#These four lines reshape the labels array representing the lables
#of the traininng set so that each label is a one dimensional
#numpy array to fit with the shape of the final dense layer of the network

labels_train_reshaped = []

for label in labels_train:
    labels_train_reshaped.append(label.reshape(labels_train [0].shape [1] * labels_train [0].shape[2], labels_train [0].shape [0]))
labels_train_reshaped = np.array (labels_train_reshaped)

#The validation sets
input_validation =  input [cutoff_point:]
labels_validation = labels [cutoff_point:]

#Reshaping the input so as to fit the neural network dimensions
input_validation_reshaped = []

for ip in input_validation:
    input_validation_reshaped.append( ip.reshape (input_train [0].shape[1], input_train [0].shape[2], input_train [0].shape[0]))

input_validation_reshaped = np.array(input_validation_reshaped)


#These four lines reshape the labels array representing the lables
#of the validation set so that each label is a one dimensional
#numpy array to fit with the shape of the final dense layer of the network
labels_validation_reshaped = []

for label in labels_validation:
    labels_validation_reshaped.append(label.reshape(labels_train [0].shape [1] * labels_train [0].shape[2], labels_train [0].shape [0]))

labels_validation_reshaped = np.array (labels_validation_reshaped)
print (labels_validation_reshaped.shape)

#The architecture of the neural network
#This first model is without pooling layers, which
#might work since the images we will pass through it
#have rather small dimensions
print (input_train.shape)
model = Sequential ([
    Conv2D(filters = 8, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same', input_shape = (120, 80, 2)),
    Conv2D(filters = 16, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same'),
    Conv2D(filters = 8, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same'),
    Flatten(),
    Dense (units = labels_train [0].shape [1] * labels_train [0].shape[2], activation='sigmoid')])

#A summary of the model architecture
print (model.summary())

#Compiling the model
model.compile (optimizer = Adam(learning_rate = 0.001), loss='mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError()])

#Fitting the model on the data and outputing the progress on each epoch
model.fit (input_train_reshaped, labels_train_reshaped, validation_data = (input_validation_reshaped, labels_validation_reshaped), epochs = 10, verbose = 2)