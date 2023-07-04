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
    datapoint = np.resize (datapoint, (1, 225, 90))
    input.append (datapoint)
    labels.append (i ['y'])

input = np.array(input)
labels = np.array (labels)

#A list containing the rmse values for the test set on each of the k validation stages
test_sets_rmse = []

#The number of validation stages
k = 5

#K-fold cross validation with k = 5
for i in range (0, k):
    #Shuffling the data before each of the k stages of validation
    input, labels = shuffle (input, labels)


    #The cutoff point between the training and validation sets
    #The first 60% of the data are taken for the training set and
    #the rest for the validation and test sets
    cutoff_point_train = int (labels.shape [0] * 0.6)

    # The cutoff point between the validation and test sets
    # The next 20% of the data are taken for the training set and
    # the final 20% are for the test set
    cutoff_point_validation = int(labels.shape[0] * 0.8)
    #The training sets
    input_train =  input [:cutoff_point_train]
    labels_train = labels [:cutoff_point_train]

    #These four lines reshape the labels array representing the lables
    #of the traininng set so that each label is a one dimensional
    #numpy array to fit with the shape of the final dense layer of the network

    labels_train_reshaped = []

    for label in labels_train:
        labels_train_reshaped.append(label.reshape(labels_train [0].shape [0] * labels_train [0].shape[1]))
    labels_train_reshaped = np.array (labels_train_reshaped)

    #The validation sets
    input_validation =  input [cutoff_point_train:cutoff_point_validation]
    labels_validation = labels [cutoff_point_train:cutoff_point_validation]

    #These four lines reshape the labels array representing the lables
    #of the validation set so that each label is a one dimensional
    #numpy array to fit with the shape of the final dense layer of the network
    labels_validation_reshaped = []

    for label in labels_validation:
        labels_validation_reshaped.append(label.reshape(labels_train [0].shape [0] * labels_train [0].shape[1]))

    labels_validation_reshaped = np.array (labels_validation_reshaped)

    #The test sets
    input_test = np.array(input[cutoff_point_validation:])
    labels_test = labels [cutoff_point_validation:]

    # These four lines reshape the labels array representing the lables
    # of the test set so that each label is a one dimensional
    # numpy array to fit with the shape of the final dense layer of the network
    labels_test_reshaped = []

    for label in labels_test:
        labels_test_reshaped.append(label.reshape(labels_train [0].shape [0] * labels_train [0].shape[1]))

    labels_test_reshaped = np.array(labels_test_reshaped)

    #The architecture of the neural network
    #This first model is without pooling layers, which
    #might work since the images we will pass through it
    #have rather small dimensions

    model = Sequential ([
        Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (1, 225, 90)),
        Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
        Conv2D(filters = 128, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same'),
        Flatten(),
        Dense (units = labels_train [0].shape [0] * labels_train [0].shape[1], activation='sigmoid')])
    #A summary of the model architecture
    print (model.summary())

    #Compiling the model
    model.compile (optimizer = Adam(learning_rate = 0.0001), loss='mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError()])

    #Fitting the model on the data and outputing the progress on each epoch
    model.fit (input_train, labels_train_reshaped, validation_data = (input_validation, labels_validation_reshaped), epochs = 10, verbose = 2)

    #Scoring the model on the test set
    print ("On the test set:")
    score = model.evaluate(input_test, labels_test_reshaped, verbose = 2)
    test_sets_rmse.append(score[1])

print ("The average RMSE on the test sets:")
print (np.mean(test_sets_rmse))


