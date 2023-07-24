from sklearn.utils import shuffle
import os
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout, BatchNormalization, \
    Input, concatenate, PReLU, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras import Model
from matplotlib import pyplot as plt
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

#Definition of a classical lee speckle filter
#not used in this particular network, but meant
#to potentially be used to transform the input SAR images
def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size, 1))
    img_sqr_mean = uniform_filter(img**2, (size, size, 1))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

dataset = np.load("D:/dataset-resized.npy", allow_pickle=True)

MSE_test_values = []
RMSE_test_values = []
dataset2 = []

#Filtering out the mostly cloudy/nighttime images
for i in dataset:
    if np.sum(abs(i["y"])) > 1200:
        dataset2.append(i)

dataset = dataset2
#K-fold cross validation
for i in range(1):
    dataset = shuffle(dataset)
    #Taking only the two SAR channels
    sar = [entry["sarImage"][:, :, 0:2] for entry in dataset]

    height = [s.shape[0] for s in sar]
    width = [s.shape[1] for s in sar]

    maxWidth = max(width)
    maxHeight = max(height)
    #Adding a decibel-like transformation to input to improve accuracy
    sar = np.array([np.log10(np.pad(s+1, [(0, maxHeight - s.shape[0]), (0, maxWidth - s.shape[1]), (0, 0)])) for s in sar])
    ndvi = np.array([entry["y"] for entry in dataset])
    lastNDVI = np.array([entry["sarImage"][:, :, 2] for entry in dataset])
    ids = np.array([entry["id"] for entry in dataset])
    lastNDVIDays = np.array([entry["lastNDVITakenBefore"] for entry in dataset])

    sarShape = sar[0].shape
    lastNDVIShape = lastNDVI[0].shape
    ndviShape = ndvi[0].shape
    #ndviReshaped = ndvi.reshape([ndvi.shape[0], ndvi.shape[1], ndvi.shape[2]])
    ndviReshaped = ndvi.reshape([ndvi.shape[0], ndvi.shape[1]* ndvi.shape[2]])

    flatLastNDVI = np.reshape(lastNDVI, [lastNDVI.shape[0], lastNDVIShape[0] * lastNDVIShape[1]])

    lastNDVIDays = np.reshape(lastNDVIDays, [lastNDVIDays.shape[0], 1])

    flatLastNDVI = np.hstack((flatLastNDVI, lastNDVIDays))

    trainID, testID, trainX, testX, trainLastNDVI, testLastNDVI, trainY, testY = train_test_split(ids, sar,
                                                                                                  flatLastNDVI,
                                                                                                  ndviReshaped,
                                                                                                  test_size=0.20,
                                                                                                  random_state=41)
    inputSar = Input(shape=sarShape)
    #img = BatchNormalization()(inputSar)
    img = inputSar
    #img = MaxPool2D(pool_size=(2, 2), strides=2)(img)
    img = Conv2D(filters=512, kernel_size=(5, 5), activation='relu', padding='same')(img)
    img = AveragePooling2D(pool_size=(2, 2), strides=2)(img)
    #img = Dropout(0.2)(img)
    #img = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(img)
    #img = MaxPool2D(pool_size=(2, 2), strides=2)(img)
    '''img = Dropout(0.2)(img)
    img = Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same')(img)
    img = MaxPool2D(pool_size=(2, 2), strides=2)(img)
    img = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(img)
    img = Dropout(0.2)(img)
    #img = AveragePooling2D(pool_size=(2, 2), strides=2)(img)
    #img = Conv2D(filters=128, kernel_size=(3, 3), activation='sigmoid', padding='same')(img)
    img = Conv2D(filters=512, kernel_size=(5, 5), activation='relu', padding='same')(img)

    #img = Conv2D(filters=1024, kernel_size=(3, 3), activation='sigmoid', padding='same')(img)
    #img = Conv2D(filters=1024, kernel_size=(3, 3), activation='sigmoid', padding='same')(img)
    #img = Conv2D(filters=1024, kernel_size=(3, 3), activation='sigmoid', padding='same')(img)
    # img = MaxPool2D(pool_size=(2, 2), strides=2)(img)
    #img = Dropout(0.2)(img)
    #img = Conv2D(filters=2048, kernel_size=(3, 3), activation='sigmoid', padding='same')(img)

    img = MaxPool2D(pool_size=(2, 2), strides=2)(img)
    #img = Dropout(0.2)(img)
    #img = Dropout(0.2)(img)'''
    img = Flatten()(img)
    #img = Model(inputs=inputSar, outputs=img)
    model = img
    model = Dense(256, activation="relu")(model)
    model = Dense(ndviReshaped.shape[1])(model)
    model = Model (inputs = inputSar, outputs = model)

    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.01), loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    history = model.fit(
        x = trainX,
        y=trainY,
        epochs=35,
        verbose='auto',
        validation_split=0.2)

    score = model.evaluate(testX, testY)
    MSE_test_values.append(score[0])
    RMSE_test_values.append(score[1])


    prevPred = np.zeros ((114, 71))
    oldNDVI = np.zeros ((114, 71))
    for i in range(2, 30):
        sarLayer1 = sar[i][:, :, 1]
        sarLayer2 = sar[i][:, :, 0]
        prediction = model.predict(sar[i:(i + 1)]).flatten()
        prevPred = np.resize(model.predict(sar[(i-1):i]).flatten(), (114,71))

        print ("difference")
        print (np.sum (abs(prevPred - np.resize(prediction, (114, 71)))))
        print ("previous pred")
        print(np.sum(abs(prevPred)))
        print ("current pred")
        print(np.sum(abs(prediction)))
        prevPred = np.resize(prediction, (114, 71))

        os.chdir("D:\PredictionVSReality3")
        plt.imshow(np.resize(prediction, (114, 71)), vmin=0, vmax=1)
        plt.savefig(ids[i - 1] + "Prediction.png")
        plt.close()
        plt.matshow(np.resize(ndviReshaped[i:(i + 1)], (114, 71)), vmin=0, vmax=1)
        plt.savefig(ids[i - 1] + "Reality.png")
        plt.close()
        #plt.matshow(sarLayer1, vmin=np.min(sarLayer1), vmax=np.max(sarLayer1))
        #plt.savefig(ids[i - 1] + "SAR1eality.png")
        #plt.close()
        #plt.matshow(sarLayer2, vmin=np.min(sarLayer2), vmax=np.max(sarLayer2))
        #plt.savefig(ids[i - 1] + "SAR2eality.png")
        #plt.close()

        print (np.sum (abs(oldNDVI - np.resize(ndviReshaped[i:(i + 1)], (114, 71)))))
        oldNDVI = np.resize(ndviReshaped[i:(i + 1)], (114, 71))
        plt.matshow((np.resize(ndviReshaped[i:(i + 1)], (114, 71)) - np.resize(prediction, (114, 71))) ** 2, vmin=0,vmax=1)
        plt.savefig(ids[i - 1] + "SquaredDifference.png")
        plt.close()
    os.chdir("D:")

print(np.average(MSE_test_values))
print(np.average(RMSE_test_values))