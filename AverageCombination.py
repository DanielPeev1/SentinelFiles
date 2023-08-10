from sklearn.utils import shuffle
import os
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout, BatchNormalization, \
    Input, concatenate, PReLU, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import Model
from matplotlib import pyplot as plt

dataset = np.load("D:/dataset-resized.npy", allow_pickle=True)

MSE_test_values = []
RMSE_test_values = []
dataset2 = []

for i in dataset:
    if np.sum(abs(i["y"])) > 500:
        dataset2.append(i)

print(np.array(dataset2).shape)
dataset = dataset2
for i in range(1):
    dataset = shuffle(dataset)

    sar = [entry["sarImage"][:, :, 0:2] for entry in dataset]

    height = [s.shape[0] for s in sar]
    width = [s.shape[1] for s in sar]

    maxWidth = max(width)
    maxHeight = max(height)

    sar = np.array(
        [np.log10(np.pad(s + 1, [(0, maxHeight - s.shape[0]), (0, maxWidth - s.shape[1]), (0, 0)])) for s in sar])
    ndvi = np.array([entry["y"] for entry in dataset])
    lastNDVI = np.array([entry["sarImage"][:, :, 2:3] for entry in dataset])
    ids = np.array([entry["id"] for entry in dataset])
    # lastNDVIDays = np.array([entry["lastNDVITakenBefore"] for entry in dataset])
    print(lastNDVI.shape)

    sarShape = sar[0].shape
    lastNDVIShape = lastNDVI[0].shape
    ndviShape = ndvi[0].shape
    ndviReshaped = ndvi.reshape([ndvi.shape[0], ndvi.shape[1] * ndvi.shape[2]])

    # flatLastNDVI = np.reshape(lastNDVI, [lastNDVI.shape[0], lastNDVIShape[0] * lastNDVIShape[1]])
    # lastNDVIDays = np.reshape(lastNDVIDays, [lastNDVIDays.shape[0], 1])
    # flatLastNDVI = np.hstack((flatLastNDVI, lastNDVIDays))

    trainID, testID, trainX, testX, trainLastNDVI, testLastNDVI, trainY, testY = train_test_split(ids, sar, lastNDVI,
                                                                                                  ndviReshaped,
                                                                                                  test_size=0.20,
                                                                                                  random_state=41)

    print(sarShape)
    inputSar = Input(shape=sarShape)
    # img = BatchNormalization()(inputSar)
    img1 = inputSar
    img1 = Conv2D(filters=512, kernel_size=(5, 5), activation='relu', padding='same')(img1)
    img1 = AveragePooling2D(pool_size=(2, 2), strides=2)(img1)
    #img1 = Dropout(0.2)(img1)
    img1 = Flatten()(img1)
    model1 = img1
    model1 = Dense(512, activation="relu")(model1)
    model1 = Dense(ndviReshaped.shape[1])(model1)
    model1 = Model(inputs=inputSar, outputs=model1)


    #img = Model(inputs=inputSar, outputs=img)

    # print(img.summary())

    print("d")
    print(lastNDVIShape)

    inputLastNDVI = Input(shape=lastNDVIShape)
    img2 = inputLastNDVI
    img2 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(img2)
    img2 = AveragePooling2D(pool_size=(2, 2), strides=2)(img2)
    #img2 = Dropout(0.2)(img2)
    img2 = Flatten()(img2)

    model2 = img2
    model2 = Dense(512, activation="relu")(model2)
    model2 = Dense(ndviReshaped.shape[1])(model2)
    model2 = Model (inputs = inputLastNDVI, outputs = model2)

    combined = (0.4*model1.output + 0.6*model2.output)
    combined = Model (inputs = [model1.input, model2.input], outputs = combined)
    model = combined

    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.01), loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    history = model.fit(
        x=[trainX, trainLastNDVI],
        y=trainY,
        epochs=25,
        verbose='auto',
        validation_split=0.2)

    score = model.evaluate([testX, testLastNDVI], testY)
    MSE_test_values.append(score[0])
    RMSE_test_values.append(score[1])
    # print (100*'_')
    # print (str(np.sum(abs(ndviReshaped[:1]))) + "  " + ids [0])

    for i in range(2, 35):
        prediction = model.predict([sar[i:(i + 1)], lastNDVI[i:(i + 1)]]).flatten()
        prevPred = np.resize(model.predict([sar[(i - 1):i], lastNDVI[(i - 1) : i]]).flatten(), (114, 71))
        print("difference between successive predictions")
        print(np.sum(abs(prevPred - np.resize(prediction, (114, 71)))))
        plt.imshow(np.resize(prediction, (114, 71)), vmin=0, vmax=1)
        os.chdir("D:\PredictionVSReality2")
        plt.savefig(ids[i - 1] + "Prediction.png")
        plt.close()
        plt.matshow(np.resize(ndviReshaped[i:(i + 1)], (114, 71)), vmin=0, vmax=1)
        plt.savefig(ids[i - 1] + "Reality.png")
        plt.close()
        plt.matshow((np.resize(ndviReshaped[i:(i + 1)], (114, 71)) - np.resize(prediction, (114, 71))) ** 2, vmin=0,
                    vmax=1)
        plt.savefig(ids[i - 1] + "SquaredDifference.png")
        plt.close()
    os.chdir("D:")

    print(np.array2string(model.predict([sar[:1], lastNDVI[:1]])))
    print(np.array2string(ndviReshaped[:1]))
print(np.average(MSE_test_values))
print(np.average(RMSE_test_values))
