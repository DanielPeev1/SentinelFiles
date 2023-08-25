import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout,\
    BatchNormalization, Input, UpSampling2D, Concatenate, Cropping2D, AveragePooling2D, PReLU, GlobalAveragePooling2D, Layer, ConvLSTM2D
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Sequential
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# Loading the dataset and preparing the data
dataset = np.load("./dataset-lee-sigma-plus.npy", allow_pickle=True)
sar = [entry["sarImage"] for entry in dataset]

height = [s.shape[0] for s in sar]
width = [s.shape[1] for s in sar]

maxWidth = max(width)
maxHeight = max(height)

sar = np.array([np.pad(s, [(0, maxHeight - s.shape[0]),(0, maxWidth - s.shape[1]),(0,0)]) for s in sar])
ndvi = np.array([entry["y"] for entry in dataset])
ids = np.array([entry["id"] for entry in dataset])

sarShape = sar[0].shape
ndviShape = ndvi[0].shape

trainID, testID, trainX, testX, trainY, testY = train_test_split(ids, sar, ndvi,
                                                                 test_size=0.20, random_state=42)
print("Sar train: ", trainX.shape)
print("NDVI train: ", trainY.shape)

print("Sar test: ", testX.shape)
print("NDVI train: ", testY.shape)


input_shape = sar.shape

# Creating the model used for training
model = Sequential([
    # A different option for normalizing the data, although GroupNormalization gives better results
    #model.add(tf.keras.Input(shape = input_shape)),
    #layers.experimental.preprocessing.Rescaling(scale=1./255, input_shape=input_shape),
    layers.GroupNormalization(groups=-1),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.AveragePooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.AveragePooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.AveragePooling2D(),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(146*162, activation='linear')
])

model.build(input_shape)

print(model.summary())
trainY = trainY.reshape((88, 23652))
print(trainY.shape)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001, weight_decay=0.01),
              loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])


history = model.fit(
    x=trainX,
    y=trainY,
    epochs=30,
    batch_size=10,
    verbose='auto',
    validation_split=0.2,
)


testY = testY.reshape((23, 23652))
model.evaluate(testX, testY, batch_size=10)


for idx, entry in enumerate(dataset):
    sar = np.reshape(entry['sarImage'], (1, entry['sarImage'].shape[0], entry['sarImage'].shape[1],
                                         entry['sarImage'].shape[2]))[:,:,:,0:]

    prediction = model.predict(sar)
    prediction = prediction[0].reshape(146, 162)

    print(sar.shape)
    print(entry["y"].shape)
    print(prediction.shape)

    actual = entry["y"]


    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(actual, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("Actual " + entry['id'])
    axes[1].imshow(prediction, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("Predicted idx " + str(idx))
    plt.show()

    mse = mean_squared_error(actual, prediction)
    print("RMSE per entry: ", np.sqrt(mse))


