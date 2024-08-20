import random
import tensorflow as tf
from tensorflow import keras
from keras import models, layers, losses
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
import pandas as pd
import os
import pickle
import cv2
import time

SHAPE = (250, 250)
STORE_IMAGES = False
STORE_LABELS = False
TRAIN = False
PREDICT = False
RESHAPE_IMAGES = False
SVM = False

devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
print(devices)

if SVM:
    with open("train.pickle", "rb") as file:
        train_x = pickle.load(file).reshape(25000, 250*250)
    with open("train_labels.pickle", "rb") as file:
        train_y = pickle.load(file)
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    model = SVC()
    model.fit(train_x, train_y)
    print(model.score(test_x, test_y))
    with open("svm_model.pickle", "wb") as file:
        pickle.dump(model, file)

if RESHAPE_IMAGES:
    NEW_SHAPE = [100, 100]
    with open("train.pickle", "rb") as file:
        images = pickle.load(file)
    new_images = np.array([cv2.resize(elem, NEW_SHAPE) for elem in images])
    with open("train_2.pickle", "wb") as file:
        pickle.dump(new_images, file)

if STORE_IMAGES:
    files = os.listdir("test")
    data = []
    it = 0
    s = time.perf_counter()
    for file in files:
        img = cv2.cvtColor(cv2.imread("test/" + file), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, SHAPE)
        data.append(img)
        it += 1
        if it % 100 == 0:
            print(it, time.perf_counter() - s)
    data = np.array(data, dtype=np.uint8)
    with open("test.pickle", "wb") as file:
        pickle.dump(data, file)

if STORE_LABELS:
    files = os.listdir("train")
    data = [0 if file[:3] == "cat" else 1 for file in files]
    with open("train_labels.pickle", "wb") as file:
        pickle.dump(np.array(data, dtype=np.uint0), file)

def model_1(shape):
    return models.Sequential([
        layers.Conv2D(32, [3, 3], input_shape=[*shape, 1], activation='relu', padding='same'),
        layers.Conv2D(32, [3, 3], activation='relu', padding='same'),
        layers.Dropout(0.3),
        layers.MaxPool2D([2, 2]),
        layers.Conv2D(64, [3, 3], activation='relu', padding='same'),
        layers.Conv2D(64, [3, 3], activation='relu', padding='same'),
        layers.Dropout(0.3),
        layers.MaxPool2D([2, 2]),
        layers.Conv2D(64, [3, 3], activation='relu', padding='same'),
        layers.Conv2D(64, [3, 3], activation='relu', padding='same'),
        layers.Dropout(0.3),
        layers.MaxPool2D([2, 2]),
        layers.Flatten(),
        layers.Dense(20, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

model_path = "model_1"

if TRAIN:
    with open("train_2.pickle", "rb") as file:
        train_x = pickle.load(file)
    with open("train_labels.pickle", "rb") as file:
        train_y = pickle.load(file)
    print(train_x.shape, train_y.shape)
    # train_x, test_x, train_y, test_y = train_test_split(train_x[:len(train_x)//2], train_y[:len(train_y)//2], test_size=0.1)
    indices = np.random.permutation(len(train_x))[int(len(train_y)//2):]
    test_x, test_y = train_x[indices[int(0.9*indices.shape[0]):]], train_y[indices[int(0.9*indices.shape[0]):]]
    train_x, train_y = train_x[indices[:int(0.9*indices.shape[0])]], train_y[indices[:int(0.9*indices.shape[0])]]
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    with tf.device('/GPU:0'):
        # model = model_1([100, 100])
        model = models.load_model("models/" + model_path)
        model.summary()
        model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
        model.fit(train_x, train_y, validation_data=[test_x, test_y], epochs=15)
        model.save("models/model_2")

if PREDICT:
    with open("test.pickle", "rb") as file:
        data = pickle.load(file)
    with tf.device('/GPU:0'):
        model = models.load_model("models/" + model_path)
        predictions = model.predict(data).flatten()
        print(predictions)
        predictions = pd.DataFrame({"id": [elem + 1 for elem in range(len(predictions))],
                                    "label": [round(elem) for elem in predictions]})
        predictions.to_csv("predictions.csv")

model = models.load_model("models/model_1")
model.summary()
with open("train_2.pickle", "rb") as file:
    train_x = pickle.load(file)
with open("train_labels.pickle", "rb") as file:
    train_y = pickle.load(file)

# model.evaluate(train_x, train_y)

ind = 14550
predictions = model.predict(train_x[ind:ind+1])
print(round(predictions[0][0]))
cv2.imshow("f", train_x[ind])
cv2.waitKey(0)