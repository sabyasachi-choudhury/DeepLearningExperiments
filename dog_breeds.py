# import json
import pickle
# import time
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers, models, losses
import numpy as np
import cv2
import os
# import pandas as pd

IM_SHAPE = [100, 100]
PREP = False
TRAINING = False
TESTING = False
NEW_DS = False

if PREP:
    files = os.listdir("train_breeds")
    labels = pd.read_csv("labels.csv").to_numpy()
    labels = {labels[i][0]: labels[i][1] for i in range(len(labels))}
    to_save, to_save_labels = [], []
    for file in files:
        img = cv2.imread("train_breeds/" + file)
        img = cv2.resize(img, IM_SHAPE)
        to_save.append(img)
        to_save_labels.append(labels[file[:-4]])
    with open("train_breeds.pickle", "wb") as file:
        pickle.dump(np.array(to_save, dtype=np.uint8), file)
    with open("breed_labels.pickle", "wb") as file:
        pickle.dump(to_save_labels, file)

def model_1():
    return models.Sequential([
        layers.Conv2D(16, [5, 5], activation='relu', input_shape=[*IM_SHAPE, 3]),
        layers.Conv2D(16, [5, 5], activation='relu'),
        layers.MaxPool2D([2, 2]),
        layers.Dropout(0.2),

        layers.Conv2D(32, [4, 4], activation='relu'),
        layers.Conv2D(32, [4, 4], activation='relu'),
        layers.MaxPool2D([2, 2]),
        layers.Dropout(0.2),

        layers.Conv2D(64, [3, 3], activation='relu'),
        layers.Conv2D(64, [3, 3], activation='relu'),
        layers.MaxPool2D([2, 2]),
        layers.Dropout(0.2),

        layers.Flatten(),
        # layers.Dense(300, activation='relu'),
        layers.Dense(120, activation='softmax')
    ])

if TRAINING:
    model = model_1()
    model.summary()
    model_name = "breed_model_2"
    model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    rec = []
    with open("train_breed_x.pickle", "rb") as f1, open("train_breed_y.pickle", "rb") as f2:
        train_x, train_y = pickle.load(f1), pickle.load(f2)
        for i, label in enumerate(train_y):
            if label in rec:
                train_y[i] = rec.index(label)
            else:
                rec.append(label)
                train_y[i] = len(rec) - 1

    with open("test_breed_x.pickle", "rb") as f1, open("test_breed_y.pickle", "rb") as f2:
        test_x, test_y = pickle.load(f1), pickle.load(f2)
        for i, elem in enumerate(test_y):
            test_y[i] = rec.index(elem)

    f = lambda x: np.array(x)
    train_x, train_y, test_x, test_y = f(train_x), f(train_y), f(test_x), f(test_y)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    with tf.device('/GPU:0'):
        model.fit(train_x, train_y, validation_data=[test_x, test_y], epochs=50)
    model.save("models/" + model_name)

if TESTING:
    with open("train_breeds.pickle", "rb") as file:
        dx = pickle.load(file)
    with open("breed_labels.pickle", "rb") as file:
        dy = pickle.load(file)
        record = {}
        for i, elem in enumerate(dy):
            try:
                dy[i] = record[elem]
            except KeyError:
                record[elem] = len(record)
                dy[i] = record[elem]

    corrects = 0
    predictions = models.load_model("models/breed_model_1").predict(dx)
    predictions = [np.argmax(pred) for pred in predictions]
    for i, elem in enumerate(predictions):
        if elem == dy[i]:
            corrects += 1
    print(corrects/len(predictions) * 100)

if NEW_DS:
    with open("train_breeds.pickle", "rb") as file:
        dx = pickle.load(file)
    with open("breed_labels.pickle", "rb") as file:
        dy = pickle.load(file)
    record = {}
    test_x, test_y = [], []
    train_x, train_y = [], []
    for i, label in enumerate(dy):
        try:
            if record[label] < 5:
                record[label] += 1
                test_y.append(label)
                test_x.append(dx[i])
            else:
                train_x.append(dx[i])
                train_y.append(label)
        except KeyError:
            record[label] = 1
            test_y.append(label)
            test_x.append(dx[i])
    with open("test_breed_x.pickle", "wb") as f1, open("test_breed_y.pickle", "wb") as f2:
        pickle.dump(test_x, f1)
        pickle.dump(test_y, f2)
    with open("train_breed_x.pickle", "wb") as f1, open("train_breed_y.pickle", "wb") as f2:
        pickle.dump(train_x, f1)
        pickle.dump(train_y, f2)


image = cv2.imread("train_breeds/0cdd66f35d9b7d8b0a98a4d506396c0d.jpg")

image = cv2.Canny(image, 50, 210)

cv2.imshow("1", image)
cv2.waitKey(0)