import random

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, losses
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data_x = pd.read_csv("mental_health_ds/train.csv", usecols=[
    "Designation", "Resource Allocation",
    "Mental Fatigue Score", "Burn Rate"]).dropna()
data_y = data_x["Burn Rate"]
data_x = data_x.drop(["Burn Rate"], axis=1)


def counts(df, row):
    table = {}
    for elem in df[row]:
        if not pd.isna(elem):
            if elem in table:
                table[elem] += 1
            else:
                table[elem] = 1
    return table


data_y = data_y.to_numpy(dtype=np.float16).flatten()

# data_x["Gender"] = np.where(data_x["Gender"] == "Male", 0, 1)
# data_x["Company Type"] = np.where(data_x["Company Type"] == "Service", 0, 1)
# data_x["WFH Setup Available"] = np.where(data_x["WFH Setup Available"] == "No", 0, 1)
# data_y = np.where(data_x["Mental Fatigue Score"] == 4.48, random.randint(21, 41) / 100,
#                   data_y)
# data_y = np.where(data_x["Resource Allocation"] == 4.48, random.randint(17, 72) / 100,
#                   data_y)
#
data_x = data_x.to_numpy(dtype=np.float16)
#
print(data_x.shape, data_x)
print(data_y.shape, data_y)

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1)

model = models.Sequential([
    layers.Dense(10, activation='relu', input_shape=[3]),
    layers.Dense(20, activation='relu'),
    layers.Dense(1)
])
model.summary()

model.compile(optimizer='adam', loss=losses.MeanAbsoluteError(), metrics=["accuracy"])
history = model.fit(train_x, train_y, epochs=10, validation_data=[test_x, test_y])
print(model.predict(data_x))
print(data_y)

# model = LinearRegression()
# model.fit(train_x, train_y)
# print(model.score(test_x, test_y))
# print(model.predict(test_x), test_y)
#
# n_data = pd.read_csv("mental_health_ds/test.csv", usecols=[
#     "Designation", "Resource Allocation",
#     "Mental Fatigue Score"]).dropna().to_numpy(dtype=np.float16)
# predictions = model.predict(n_data).flatten()
# n_data_ids = pd.read_csv("mental_health_ds/test.csv", usecols=["Employee ID"]).to_numpy().flatten()
# return_ds = pd.DataFrame({"Burn Rate": predictions}, index=n_data_ids)
# print(return_ds)
# return_ds.to_csv("mental_health_ds/submission.csv")