#!/usr/bin/env python3
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
import pandas as pd
import pdb
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from tensorflow import keras

df = pd.read_csv('train3_balanced.csv')
df = df.sample(frac=1.0)
print(f'prog fraction = {df.target.mean()}')

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 3:], df.target, test_size=0.2)

model = keras.models.Sequential()
model.add(keras.layers.Input(shape=x_train.shape[1:]))
model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(50, activation='relu'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))
print(model.summary())

# input("Press enter to start")

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['acc'])
hist = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test))
model.save('dnn-v0.h5')

fig_hist = plt.figure()
plt.plot(hist.history['acc'], label='acc')
plt.plot(hist.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

y_pred = model.predict_classes(x_train)
# print(y_pred)
conf_mat = confusion_matrix(y_train, y_pred)

print(conf_mat)
# fig_pref = plt.figure()
# plt.matshow(conf_mat)
# plt.legend()
# plt.show()
