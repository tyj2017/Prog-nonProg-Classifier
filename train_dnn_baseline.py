#!/usr/bin/env python3
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
import pandas as pd
import pdb
import os
import sys
import time
import datetime

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
model.add(keras.layers.Dense(32, activation='relu'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(32, activation='relu'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1, activation='sigmoid'))
print(model.summary())

# input("Press enter to start")

# Save model every n epochs
model_name = 'dnn.h64h32'
output_path = 'dnn'

def plot_conf_mat(mat):
    plt.imshow(mat, cmap='Reds')
    plt.gca().xaxis.set_ticks_position('top')
    # plt.gca().xaxis.tick_top()
    
    plt.xticks([0, 1], ['Non-prog', 'Prog'])
    plt.yticks([0, 1], ['Non-prog', 'Prog'])
    for row in range(len(mat)):
        for col in range(len(mat[0])):
            plt.text(col, row, mat[row][col], ha='center', va='center')
    plt.colorbar()

class CustomSaver(keras.callbacks.Callback):
    def __init__(self, freq=50):
        super().__init__()
        self.freq = freq
    
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.freq == 0:
            self.model.save(f'{output_path}/{model_name}.epoch{epoch + 1}.h5')

            # Also plot the confusion matrix
            fig = plt.figure(figsize=(12, 6)) 
            plt.subplot(121)
            y_pred = (self.model.predict(x_train) > 0.5).astype('int32')
            mat = confusion_matrix(y_train, y_pred, normalize=None, labels=[0, 1])
            plot_conf_mat(mat)
            plt.title('Confusion matrix for training set')

            plt.subplot(122)
            y_pred = (self.model.predict(x_test) > 0.5).astype('int32')
            mat = confusion_matrix(y_test, y_pred, normalize=None, labels=[0, 1])
            plot_conf_mat(mat)
            plt.title('Confusion matrix for validation set')
            # plt.show()
            fig.savefig(f'{output_path}/{model_name}.epoch{epoch + 1}.mat.png')
        
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['acc'])

time_start = time.time()
hist = model.fit(x_train, y_train, batch_size=32, epochs=1000, 
    validation_data=(x_test, y_test), 
    callbacks=[CustomSaver(freq=50)])
duration = time.time() - time_start
print(f'The training took {datetime.datetime.fromtimestamp(duration).strftime("%M:%S")}')

# Plot training history
fig = plt.figure(figsize=(12, 6)) 
plt.subplot(121)
plt.plot(hist.history['acc'], label='acc')
plt.plot(hist.history['val_acc'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(122)
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{output_path}/{model_name}.hist.png')
