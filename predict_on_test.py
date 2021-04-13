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

df = pd.read_csv('test.csv')
print(f'prog fraction = {df.target.mean()}')

output_path = 'dnn'
model_name = 'dnn.h64h32'
epochs=600

model = keras.models.load_model(f'{output_path}/{model_name}.epoch{epochs}.h5')
df['pred'] = model.predict(df.iloc[:, 3:])
res = df.groupby(by=['filename', 'target'], as_index=False)['pred'].mean()
res['pred_class'] = (res.pred > 0.5)
res['pred_class'] = res['pred_class'].astype(int)

# Plot the confusion matrix
fig = plt.figure(figsize=(6, 6)) 
mat = confusion_matrix(res.target, res.pred_class, normalize=None, labels=[0, 1])
plot_conf_mat(mat)
plt.title('Confusion matrix for test set')
fig.savefig(f'{output_path}/{model_name}.test.mat.png')
plt.show()

print(f'overall acc = {np.mean(res.pred_class == res.target)}')
print(f'nonprog acc = {mat[0][0] / sum(mat[0])}')
print(f'   prog acc = {mat[1][1] / sum(mat[1])}')