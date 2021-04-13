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

df = pd.read_csv('test.csv')
print(f'prog fraction = {df.target.mean()}')

model = keras.models.load_model('dnn/dnn.h64h32.epoch1000.h5')
df['pred'] = model.predict(df.iloc[:, 3:])
res = df.groupby(by=['filename', 'target'], as_index=False)['pred'].mean()
res['pred_class'] = (res.pred > 0.5)
res['pred_class'] = res['pred_class'].astype(int)
# print(res.head())
print(f'acc = {np.mean(res.pred_class == res.target)}')