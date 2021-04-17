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
import xgboost as xgb

df = pd.read_csv('train4.csv')
df = df.sample(frac=1.0)
print(f'prog fraction = {df.target.mean()}')

df_train, df_valid, _, _ = train_test_split(df, df.target, test_size=0.2)

print(f'df_train.shape = {df_train.shape}')
dtrain = xgb.DMatrix(df_train.iloc[:, 3:], df_train.target)
dvalid = xgb.DMatrix(df_valid.iloc[:, 3:], df_valid.target)

param = {'max_depth': 4, 'eta': 0.5, 'subsample':0.5, 'gamma':0.01,
    'objective': 'binary:logistic', 'eval_metric':['logloss', 'error']}
watchlist = [(dtrain, 'train'), (dvalid, 'val')]
num_round = 200
bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=10)
# hist = xgb.cv(param, dtrain, num_round, metrics=['logloss', 'error'])
# print(hist)
# pred_val = bst.predict(dtest)

def plot_confusion_matrix(bst, df, figtitle, figpath):
    df = df.copy()
    x = df.iloc[:, 3:]
    y = df.target
    ds = xgb.DMatrix(x, y)
    df['pred'] = bst.predict(ds)
    # Average over the segments
    res = df.groupby(by=['filename', 'target'], as_index=False)['pred'].mean()
    res['pred_class'] = (res.pred > 0.5)
    res['pred_class'] = res['pred_class'].astype(int)

    fig = plt.figure(figsize=(6, 6)) 
    mat = confusion_matrix(res.target, res.pred_class, normalize=None, labels=[0, 1])
    plt.imshow(mat, cmap='Reds')
    plt.gca().xaxis.set_ticks_position('top')
    # plt.gca().xaxis.tick_top()
    
    plt.xticks([0, 1], ['Non-prog', 'Prog'])
    plt.yticks([0, 1], ['Non-prog', 'Prog'])
    for row in range(len(mat)):
        for col in range(len(mat[0])):
            plt.text(col, row, mat[row][col], ha='center', va='center')
    plt.colorbar()
    plt.title(figtitle)
    fig.savefig(figpath)

    print(f'overall acc = {np.mean(res.pred_class == res.target)}')
    print(f'nonprog acc = {mat[0][0] / sum(mat[0])}')
    print(f'   prog acc = {mat[1][1] / sum(mat[1])}')

plot_confusion_matrix(bst, df_train, 'Confusion matrix for training set', 'xgboost/train.mat.pdf')
plot_confusion_matrix(bst, df_valid, 'Confusion matrix for validation set', 'xgboost/valid.mat.pdf')

df_test = pd.read_csv('test4.csv')
print(f'test shape: {df_test.shape}')
plot_confusion_matrix(bst, df_test, 'Confusion matrix for test set', 'xgboost/test.mat.pdf')