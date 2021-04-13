#!/usr/bin/env python3

# Create features from the audio files

import librosa 
import librosa.display 
import pdb
from tqdm import tqdm
import argparse

import os
import sys
import h5py

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import preprocessing

# Suppress librosa warning due to mp3 format
import warnings
warnings.filterwarnings('ignore')

import utils

SAMPLE_RATE = 22050

def create_features(seg):
    '''Create features from an audio segment'''
    mfcc = librosa.feature.mfcc(seg, n_mfcc=40)
    mfcc_mean          = list(np.mean(mfcc, axis=1))
    rms                = list(np.mean(librosa.feature.rms(seg), axis=1))
    zcr                = list(np.mean(librosa.feature.zero_crossing_rate(seg), axis=1))
    spectral_centroid  = list(np.mean(librosa.feature.spectral_centroid(seg), axis=1))
    spectral_bandwidth = list(np.mean(librosa.feature.spectral_bandwidth(seg), axis=1))
    # spectral_contrast  = list(np.mean(librosa.feature.spectral_contrast(seg), axis=1))
    spectral_flatness  = list(np.mean(librosa.feature.spectral_flatness(seg), axis=1))
    spectral_rolloff   = list(np.mean(librosa.feature.spectral_rolloff(seg), axis=1))
    feat = mfcc_mean + rms + zcr + spectral_centroid + spectral_bandwidth + spectral_flatness + spectral_rolloff
    # print(feat)
    # print(len(feat))
    return feat

def test_create_features():
    prog, nonprog = utils.get_dataset('data/train_22050hz')
    song = prog[0]
    print(f'Using {song} as a test sample')
    y, sr = librosa.load(song)
    assert sr == SAMPLE_RATE, 'Sample rate is not 22050Hz'
    seg = y[10 * SAMPLE_RATE: 15 * SAMPLE_RATE]
    feat = create_features(seg)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='input path to the dataset')
    parser.add_argument('output', help='output filename')
    parser.add_argument('--limit', type=int, help='limit the number of files processed (for testing)')
    return parser.parse_args()

# test_create_features()
# sys.exit(0)
if __name__ == '__main__':
    args = get_arguments()
    prog, nonprog = utils.get_dataset(args.dataset)
    if args.limit:
        print(f'Process the first {args.limit} files only')
        prog = prog[:args.limit]
        nonprog = nonprog[:args.limit]

    # data = [[target, filename, part, features], ...]
    data = []
    for target, files in zip([1, 0], [prog, nonprog]):
        for fname in tqdm(files):
            raw, sr = librosa.load(fname)
            assert sr == SAMPLE_RATE, 'Invalid sample rate'
            # Segment duration = 4 seconds
            # Skip = 2 seconds
            part = 0
            for start in np.arange(10 * SAMPLE_RATE, len(raw) - 10 * SAMPLE_RATE, 4 * SAMPLE_RATE):
                end = start + 4 * SAMPLE_RATE
                seg = raw[start: end]
                feat = create_features(seg)
                data.append([target, os.path.basename(fname), part] +feat)
                part += 1

    features = [f'mfcc_{i}' for i in range(40)] + \
        ['rms', 'zcr', 'sp_centroid', 'sp_bandwidth', 'sp_flatness', 'sp_rolloff']
    df = pd.DataFrame(data, columns=['target', 'filename', 'part'] + features)

    print('Normalizing data...')
    scaler = preprocessing.Normalizer()
    df.iloc[:, 3:] = scaler.fit_transform(df.iloc[:, 3:])

    df.to_csv(args.output, index=False)
    print(f'Shape of the data: {df.shape}')
    print(f'Fraction of prog segments: {df.target.mean()}')


# test_create_features()
