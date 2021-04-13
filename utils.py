import glob
import h5py
import numpy as np 
import os

def get_dataset(path):
    '''Returns a pair of (prog, nonprog) where
    `prog` is a list of prog songs and `nonprog` is a 
    list of nonprog songs.'''
    res = []
    for song_type in ['prog', 'nonprog']:
        input_path = os.path.join(path, song_type, '*')
        input_files = glob.glob(input_path)
        res.append(input_files)
        print(f'{input_path} --> {len(input_files)} samples')
    return res
