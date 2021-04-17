import librosa 
import librosa.display 
import pdb
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import utils
import warnings
import h5py

warnings.filterwarnings('ignore')

SAMPLE_RATE = 22050
HOP_LENGTH = 1024

def create_features(seg):
    mfcc = librosa.feature.melspectrogram(seg, hop_length=HOP_LENGTH)
    feat = np.log(mfcc ** 2)
    feat = mfcc
    return feat

def create_training_set(prog, nonprog):
    y_train = []
    x_train = []
    for target, files in zip([1, 0], [prog, nonprog]):
        for f in tqdm(files):
            raw, sr = librosa.load(f)
            if sr != SAMPLE_RATE:
                raise RuntimeError(f)
            # Remove first and last 10 seconds
            raw = raw[10*SAMPLE_RATE: -10*SAMPLE_RATE]
            # Each segment is 10 seconds long
            seg_size = 10 * SAMPLE_RATE
            for seg_no in range(len(raw) // seg_size):
                if seg_no > 0:
                    pass
                seg = raw[seg_size * seg_no: seg_size * (seg_no + 1)]
                feat = create_features(seg)
                x_train.append(feat)
                y_train.append(target)
    x_train = np.array(x_train)
    y_train = np.array(y_train, dtype='int')
    return (x_train, y_train)

def test_create_features():
    create_features('data/train/nonprog/01.ArmenMiran-PreciousStory.mp3', 0)

    
prog, nonprog = utils.get_dataset()

prog = prog[:20]
nonprog = nonprog[:40]

x_train, y_train = create_training_set(prog, nonprog)
print(f'x_train shape = {x_train.shape}')
print(f'y_train shape = {y_train.shape}')
print(f'fraction of prog = {y_train.mean()}')

outfile = 'feat-f.h5'
with h5py.File(outfile, 'w') as h5:
    h5.create_dataset('x_train', data=x_train)
    h5.create_dataset('y_train', data=y_train)
