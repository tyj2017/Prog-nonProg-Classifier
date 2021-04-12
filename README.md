# Prog-nonProg-Classifier
the hemispheres

## Set up the environment

    $ conda create --name rock --file requirements.txt
    $ conda activate rock

You also need ffmpeg to process the audio files.

## Preprocessing

First we need to convert all audio files to the same sample rate 22050Hz.
The folder structure is listed below. The `*_raw` folders contains the 
raw audio files of different formats and sample rates.
The `*_22050hz` folders are created by the `fix_sample_rate.py` script
automatically.

```
data
├── test_22050hz
│   ├── nonprog
│   └── prog
├── test_raw
│   ├── nonprog
│   └── prog
├── train_22050hz
│   ├── nonprog
│   └── prog
└── train_raw
    ├── nonprog
    └── prog
```
To convert all audio files in the test set to mp3 with sample
rate 22050Hz, run

    $ ./fix_sample_rate data/test_raw data/test_22050hz

By default, the script will use 4 worker processed to do the conversion.
