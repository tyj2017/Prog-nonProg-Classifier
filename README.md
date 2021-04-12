# Prog-nonProg-Classifier
the hemispheres

## Set up the environment

    $ conda create --name rock --file requirements.txt
    $ conda activate rock

You also need ffmpeg to process the audio files.

## Preprocessing

First we need to convert all audio files to the same sample rate 22050Hz.
The initial folder structure:

```
data
├── test_raw
│   ├── Not_Progressive_Rock
│   ├── Other
│   └── Progressive Rock Songs
```
Then run the convesion script

    $ ./fix_sample_rate data/test_raw data/test_22050hz

The resulting folder structure: 
```
data
├── test_22050hz
│   ├── Not_Progressive_Rock
│   ├── Other
│   └── Progressive Rock Songs
├── test_raw
│   ├── Not_Progressive_Rock
│   ├── Other
│   └── Progressive Rock Songs
```