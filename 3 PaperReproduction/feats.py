import librosa
import numpy as np
from getlist import get_list
from python_speech_features import mfcc,delta

def get_mfcc(filename):
    x, fs = librosa.load(filename, sr=16000, mono=True)
    x, idx = librosa.effects.trim(x, 20)
    feat0 = mfcc(x, fs)
    feat1 = delta(feat0, 1)
    feat2 = delta(feat0, 2)
    return np.hstack((feat0, feat1, feat2))

if __name__ == "__main__":

    wavlist, train, test = get_list('new_data')
    feats = get_mfcc(wavlist[-1])
    print(feats.shape)
