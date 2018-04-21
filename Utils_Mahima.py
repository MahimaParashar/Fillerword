import os
import librosa
import scipy
import numpy as np
from tqdm import tqdm
import pickle as pkl
import speechpy
import csv
import random
import keras
import pickle as pkl
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
# from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,make_scorer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def get_normalized_mfcc(mfcc):
    mean = np.mean(mfcc)
    std = np.std(mfcc)
    mfcc_normalized = (mfcc - mean) / std
    return mfcc_normalized


def get_balanced_data(data_dir, pkl_dump_name):
    features = np.zeros((16308, 46))  # feature vector length
    labels = np.zeros(16308)
    ind = 0
    file_names = np.empty((16308), dtype='object')

    for file in tqdm(os.listdir(data_dir)):
        if file.endswith(".wav"):
            data_file_name = file[:-4]
            with open(os.path.join(data_dir, data_file_name + "_label.txt")) as l:
                label = l.read()
            if label:
                labels[ind] = 1  # discrepancy
            mfccs, tonnetz = extract_feature(os.path.join(data_dir, file))
            mfccs_normalized = get_normalized_mfcc(mfccs)
            file_names[ind] = data_file_name
            features[ind] = np.hstack([mfccs_normalized, tonnetz])
            ind += 1

    with open('./data/' + pkl_dump_name + '.pkl', 'wb') as f:
        pkl.dump((file_names, features, labels), f, protocol=2)
    return features, labels


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    #     chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    #     mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    #     contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    #     zcr = np.mean(librosa.feature.zero_crossing_rate(y = X).T, axis=1)
    return mfccs, tonnetz


get_balanced_data("./Augmented_Data",'umm_balanced_Mahima_augdata')