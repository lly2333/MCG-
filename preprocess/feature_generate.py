import warnings
import sklearn.metrics as skm
warnings.filterwarnings("ignore")
from scipy.fftpack import fft,ifft
import math
import os
import scipy.io as sio
from DE_PSD import *
seed = 20200220  # random seed to make results reproducible
# Set random seed to be able to reproduce results
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hanning, filtfilt
import tensorflow as tf

def compute_mulinfo(data, chans):
    adj_matrix = np.zeros((chans, chans))
    matrix = data
    for i in range(len(matrix)):
        mid = np.median(matrix[i])
        for j in range(len(matrix[0])):
            if matrix[i][j]>mid:
                matrix[i][j]=1
            else:
                matrix[i][j]=0
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            tmp = skm.mutual_info_score((matrix[i]), (matrix[j]))
            adj_matrix[i][j] = tmp
            adj_matrix[j][i] = tmp
    return adj_matrix

def features_extract(data):
    freq_start = [2, 5, 8,  11, 14, 17, 20, 23, 26, 29, 32]
    freq_end = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
    #freq_start = [2, 4, 8, 12, 16, 20, 24, 28, 32]
    #freq_end = [6, 10, 14, 18, 22, 26, 30, 34, 38]
    MYpsd = np.zeros([len(data), 22, len(freq_start)], dtype=float)
    MYde = np.zeros([len(data), 22, len(freq_start)], dtype=float)
    for i in range(0, len(data)):
        data1 =data[i][0]
        #print(data1.shape)
        MYpsd[i], MYde[i] = DE_PSD(data1, freq_start, freq_end)
    return MYpsd, MYde

def features_generate(train_set, valid_set, subject_id):
    train_data = []
    train_label = []
    valid_data = []
    valid_label = []
    for i in range(len(train_set)):
        train_data.append([train_set[i][0]])
        train_label.append(train_set[i][1])
    for i in range(len(valid_set)):
        valid_data.append([valid_set[i][0]])
        valid_label.append(valid_set[i][1])
    ss = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    train_data = np.array(train_data)
    valid_data = np.array(valid_data)
    for i in range(len(train_data)):
        train_data[i, 0, :, :] = ss.fit_transform(np.array(train_data[i, 0, :, :]))
        valid_data[i, 0, :, :] = ss.fit_transform(np.array(valid_data[i, 0, :, :]))
    _, train_feature = features_extract(np.array(train_data))
    train_feature = tf.constant(train_feature)
    train_label = tf.constant(np.array(train_label))
    # train_feature = np.expand_dims(train_feature, 1)
    _, valid_feature = features_extract(np.array(valid_data))
    valid_feature = tf.constant(valid_feature)
    valid_label = tf.constant(np.array(valid_label))
    all_data = np.concatenate([train_feature, valid_feature], axis=0)
    all_label = np.concatenate([train_label, valid_label], axis=0)
    de_train = np.zeros((288,22,22))
    de_valid = np.zeros((288,22,22))
    for i in range(len(train_feature)):
        de_train[i] = compute_mulinfo(train_feature[i], 22)
    for i in range(len(valid_feature)):
        de_train[i] = compute_mulinfo(valid_feature[i], 22)
    np.savez(r'.\\test\\trial_features_' + str(subject_id) + '.npz', all_data = all_data, all_label = all_label)
    np.savez(r'.\test\de_adj_'+str(subject_id)+'.npz',train_adj=de_train,valid_adj=de_valid)
    print('ok')


