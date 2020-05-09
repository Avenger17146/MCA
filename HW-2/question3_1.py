import wave 
import numpy as np 
import pandas as pd 
import os
import librosa as lb
import random 
import tqdm
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from librosa.core import stft
from sklearn.svm import LinearSVC 
from sklearn import decomposition
import pickle
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


def spectro() :
    x = np.load('spectro.npy')
    x  = np.real(x)
    test_x = np.load('spectro_test.npy')
    test_x = np.real(test_x)
    y = np.load('label.npy')
    y = np.real(y)
    test_y = np.load('label_test.npy')
    test_y = np.real(test_y)

    # print(np.shape(x))
    X  = normalize(x)
    test_x = normalize(test_x)
    # t = MinMaxScaler()
    # X  = t.fit_transform(x)
    # tq = MinMaxScaler()
    # test_X = tq.fit_transform(test_x)

    model = LinearSVC(verbose=1,C=0.5,max_iter=2000)
    model.fit(X,y)
    pickle.dump(model,open('model.pkl', 'wb'))
    y_pred  = model.predict(test_x)
    print(classification_report(test_y, y_pred))

def mfcc():
    x = np.load('mfcc.npy')
    test_x = np.load('mfcc_test.npy')
    y = np.load('mfcc_label.npy')
    test_y = np.load('mfcc_label_test.npy')

    # print(np.shape(x))
    # x  = normalize(x)
    # test_x = normalize(test_x)
    t = MinMaxScaler()
    X  = t.fit_transform(x)
    # tq = MinMaxScaler()
    # test_X = tq.fit_transform(test_x)
    # pca = decomposition.PCA(n_components=2200)
    # pca.fit(x)
    # x = pca.transform(x)
    # pca.fit(test_x)
    # test_x = pca.transform(test_x)
    # np.save('pca.npy',x)
    # np.save('pca_test.npy',test_x)

    model = LinearSVC(verbose=1,C=0.5,max_iter=2000)
    model.fit(X,y)
    # pickle.dump(model,open('mfcc_model.pkl', 'wb'))
    y_pred  = model.predict(test_x)
    print(classification_report(test_y, y_pred))

spectro()
# mfcc()