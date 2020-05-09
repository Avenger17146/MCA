import wave 
import numpy as np 
import pandas as pd 
import os
import librosa as lb
import random 
import tqdm
from sklearn.preprocessing import normalize
from librosa.core import stft
import matplotlib.pyplot as plt
import sys


np.set_printoptions(threshold=sys.maxsize)
path = './validation/'
window_length = 2048
hop_length = 1024
noise_coeff = 0.1
mati = []
label= []
pos = [8,5,4,9,1,7,6,3,2,0]
l = 0
noises = []

import random 
  
def plot(x):
    data = 20*np.log10(x + 0.0000001)
    plt.pcolormesh(data.T)
    plt.show()    

def dft(x):
    # refernce : https://towardsdatascience.com/fast-fourier-transform-937926e591cb
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def Stft(x):
    fft_size = len(x)
    ze = np.zeros(fft_size)
    window = np.hanning(fft_size)
    x = x*window
    x = np.append(x,ze)
    x = dft(x)/fft_size
    x  = np.abs(x*np.conj(x))
    return x[:fft_size]


def spectro(x):
    mat = []
    for i in x:
        v = Stft(i)
        mat.append(v)
    mat = np.array(mat)
    # plot(mat)
    mat = np.ndarray.flatten(mat)
    return mat  


z  = os.listdir('./_background_noise_')

for y in range(6):
    k, nr = lb.load('./_background_noise_/'+z[y])
    noises.append(k)

for i in os.listdir(path):
    print(i)
    dk = 0
    for j in os.listdir(path+i):
        label.append(pos[l])
        print(dk)
        dk+=1
        mat = []
        x, sr  = lb.load(path+i+'/'+j, sr=None) 
        y = random.randrange(6)
        p = noises[y]
        x+= noise_coeff*p[:np.shape(x)[0]]
        for k in range(0,np.shape(x)[0]-window_length+1,hop_length):
            if k+window_length < np.shape(x)[0] :
                mat.append(x[k:k+window_length])
        x = spectro(mat)
        # print(np.shape(x))
        # if ( np.shape(x)[0] > 10000):
        #     sub = np.shape(x)[0] - 10000
        #     sub = int(sub/2)
        #     x = x[sub:np.shape(x)[0]-sub]
        # if ( np.shape(x)[0] < 10000):
        #     sub = 10000- np.shape(x)[0]
        #     for h in range(sub):
        #         x  = np.append(x,0)
        # x  = stft(x,win_length=2048,hop_length=1024)
        x = lb.amplitude_to_db(x)
        # x  = np.abs(x*np.conj(x))
        # x = np.ndarray.flatten(x)
        # plot(x)
        mati.append(x)
    l+=1

length = max(map(len, mati))
length1 = min(map(len, mati))
print(length)
print(length1)
# length = 16400
# y=np.array([xi+[0]*(length-len(xi)) for xi in mat])
goo = []
for i in mati:
    for j in range(length-np.shape(i)[0]):
        i = np.append(i,0)
    goo.append(i)

goo = np.array(goo)

if path == './training/':
    np.save('spectro_noise.npy',goo)
    label= np.array(label)
    print('saved')
    np.save('label.npy',label)
elif path == './validation/':
    np.save('spectro_test_noise.npy',goo)
    label= np.array(label)
    np.save('label_test.npy',label)
    print('saved')

 
        

