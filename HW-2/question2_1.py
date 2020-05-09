import wave 
import numpy as np 
import pandas as pd 
import os
import librosa as lb
import random 
import tqdm
from sklearn.preprocessing import normalize
from librosa.core import stft
from scipy.signal import periodogram
from librosa.filters import mel
from scipy.fftpack import dct
import sys
import matplotlib.pyplot as plt
from librosa.feature import mfcc

np.set_printoptions(threshold=sys.maxsize)

label =[]
pos = [8,5,4,9,1,7,6,3,2,0]
l = 0
noises = []
window_length = 512
hop_length = 256
noise_coeff = 0.1
sr  = 0
path = './validation/'

#refernce : https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
def hz2mel(hz):
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

def plot_spectrogram(data):
    data = 20*np.log10(data + 0.0000001)
    plt.pcolormesh(data.T)
    plt.show()

def Mfcc(x,sr,window_length,hop_length):
    nfft = 512
    mat = []
    for k in range(0,np.shape(x)[0]-window_length,hop_length):
            if k+window_length < np.shape(x)[0] :
                mat.append(x[k:k+window_length])
    start = hz2mel(300)
    end = hz2mel(8000)
    filterbank=[]
    filterbank.append(start)
    for i in range(40):
        filterbank.append(start+(end-start)*(i/40))
    filterbank.append(end)
    filterbank = np.array(filterbank)
    filterbank = mel2hz(filterbank)
    filterbank = np.floor((nfft+1)*filterbank/sr)
    dk = np.zeros((40,257))
    for j in range(1,41):
        for k in range(1,258):
            if filterbank[j-1]<= k <= filterbank[j] :
                dk[j-1,k-1]+= (k-filterbank[j-1])/(filterbank[j]-filterbank[j-1])
            elif filterbank[j] <= k <= filterbank[j+1] :
                dk[j-1,k-1]+= (filterbank[j+1]-k)/(filterbank[j+1]-filterbank[j])
    # filterbank = mel(sr,nfft,n_mels=40,fmin=300,fmax=8000)
    filterbank = dk  
    energies = []
    for i in mat:
        coeff =[]
        f, px = periodogram(i)
        px = px[:257]
        for j in filterbank:
            y = np.sum(np.dot(px,j))
            coeff.append(y)
        energies.append(coeff)      
    energies = np.array(energies)
    energies = np.log10(energies) 
    energies = np.nan_to_num(energies) 
    energies = dct(energies,axis =1)
    MFcc = energies[:,2:14]
    # plot_spectrogram(np.transpose(Mfcc))
    MFcc = np.ndarray.flatten(MFcc)
    return MFcc

z  = os.listdir('./_background_noise_')

for y in range(6):
    k, nr = lb.load('./_background_noise_/'+z[y])
    noises.append(k)

mati = []
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
        x = Mfcc(x,sr,window_length,hop_length)
        # x  = mfcc(x,sr,n_mfcc=12)
        # x = np.ndarray.flatten(x)
        mati.append(x)
    l+=1


length = max(map(len, mati))
print(length)
goo = []
for i in mati:
    for j in range(length-np.shape(i)[0]):
        i = np.append(i,0)
    goo.append(i)

if path == './training/':
    goo = np.array(goo)
    np.save('mfcc_noise.npy',goo)
    label= np.array(label)
    np.save('mfcc_label_noise.npy',label)
elif path == './validation/':
    np.save('mfcc_test_noise.npy',goo)
    label= np.array(label)
    np.save('mfcc_label_test_noise.npy',label)

 