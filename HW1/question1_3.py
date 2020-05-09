import cv2 
import numpy as np 
import os 
import pickle
from time import time
from scipy.ndimage import gaussian_laplace as LoG
import matplotlib.pyplot as plt

query_path = "./train/query"
image_path = "./images"
oct1 = [9,15,21,27]
oct2 = [15,27,39,51]
oct3 = [27,51,75,99]
oct4 = [51,99,147,195]

def non_max(x):
    pass

for i in os.listdir(image_path):
    img = cv2.imread(image_path+'/'+i)
    w = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = w/255
    oc1 = []
    prev = oct1[0]
    for j in oct1:
        oc1.append(cv2.GaussianBlur(w,(j,j),1.6*j/prev))
        prev = j
    oc2 = []
    prev = oct2[0]
    for j in oct2:
        oc2.append(cv2.GaussianBlur(w,(j,j),1.6*j/prev))
        prev = j
    oc3 = []
    prev = oct3[0]
    for j in oct3:
        oc3.append(cv2.GaussianBlur(w,(j,j),1.6*j/prev))
        prev = j
    oc4 = []
    prev = oct4[0]
    for j in oct1:
        oc4.append(cv2.GaussianBlur(w,(j,j),1.6*j/prev))
        prev = j
    key1 = non_max(oc1)
    key2 = non_max(oc2)
    key3 = non_max(oc3)
    key4 = non_max(oc4)

