import cv2 
import numpy as np 
import os 
import pickle
from time import time
from scipy.ndimage import gaussian_laplace as LoG
import matplotlib.pyplot as plt

query_path = "./train/query"
image_path = "./images"
k = 2**(1/2)
thresh = 0.03
blobs = []
show = True
dk = 0
train = False

if train : 
    for i in os.listdir(image_path):
        print(dk)
        dk+=1
        result =[]
        blobs  =[]
        img = cv2.imread(image_path+'/'+i)
        w = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w = w/255
        # x = np.array(img, dtype=np.float64)/255
        for j in range(1,11):
            sigma = (k**j)
            g = cv2.GaussianBlur(w,(3,3),sigma)
            result.append(np.square(cv2.Laplacian(g,cv2.CV_64F)))

        result = np.stack(result)
        for j in range(np.shape(w)[0]):
            for h in range(np.shape(w)[1]):
                arr = result[:, j:j+1, h:h+1]
                # print(np.argmax(arr))
                if  np.amax(arr) > thresh :
                    z,x,y = np.unravel_index(np.argmax(arr),arr.shape)
                    blobs.append((x+j-1,y+h-1,(k**z)))

            blobs  = list(set(blobs))
            pickle.dump(blobs,open('./log/'+i.split('.')[0]+'.p','wb'))
        
if show :
    for i in os.listdir('./log'):
        # i = 'all_souls_000001.p'
        blob = pickle.load(open('./log/'+i , 'rb'))
        if np.shape(blob)[0] > 2000 :
            blobs = []
            for j in range(0,np.shape(blob)[0],4):
                blobs.append(blob[j])
        else :
            blobs = blob
        img = cv2.imread(image_path+'/'+i.split('.')[0]+'.jpg')
        # img/= 255
        print(np.shape(blobs))
        fig, ax = plt.subplots()
        ax.imshow(img)
        for blob in blobs:
            ax.add_patch(plt.Circle((blob[1], blob[0]), blob[2]*1.414, color='red', linewidth=0.5, fill=False))
        ax.plot()  
        plt.show()	






    
    



