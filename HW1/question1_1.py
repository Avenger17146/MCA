import cv2 
import numpy as np 
import os 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
import pickle
from time import time
import pandas as pd
import PIL
from PIL import Image

query_path = "./train/query"
image_path = "./images"
match = True
train = False
truth = ['good','ok', 'junk']

n_colors = 64
d = [1]

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def quant(x):
    #refernce : https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
    # cv2.imshow('hel',img)
    x = np.array(img, dtype=np.float64)
    w, h, d = original_shape = tuple(x.shape)
    assert d == 3
    image_array = np.reshape(x, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    return recreate_image(kmeans.cluster_centers_, labels, w, h)

if train :   
    dk = 3800
    for t in os.listdir(image_path)[3800:4000]:
        print(dk)
        dk+=1
        img = cv2.imread(image_path+'/'+t)
        # img = cv2.imread('./images/all_souls_000047.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)
        img = img.quantize(64)
        img = np.array(img)
        # print(np.shape(img))
        ac = np.zeros((64,np.shape(d)[0]))
        count = 0
        color = np.zeros(64)
        for l in range(np.shape(img)[0]):
            for k in range(np.shape(img)[1]):
                color[img[l][k]]+=1
        for l in range(np.shape(img)[0]):
            for k in range(np.shape(img)[1]):
                c = img[l][k]
                for m in d:
                    for j in range(-m,m+1):
                        if (l+m < np.shape(img)[0]) :
                            if ( 0 <= k + j and k+j < np.shape(img)[1] ) :
                                if ( img[l+m][k+j] == c):
                                    ac[c][int(m/2)]+=1
                        if (l-m >= 0) :
                            if ( 0 <= k + j and k+j < np.shape(img)[1] ) :
                                if ( img[l-m][k+j] == c):
                                    ac[c][int(m/2)]+=1
                    for j in range(-m+1,m):
                        if (k+m < np.shape(img)[1]) :
                            if ( 0 <= l + j and l+j < np.shape(img)[0] ) :
                                if ( img[l+j][k+m] == c):
                                    ac[c][int(m/2)]+=1
                        if (k-m >= 0) :
                            if ( 0 <= l + j and l+j < np.shape(img)[0] ) :
                                if ( img[l+j][k-m] == c):
                                    ac[c][int(m/2)]+=1
                        
        for i in range(64):
            for j in range(1):
                ac[i][j]/= color[i]*8
        # print(ac)
        pickle.dump(ac,open('./cc/'+t.split('.')[0]+'.p','wb'))

def dist(cac1, cac2):
    if (cac1.shape != cac2.shape):
        print(cac1.shape)
        print(cac2.shape)
        print("Dimensions doen't match in the correlograms.")
        return
    size, dim =  cac1.shape

    score = 0.0
    for i in range(size):
        for j in range(dim):
            score += abs(cac1[i, j] - cac2[i, j])/(1 + cac1[i, j] + cac2[i, j])

    # for i in range(size):
    #         score += abs(cac1[i] - cac2[i])/(1 + cac1[i] + cac2[i])

    return score

no_q = 0
good  = 0
ok = 0
junk  = 0
if match :
    for i in os.listdir(query_path):
        t1 = time()
        no_q +=1
        names = [] 
        score =[]
        comp = []
        f = open(query_path+'/'+i)
        x = f.readline()
        print(x)
        x  = x.split()[0]
        x = x[5:]
        y = pickle.load(open('./cc/'+x +'.p', 'rb'))
        dk  = 0
        for j in os.listdir('./cc'):
            # print(dk)
            if ( j.split('.')[0] != x) :
                dk+=1
                z = pickle.load(open('./cc/'+j , 'rb'))
                # print(z)
                names.append(j.split('.')[0])
                score.append(dist(y,z))
        results = pd.DataFrame({'name': names ,'score':score})
        results = results.sort_values('score')
        print('time taken :')
        print(time()-t1)
        # print(results)
        for a in truth:
            comp = []
            path = i.split('query')[0]
            comp_f = open('./train/ground_truth/'+path+a+'.txt')
            for q in comp_f :
                comp.append(q.split('\n')[0])
            check = list(results['name'][:np.shape(comp)[0]])
            for q in comp :
                if q in check and a == 'good':
                    good+=1
                elif q in check and a == 'ok':
                    ok+=1
                elif q in check and a == 'junk':
                    junk+=1

            print(a)
            print('ground_truth')
            print(comp)
            print('Retrieved')
            print(list(results['name'][:np.shape(comp)[0]]))
            print('Scores of retrieved')
            print(results[:np.shape(comp)[0]])
            # print(np.shape(comp))
            p, r, f1, s = precision_recall_fscore_support(comp,list(results['name'][:np.shape(comp)[0]]),average='macro')
            print('precision '+ str(p))
            print('recall '+ str(r))
            print('f1 '+ str(f1))
        
    print('avg good retrieved : '+ str(good/no_q))
    print('avg ok retrieved : '+ str(ok/no_q))
    print('avg junk retrieved : '+ str(junk/no_q))

    
    

                         
    





