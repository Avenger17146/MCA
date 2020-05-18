import pandas as pd 
import numpy as np 
import tensorflow as tf 
import nltk 
import sklearn
from nltk.corpus import abc
import re   
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import matplotlib.cm as cm

#refernce : https://towardsdatascience.com/word2vec-skip-gram-model-part-2-implementation-in-tf-7efdf6f58a27
def get_batches(words, batch_size, window_size=5, c = 5):
    n_batches = int(np.shape(words)[0]/batch_size)
    words = words[:n_batches*batch_size]
    for idx in range(0, len(words), batch_size):
        x =[]
        y = []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            R = np.random.randint(1, window_size+1)
            if (idx - R) > 0 :
                start = idx - R 
            else: 
                start =  0
            stop = idx + R
            target_words = set(words[start:idx] + words[idx+1:stop+1])
            batch_y = target_words
            x.extend([batch_x]*len(batch_y))
            y.extend(batch_y)
        yield x, y

#refernce : https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d
def tsne_plot(label, embedding):
  print('Plotting...')
  plt.figure(figsize=(16, 9))
  colors = cm.rainbow(np.linspace(0, 1, 1))
  plt.legend(loc=4)
  x = embedding[:,0]
  y = embedding[:,1]
  plt.scatter(x, y, c=colors, alpha=0.2, label=label)
  plt.savefig(label+'.png')
#   plt.show()    


t = 1e-5
x1  = abc.raw()
x1 = re.findall(r"[\w']+", x1) 
vocab_to_int = dict()
int_to_vocab = dict()

x2 = set(x1)
x2 = list(x2)
for i in range(len(x2)):
    vocab_to_int[x2[i]] = i
    int_to_vocab[i] = x2[i]

# vocab_to_int, int_to_vocab = utils.create_lookup_tables(x1)
int_words = [vocab_to_int[word] for word in x1]

y = dict()

for i in int_words:
    if i in y:
        y[i] += 1
    else :
        y[i] = 1

X = []
# int_words = set(int_words)
for i in x1[:10000]:
    if 1 - np.sqrt(t/y[vocab_to_int[i]]) > random.random() and y[vocab_to_int[i]] > 5: 
        X.append(vocab_to_int[i])

n_vocab = len(X)
print(n_vocab)
n_embedding =  100
n_sampled = 100

train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name='inputs')
    labels = tf.placeholder(tf.int32, [None, None], name='labels')
    embedding = tf.Variable(tf.random_normal((n_vocab, n_embedding), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)
    softmax_b = tf.Variable(tf.zeros(n_vocab), name="softmax_bias")
    softmax_w = tf.Variable(tf.random_normal((n_vocab, n_embedding))) 
    hiddden = tf.matmul(inputs,softmax_w)
    loss = tf.nn.sampled_softmax_loss(inputs=embed,weights=softmax_w,biases=softmax_b,num_sampled=100,num_classes=n_vocab, labels=labels)
    cost = tf.reduce_mean(loss)
    # cost = tf.reduce_mean(tf.losses.mean_squared_error(_y,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()

batch_size = 32
window_size  =5 
epochs = 20

with tf.Session(graph=train_graph)as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for ep in range(epochs):
        batches = get_batches(X, batch_size, window_size, c =5)
        for x, y in batches:
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed) 
            loss += train_loss
            if ( iteration % 20 == 0):
                print(ep , iteration)
                print(loss)
            iteration+=1
            loss = 0
        save_path = saver.save(sess, "here.model")
        # embed_mat = sess.run(normalized_embedding)
        print('plotting')
        e = sess.run(embedding)
        # viz_words = 2500
        # tsne = TSNE()
        # embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])
        # fig, ax = plt.subplots(figsize=(14, 14))
        # for idx in range(viz_words):
        #     plt.scatter(*embed_tsne[idx, :], color='steelblue')
        #     # plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
        # plt.show()
        # plt.savefig(str(epochs) +'.png')
        tsne= TSNE(verbose=1)
        # e = e.detach().cpu().numpy()
        embeddings_= tsne.fit_transform(e)
        tsne_plot('Epoch #' + str(ep), embeddings_)