#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import math

import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
import random
from scipy.stats import norm

import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf
import numpy as np
import time
import sys


# In[84]:


x = tf.convert_to_tensor(train_data.iloc[:,0])
print(x)

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
print(dz_dy)


# In[28]:


def input_Data(Num):
    
    K = 100
    T = 1
    
    output = []
    
    for i in range(0, Num):
        S = random.uniform(0,300)
        t = random.uniform(0,T)
        Bflag = random.choice([1,2])
        if (Bflag!=1):
            S_term = math.inf
            t_term = random.uniform(0,T)
            Term = S_term
        else:
            S_term = random.uniform(0,300)
            t_term = T
            Term = max(S_term - K, 0)
        S_int = random.uniform(0,300)

        output.append([])

        output[i].append(S)
        output[i].append(t)
        output[i].append(S_term)
        output[i].append(t_term)
        output[i].append(S_int)
        output[i].append(Term)
    
    return output


# In[52]:


headers=['Stock', 'Time', 'Term_Stock', 'S_Time', 'Int_Stock', 'Term_Value']
raw_data = pd.DataFrame(input_Data(10), columns=headers)
raw_data


# In[57]:


data = raw_data.copy()
data = shuffle(data)
data.tail()

train_data = data.sample(frac=0.8,random_state=100)
test_data = data.drop(train_data.index)

train_labels = train_data.pop("Term_Value")
test_labels = test_data.pop("Term_Value")

print(train_data.head())
print(test_data.head())


# In[105]:





# In[62]:


def build_model():
    model = keras.Sequential([
    layers.Dense(10, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
    layers.Dense(10, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.relu),
    layers.Dense(1)
  ])

    optimizer = tf.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss= customLoss, optimizer=optimizer, metrics=['mae', 'mse'])
    return model


# In[63]:


model = build_model()
model.summary()


# In[106]:


pred = model.predict(train_data)
pred


# In[103]:


def step(X, Y):
    Stock = tf.convert_to_tensor(X.iloc[:,0])
    Time = tf.convert_to_tensor(X.iloc[:,1])
    with tf.GradientTape() as t:
        t.watch(Stock)
        pred = model(X)
    print(type(pred))
    dC_dS = t.gradient(pred, Stock)

    return dC_dS


# In[104]:


step(train_data, train_labels)


# In[16]:


def customLoss(true,predicted):
    epsilon = 0.1
    summ = K.maximum(K.abs(true) + K.abs(predicted) + epsilon, 0.5 + epsilon)
    smape = K.abs(predicted - true) / summ * 2.0
    return smape


# In[2]:


def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


# In[ ]:




