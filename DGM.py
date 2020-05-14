#!/usr/bin/env python
# coding: utf-8

# In[15]:


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

from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf
import numpy as np
import time
import sys

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[16]:


def Input_Data(Num):
    
    K = 100
    T = 1
    sigma = 0.4
    r = 0.5
    
    interior = []
    initial = []
    terminal = []
    terminal_value = []
    
    for i in range(0, Num):
        S = random.uniform(0,300)
        t = random.uniform(0,T)
        Bflag = random.choice([0,1])
        if (Bflag!=1.0):
            S_term = 99999999
            t_term = random.uniform(0,T)
            Term = S_term
        else:
            S_term = random.uniform(0,300)
            t_term = T
            Term = max(S_term - K, 0)
        S_int = random.uniform(0,300)

        interior.append([])
        initial.append([])
        terminal.append([])
        terminal_value.append([])

        interior[i].append(S)
        interior[i].append(t)
        terminal[i].append(S_term)
        terminal[i].append(t_term)
        initial[i].append(S_int)
        initial[i].append(0)
        terminal_value[i].append(Term)
    
    return tf.convert_to_tensor(interior), tf.convert_to_tensor(initial), tf.convert_to_tensor(terminal), tf.convert_to_tensor(terminal_value)


# In[17]:


headers=['Stock', 'Time', 'Term_Stock', 'S_Time', 'Int_Stock', 'Term_Value']
size = 10000
data_inter, data_intit, data_term, data_termV = Input_Data(size)


# In[18]:


input_data = tf.concat([data_inter, data_intit, data_term], 1)
with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(data_termV))


# In[ ]:


data = raw_data.copy()
data = shuffle(data)
data.tail()

train_data = data.sample(frac=0.8,random_state=100)
test_data = data.drop(train_data.index)

train_labels = train_data.pop("Term_Value")
test_labels = test_data.pop("Term_Value")

print(train_data.head())
print(test_data.head())


# In[6]:


def build_model():
    model = keras.Sequential([
    layers.Dense(100, activation=tf.nn.relu, input_shape=[2]),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dense(1)
  ])
    return model


# In[32]:


def customLoss(Input, Term):
    S = tf.slice(Input, [0, 0], [-1, 1])
    t = tf.slice(Input, [0, 1], [-1, 1])
    int_input = tf.slice(Input, [0, 2], [-1, 2])
    term_input = tf.slice(Input, [0, 4], [-1, 2])
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(S)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(S)
            tape.watch(t)
            inter_input = tf.concat([S, t], 1)
            pred_inter = model(inter_input)
        dC_dS = tape.gradient(pred_inter, S)
        dC_dt = tape.gradient(pred_inter, t)
    d2C_dS2 = tape2.gradient(dC_dS, S)
 
    #gradS = tf.gradients(pred_inter,SS)
    #gradS2 = tf.hessians(pred_inter,SS)
    
    L1 = dC_dt + 0.5 * 0.4**2 * S**2 * d2C_dS2 + 0.05 * S * dC_dS - 0.05 * pred_inter
    L2 = model(int_input)
    L3 = model(term_input) - Term
    return L1**2 + L2**2 + L3**2
#gradS = tf.gradients(pred,S)
#gradt = tf.gradients(pred,t)
#gradS2 = tf.hessians(pred,S)


# In[33]:


def step(Input, Term):
    with tf.GradientTape() as tape:
        Loss = tf.math.reduce_sum(customLoss(input_data, data_termV))
    
    grads = tape.gradient(Loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(Loss))


# In[34]:


model = build_model()
opt = Adam(lr=0.001)
model.summary()


# In[35]:


model.compile(loss= customLoss, optimizer=opt, metrics=['mae', 'mse'])


# In[38]:


for i in range(size):
    a = tf.slice(input_data, [i, 0], [1, 6])
    b = tf.slice(data_termV, [i, 0], [1, 1])
    step(a, b)


# In[ ]:




