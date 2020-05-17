#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
import random
import matplotlib.pyplot as plt


# In[2]:


def initialize_parameters():
    initializer_X = tf.initializers.GlorotUniform()
    initializer_Z = tf.initializers.zeros()
    
    W1 = tf.Variable(initializer_X(shape = (100,2)))
    b1 = tf.Variable(initializer_Z(shape = (100,1)))
    W2 = tf.Variable(initializer_X(shape = (100,100)))
    b2 = tf.Variable(initializer_Z(shape = (100,1)))
    W3 = tf.Variable(initializer_X(shape = (100,100)))
    b3 = tf.Variable(initializer_Z(shape = (1,1)))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
                  
    
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = tf.add(tf.matmul(W1, X), b1) 
    A1 = tf.nn.sigmoid(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  
    A2 = tf.nn.sigmoid(Z2)                               
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3


# In[3]:


def gen():   
    K = 100
    T = 1
    sigma = 0.4
    r = 0.5
    cap = 1000
    
    S = tf.Variable([random.uniform(0,300)], name = 'S' ,dtype = 'float32')
    t = tf.Variable([random.uniform(0,T)], name = 'T' ,dtype = 'float32')
    
    Bflag = 1
    if (Bflag!=1.0):
        S_term = tf.Variable([cap], name = 'S' ,dtype = 'float32')
        T_term = tf.Variable([T], name = 'T' ,dtype = 'float32')
        term_input = tf.stack([S_term, T_term], 0)
        term = tf.Variable([cap], name = 'term' ,dtype = 'float32')
    else:
        S_t = random.uniform(0,300)
        S_term = tf.Variable([S_t], name = 'S' ,dtype = 'float32')
        T_term = tf.Variable([T], name = 'T' ,dtype = 'float32')
        term_input = tf.stack([S_term, T_term], 0)
        term = tf.Variable([max(S_t - K, 0)], name = 'term' ,dtype = 'float32')
        
    int_input = tf.Variable([[random.uniform(0,300)], [0]], name = 'term' ,dtype = 'float32')
    
    return S, t, int_input, term, term_input

def CLoss(para):
    S, t, int_input, term, term_input = gen()

    with tf.GradientTape() as tape2:
        tape2.watch(S)
        with tf.GradientTape(persistent=True) as tape:
            tape2.watch(S)
            tape2.watch(t)
            X = tf.stack([S,t],0)
            tape.watch(X)
            pred_inter = forward_propagation(X, para)
        dC_dS = tape.gradient(pred_inter, S)
        dC_dt = tape.gradient(pred_inter, t)
    d2C_dS2 = tape2.gradient(dC_dS, S)


    L1 = dC_dt + 0.5 * 0.4**2 * S**2 * d2C_dS2 + 0.05 * S * dC_dS - tf.math.scalar_mul(0.05, pred_inter)
    L2 = forward_propagation(int_input, para)
    L3 = forward_propagation(term_input, para) - term
    
    return tf.math.reduce_sum(L1**2 + L2**2 + L3**2)


# In[5]:


para = initialize_parameters()
K = tf.keras.backend
optimizer = tf.optimizers.Adam(learning_rate = 0.001)
batch_loss = 0
cost = []
for j in range(1000):
    with tf.GradientTape() as tape:
        tape.watch(para)
        for i in range(100):
            Loss = CLoss(para)
            batch_loss += Loss
        batch_loss = batch_loss / 100
    grads = tape.gradient(batch_loss, [para["W1"],para["b1"],para["W2"],para["b1"],para["W3"],para["W3"]])
    optimizer.apply_gradients(zip(grads,[para["W1"],para["b1"],para["W2"],para["b1"],para["W3"],para["W3"]]))
    print("Batch ", j, " Loss: ", K.get_value(batch_loss))
    cost.append(K.get_value(batch_loss))
    batch_loss = 0
    
plt.plot(np.squeeze(cost))
plt.ylabel('cost')
plt.xlabel('Batches')
plt.show()   

