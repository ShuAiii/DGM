#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:21:57 2020

@author: Kevin
"""


import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from scipy.stats import norm
import math
from mpl_toolkits.mplot3d import Axes3D


T = 1
sigma1 = 0.4
sigma2 = 0.4
r = 0.5
rho = 0.5


def initialize_parameters():
    x = 2000
    initializer_X = tf.initializers.GlorotUniform()
    initializer_Z = tf.initializers.zeros()
    
    W1 = tf.Variable(initializer_X(shape = (x,3)))
    b1 = tf.Variable(initializer_Z(shape = (x,1)))
    W2 = tf.Variable(initializer_X(shape = (x,x)))
    b2 = tf.Variable(initializer_Z(shape = (x,1)))
    W3 = tf.Variable(initializer_X(shape = (x,x)))
    b3 = tf.Variable(initializer_Z(shape = (x,1)))
    W4 = tf.Variable(initializer_X(shape = (x,x)))
    b4 = tf.Variable(initializer_Z(shape = (x,1)))
    W5 = tf.Variable(initializer_X(shape = (x,x)))
    b5 = tf.Variable(initializer_Z(shape = (x,1)))
    W6 = tf.Variable(initializer_X(shape = (1,x)))
    b6 = tf.Variable(initializer_Z(shape = (1,1)))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6}
    
    return parameters

def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    W6 = parameters['W6']
    b6 = parameters['b6']
    Z1 = tf.add(tf.matmul(W1, X), b1) 
    A1 = tf.nn.swish(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  
    A2 = tf.nn.swish(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  
    A3 = tf.nn.swish(Z3)                                  
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    A4 = tf.nn.swish(Z4)  
    Z5 = tf.add(tf.matmul(W5, A4), b5)
    A5 = tf.nn.swish(Z5)  
    Z6 = tf.add(tf.matmul(W6, A5), b6)
    A6 = Z6
    return A6


# In[3]:


def gen():
    cap = 10000
    floor = 0

    # generate a point in the interior
    S1 = tf.Variable([np.random.beta(2,100)*10000], dtype = 'float32')
    S2 = tf.Variable([np.random.beta(2,100)*10000], dtype = 'float32')
    t = tf.Variable([random.uniform(0,T)], dtype = 'float32')
    
    # generate a boundary point
    Bflag = random.choice([0, 1, 2, 3])
    if (Bflag==1.0):
        S1_term = tf.Variable([cap], dtype = 'float32')
        S2_term = tf.Variable([np.random.beta(2,100)*10000], dtype = 'float32')
        T_term = tf.Variable([random.uniform(0,T)], dtype = 'float32')
        term_input = tf.stack([S1_term, S2_term, T_term], 0)
        term = tf.Variable([cap], dtype = 'float32')
    elif (Bflag==0.0):
        S1_term = tf.Variable([np.random.beta(2,100)*10000], dtype = 'float32')
        S2_term = tf.Variable([floor], dtype = 'float32')
        T_term = tf.Variable([random.uniform(0,T)], dtype = 'float32')
        term_input = tf.stack([S1_term, S2_term, T_term], 0)
        term = S1_term
    elif (Bflag==2.0):
        S1_term = tf.Variable([0], dtype = 'float32')
        S2_term = tf.Variable([np.random.beta(2,100)*10000], dtype = 'float32')
        T_term = tf.Variable([random.uniform(0,T)], dtype = 'float32')
        term_input = tf.stack([S1_term, S2_term, T_term], 0)
        term = tf.Variable([0], dtype = 'float32')
    else:
        S1_term =  np.random.beta(2,100)*10000
        term = tf.Variable([S1_term*np.exp(0.5**T*sigma1**2)], dtype = 'float32')
        S1_term = tf.Variable([S1_term], dtype = 'float32')
        S2_term = tf.Variable([0], dtype = 'float32')
        T_term = tf.Variable([random.uniform(0,T)], dtype = 'float32')
        term_input = tf.stack([S1_term, S2_term, T_term], 0)
    
    # generate an initial or terminal point 
    int_input = tf.Variable([[0], [random.uniform(0,T)]], dtype = 'float32')
    S1_int = np.random.beta(2,100)*10000
    S1_int = tf.Variable([S1_int], dtype = 'float32')
    S2_int = np.random.beta(2,100)*10000
    S2_int = tf.Variable([S2_int], dtype = 'float32')
    T_int = tf.Variable([T], dtype = 'float32')
    int_input = tf.stack([S1_int, S2_int, T_int], 0)
    initial = tf.Variable([max(S1_int - S2_int, 0)], dtype = 'float32')
    
    return S1, S2, t, initial, int_input, term, term_input

def CLoss(para):
    S1, S2, t, initial, int_input, term, term_input = gen()

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(S1)
        tape2.watch(S2)
        with tf.GradientTape(persistent=True) as tape:
            tape2.watch(S1)
            tape2.watch(S2)
            tape2.watch(t)
            X = tf.stack([S1,S2,t],0)
            pred_inter = forward_propagation(X, para)
        dC_dS1 = tape.gradient(pred_inter, S1)
        dC_dS2 = tape.gradient(pred_inter, S2)
        dC_dt = tape.gradient(pred_inter, t)
    d2C_dS1 = tape2.gradient(dC_dS1, S1)
    d2C_dS2 = tape2.gradient(dC_dS2, S2)
    d2C_dS1S2 = tape2.gradient(dC_dS1, S2)
    d2C_dS2S1 = tape2.gradient(dC_dS2, S1)
    d2C_dS12 = (d2C_dS1S2 + d2C_dS2S1) / 2
    L1 = dC_dt + r * S1 * dC_dS1 + r * S2 * dC_dS2 + r * S1 * dC_dS1 + 0.5 * sigma1**2 * S1**2 * d2C_dS1 + 0.5 * sigma2**2 * S2**2 * d2C_dS2 + rho * sigma1 * sigma2 * S1 * S2 * d2C_dS12 - tf.math.scalar_mul(r, pred_inter)
    L2 = forward_propagation(int_input, para)
    L3 = forward_propagation(term_input, para) - term
    
    return tf.math.reduce_sum(L1**2 + L2**2 + L3**2)


# In[5]:

enum = 10
epoch = 100
batch = 200

para = initialize_parameters()
parameters = []
for key in para.keys():
    parameters.append(para[key])
K = tf.keras.backend
batch_loss = 0
cost = []
learning_rate = 0.0001
    
for k in range(enum):
    learning_rate = learning_rate * 0.5 ** (k/enum)
    optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
    for j in range(epoch):
        with tf.GradientTape() as tape:
            tape.watch(para)
            for i in range(batch):
                Loss = CLoss(para)
                batch_loss += Loss
            batch_loss = batch_loss / batch
        grads = tape.gradient(batch_loss, parameters)
        optimizer.apply_gradients(zip(grads,parameters))
        print("Enum: ",k+1, ". Batch ", j+1, " Loss: ", K.get_value(batch_loss),". Learning Rate: ",learning_rate)
        cost.append(K.get_value(batch_loss))
        batch_loss = 0
plt.plot(np.squeeze(cost))
plt.ylabel('cost')
plt.xlabel('Batches')
plt.show()   

def Margrabe(S1,S2):
    nu = np.sqrt(sigma1**2 + sigma2**2 - 2*rho*sigma1*sigma2)
    d1 = (math.log(S1/S2) + (nu**2/2)*T)/(nu*math.sqrt(T))
    d2 = d1 - nu*math.sqrt(T)
    return S1*norm.cdf(d1)-S2*norm.cdf(d2)

B = []
DGM = []
X = []
Y = []
for i in range(1,10):
    X.append(i)
    for j in range(1,10):
        B.append(Margrabe(i,j))
        S1_test = tf.Variable([i], dtype = 'float32')
        S2_test = tf.Variable([j], dtype = 'float32')
        T_test = tf.Variable([1], dtype = 'float32')
        I = tf.stack([S1_test,S2_test, T_test], 0)
        DGM.append(K.get_value(forward_propagation(I,para))[0][0])

print(B)
print(DGM)
X,Y = np.meshgrid(X,X)
fig = plt.figure()
ax1 = fig.add_subplot(211, projection='3d')
ax1.invert_yaxis()
ax2 = fig.add_subplot(212, projection='3d')
ax2.invert_yaxis()
B = np.array(B)
B = B.reshape(9,9)
DGM = np.array(DGM)
DGM = DGM.reshape(9,9)
ax1.plot_surface(X,Y,B,rstride=1, cstride=1)
ax2.plot_surface(X,Y,DGM,rstride=1, cstride=1)