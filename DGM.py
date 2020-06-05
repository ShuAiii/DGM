#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from scipy.stats import norm
import math


# In[2]:


def initialize_parameters():
    x = 2000
    initializer_X = tf.initializers.GlorotUniform()
    initializer_Z = tf.initializers.zeros()
    
    W1 = tf.Variable(initializer_X(shape = (x,2)))
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
    K = 100
    T = 1
    sigma = 0.4
    r = 0.5
    cap = 10000
    
    # generate a point in the interior
    S = tf.Variable([np.random.beta(2,100)*10000], dtype = 'float32')
    t = tf.Variable([random.uniform(0,T)], dtype = 'float32')
    
    # generate a point on the boundary
    Bflag = random.choice([0, 1])
    if (Bflag!=1.0):
        S_bound = tf.Variable([cap], dtype = 'float32')
        T_bound = tf.Variable([random.uniform(0,T)], dtype = 'float32')
        bound_input = tf.stack([S_bound, T_bound], 0)
        bound = tf.Variable([cap], dtype = 'float32')
    else:
        bound_input = tf.Variable([[0], [random.uniform(0,T)]], dtype = 'float32')
        bound = tf.Variable([0], dtype = 'float32')
        
    # generate an initial or terminal point 
    S_int = np.random.beta(2,100)*10000
    S_int = tf.Variable([S_int], dtype = 'float32')
    T_int = tf.Variable([T], dtype = 'float32')
    int_input = tf.stack([S_int, T_int], 0)
    initial = tf.Variable([max(S_int - K, 0)], dtype = 'float32')
    
    return S, t, initial, int_input, bound, bound_input

def CLoss(para):
    S, t, initial, int_input, bound, bound_input = gen()

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
    L2 = forward_propagation(int_input, para) - initial
    L3 = forward_propagation(bound_input, para) - bound
    
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
learning_rate = 0.00001
    
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

def BSM(S):
    K = 100
    T = 1
    r = 0.05
    sigma = 0.4
    d1 = (math.log(S/K) + (r+sigma**2/2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)

B = []
DGM = []
index = []
for i in range(1,200):
    index.append(i)
    B.append(BSM(i))
    S_test = tf.Variable([i], dtype = 'float32')
    T_test = tf.Variable([1], dtype = 'float32')
    I = tf.stack([S_test, T_test], 0)
    DGM.append(K.get_value(forward_propagation(I,para))[0][0])

print(B)
print(DGM)

plt.plot(index,B, color="blue")
plt.plot(index,DGM, color="red")
plt.xlabel('Stock Price')
plt.ylabel('Option Price')
plt.legend()
plt.show()   

