#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:21:57 2020

@author: Kevin
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import DGMnet2
    
import pandas as pd
import seaborn as sns


from scipy.stats import norm
import math

T = 1
sigma1 = 0.4
sigma2 = 0.4
r = 0.05
rho = 0.5


#model = DGMnet2.DGMNet(3,50,2)
def Margrabe(S1,S2):
    nu = np.sqrt(sigma1**2 + sigma2**2 - 2*rho*sigma1*sigma2)
    d1 = (math.log(S1/S2) + (nu**2/2)*T)/(nu*math.sqrt(T))
    d2 = d1 - nu*math.sqrt(T)
    return S1*norm.cdf(d1)-S2*norm.cdf(d2)

def graph(model,k):
    B = []
    DGM = []
    X = []
    Y = []
    I = []
    for i in range(100,210,10):
        I.append(i)
        for j in range(100,210,10):
            X.append(j)
            Y.append(i)
            B.append(Margrabe(j,i))
    
    M = np.reshape(np.array(B), newshape=(11,11))
    
    X = np.array(X)
    Y = np.array(Y)
    
    S1_test = tf.Variable(X, dtype = 'float32')
    S2_test = tf.Variable(Y, dtype = 'float32')
    T_test = tf.Variable(np.zeros(shape=(121)), dtype = 'float32')
    
    
    Input = tf.stack([S1_test, S2_test, T_test], axis=1)
    DGM = model(Input)
    DGM = K.get_value(tf.reshape(DGM, shape=(11,11)))
    
    error = np.round(abs(DGM - M), decimals=2)
    
    vmax = max(np.max(error),1)
    
    Mdata = pd.DataFrame(error, columns=I, index=I)
    
    ax = sns.heatmap(Mdata, cmap="Spectral_r",vmin=0, vmax=vmax,annot = True)
    ax.invert_yaxis()
    plt.title(r"$L_1$" + " error     epoch: " + str(k))
    plt.xlabel(r"$S_1$")
    plt.ylabel(r"$S_2$")
    plt.show()
    

model = DGMnet2.DGMNet(3, 50, 2)


# In[3]:


def gen(batch_size):
    cap = 1000
    floor = 0
    bound_size = int(batch_size/4)

    # generate a point in the interior
    S1 = tf.Variable(np.random.uniform(50, 250, size=[batch_size, 1]), dtype = 'float32')
    S2 = tf.Variable(np.random.uniform(50, 250, size=[batch_size, 1]), dtype = 'float32')
    t = tf.Variable(np.random.uniform(0,T, size=[batch_size, 1]), dtype = 'float32')
    
    # generate a boundary point
    S1_bound1 = tf.Variable(cap * np.ones(shape=[bound_size, 1]), dtype = 'float32')
    S2_bound1 = tf.Variable(np.random.beta(2, 10, size=[bound_size, 1])*cap, dtype = 'float32')
    T_bound1 = tf.Variable(np.random.uniform(0,T,size=[bound_size, 1]), dtype = 'float32')
    X_bound1 = tf.concat([S1_bound1, S2_bound1, T_bound1], axis = 1)
    bound1 = S1_bound1 - S2_bound1
    
    S1_bound2 = tf.Variable(np.random.beta(3, 10, size=[bound_size, 1])*cap, dtype = 'float32')
    S2_bound2 = tf.Variable(floor * np.ones(shape=[bound_size, 1]), dtype = 'float32')
    T_bound2 = tf.Variable(np.random.uniform(0,T,size=[bound_size, 1]), dtype = 'float32')
    X_bound2 = tf.concat([S1_bound2, S2_bound2, T_bound2], axis = 1)
    bound2 = S1_bound2
    
    S1_bound3 = tf.Variable(floor * np.ones(shape=[bound_size, 1]), dtype = 'float32')
    S2_bound3 = tf.Variable(np.random.beta(2, 10, size=[bound_size, 1])*cap, dtype = 'float32')
    T_bound3 = tf.Variable(np.random.uniform(0,T,size=[bound_size, 1]), dtype = 'float32')
    X_bound3 = tf.concat([S1_bound3, S2_bound3, T_bound3], axis = 1)
    bound3 = tf.Variable(np.zeros(shape=[bound_size, 1]), dtype = 'float32')
    
    S1_bound4 = tf.Variable(np.random.beta(3, 10, size=[bound_size, 1])*cap, dtype = 'float32')
    S2_bound4 = tf.Variable(cap * np.ones(shape=[bound_size, 1]), dtype = 'float32')
    T_bound4 = tf.Variable(np.random.uniform(0,T,size=[bound_size, 1]), dtype = 'float32')
    X_bound4 = tf.concat([S1_bound4, S2_bound4, T_bound4], axis = 1)
    bound4 = tf.Variable(np.zeros(shape=[bound_size, 1]), dtype = 'float32')
    
    # generate an initial or terminal point 
    S1_int = tf.Variable(np.random.beta(3, 10, size=[batch_size, 1])*cap, dtype = 'float32')
    S2_int = tf.Variable(np.random.beta(2, 10, size=[batch_size, 1])*cap, dtype = 'float32')
    T_int = tf.Variable(T*np.ones(shape=[batch_size,1]), dtype = 'float32')
    int_input = tf.concat([S1_int, S2_int, T_int], axis=1)
    initial = tf.Variable(tf.maximum(S1_int - S2_int, 0), dtype = 'float32')
    
    return S1, S2, t, initial, int_input, X_bound1, bound1, X_bound2, bound2, X_bound3, bound3, X_bound4, bound4

def CLoss(model, b_size):
    S1, S2, t, initial, int_input, X_bound1, bound1, X_bound2, bound2, X_bound3, bound3, X_bound4, bound4 = gen(b_size)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(S1)
        tape2.watch(S2)
        with tf.GradientTape(persistent=True) as tape:
            tape2.watch(S1)
            tape2.watch(S2)
            tape2.watch(t)
            X = tf.concat([S1,S2,t], axis=1)
            pred_inter = model(X)
        dC_dS1 = tape.gradient(pred_inter, S1)
        #print("dC_dS1: ",tf.math.reduce_mean(dC_dS1))
        dC_dS2 = tape.gradient(pred_inter, S2)
        #print("dC_dS2: ",tf.math.reduce_mean(dC_dS2))
        dC_dt = tape.gradient(pred_inter, t)
        #print("dC_dt: ",tf.math.reduce_mean(dC_dt))
    d2C_dS1 = tape2.gradient(dC_dS1, S1)
    #print("d2C_dS1: ",tf.math.reduce_mean(d2C_dS1))
    d2C_dS2 = tape2.gradient(dC_dS2, S2)
    #print("d2C_dS2: ",tf.math.reduce_mean(d2C_dS2))
    d2C_dS1S2 = tape2.gradient(dC_dS1, S2)
    #print("d2C_dS1S2: ",tf.math.reduce_mean(d2C_dS1S2))
    d2C_dS2S1 = tape2.gradient(dC_dS2, S1)
    #print("d2C_dS1S2: ",tf.math.reduce_mean(d2C_dS1S2))
    d2C_dS12 = (d2C_dS1S2 + d2C_dS2S1) / 2
    #print("d2C_dS12: ",tf.math.reduce_mean(d2C_dS12))
    L1 = dC_dt + r * S1 * dC_dS1 + r * S2 * dC_dS2 + 0.5 * sigma1**2 * S1**2 * d2C_dS1 + 0.5 * sigma2**2 * S2**2 * d2C_dS2 + rho * sigma1 * sigma2 * S1 * S2 * d2C_dS12 - tf.math.scalar_mul(r, pred_inter)
    L2 = model(int_input) - initial
    L3 = tf.concat([model(X_bound1) - bound1, model(X_bound2) - bound2, model(X_bound3) - bound3, model(X_bound4) - bound4], axis=0)
    
    return tf.math.reduce_mean(tf.square(L1) + 0.1*tf.square(L2) + 0.1*tf.square(L3))


# In[5]:

epoch= 50
batch_number = 100
batch_size = 1024


K = tf.keras.backend
batch_loss = 0
cost = []
learning_rate = 0.01

for k in range(epoch):
    optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
    learning_rate = learning_rate * 0.5 ** (k/epoch)
    for j in range(batch_number):
        with tf.GradientTape() as tape:
            batch_loss = CLoss(model,batch_size)
        grads = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        print("Enum: ",k+1, ". Batch ", j+1, " Loss: ", K.get_value(batch_loss),". Learning Rate: ",learning_rate)
        cost.append(math.log(K.get_value(batch_loss)))
        batch_loss = 0
    graph(model,k+1)
    model.save_weights('/Users/Jkzhang/Desktop/GitHub/DGM/Exchange32/weight32' + str(k))
    #model.save('/Users/Jkzhang/Desktop/GitHub/DGM/Exchange32/model32' + str(k) + '.h5')
plt.plot(np.squeeze(cost))
plt.ylabel('cost')
plt.xlabel('Batches')
plt.show()   

#

#print(DGM-M)
