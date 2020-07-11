#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:21:57 2020

@author: Kevin
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import DGMnets
import DGMnet2
import SpreadFourier
    
import pandas as pd
import seaborn as sns


import math
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import cm


K = tf.keras.backend

T = 1
sigma1 = 0.4
sigma2 = 0.2
r = 0.05
rho = 0.5
strike = 4
#f = SpreadFourier.spread_inter(300, 300, strike, r, sigma1, sigma2, rho, 1, 512)


data1 = pd.read_csv("/Users/Jkzhang/Desktop/GitHub/DGM/SpreadFourierK4.csv", header=None)
data2 = pd.read_csv("/Users/Jkzhang/Desktop/GitHub/DGM/SpreadBSMK4.csv", header=None)

data1 = data1.transpose().to_numpy()

def graph(model,k):
    DGM = []
    X = []
    Y = []
    N = []
    I = []
    for i in range(5,105,5):
        I.append(i)
        for j in range(5,105,5):
            X.append(j)
            Y.append(i)
            N.append(max(j-i-k,0))
    

    
    X = np.array(X)
    Y = np.array(Y)
    
    S1_test = tf.Variable(X, dtype = 'float32')
    S2_test = tf.Variable(Y, dtype = 'float32')
    T_test = tf.Variable(np.zeros(shape=(400)), dtype = 'float32')
    
    
    Input = tf.stack([S1_test, S2_test, T_test], axis=1)
    DGM = model(Input)
    DGM = K.get_value(tf.reshape(DGM, shape=(20,20)))

    error = np.round(abs(DGM - data1), decimals=2)

    vmax = max(np.max(error),1)

    Mdata = pd.DataFrame(DGM , columns=I, index=I)

    ax = sns.heatmap(Mdata, cmap="coolwarm",vmin=0, vmax = vmax)
    ax.invert_yaxis()
    plt.title(r"$L_1$" + " error on batch: " + str(k))
    plt.xlabel(r"$S_1$")
    plt.ylabel(r"$S_2$")
    plt.show()
    
    xx, yy = np.meshgrid(I,I)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, DGM, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.7, aspect = 10)
    plt.title("Option Price    Batch: " + str(k))
    plt.xlabel(r"$S_1$")
    plt.ylabel(r"$S_2$")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, data1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect = 10)
    ax.invert_yaxis()
    plt.show()


model = DGMnet2.DGMNet(3,50,2)



def tfnormcdf(tensor):
    return tf.math.scalar_mul(0.5, tf.math.erfc(-tf.math.scalar_mul(math.sqrt(0.5), tensor)))

def tf_bsm(S1, t, strike):
    d1 = (tf.math.log(S1 / strike) + (r + 0.5 * sigma1 * sigma1) * (T - t)) / (tf.math.sqrt(T - t) * sigma1)
    d2 = d1 - math.sqrt(T) * sigma1
    return S1 * tfnormcdf(d1) - strike * tf.math.exp(-r * (T - t)) * tfnormcdf(d2)
    
    
    

# In[3]:


def gen(batch_size):
    cap = 1000
    floor = 0
    bound_size = int(batch_size/4)
    
    S1 = tf.Variable(np.random.beta(4, 50, size=[batch_size, 1]) * cap, dtype = 'float32')
    S2 = tf.Variable(np.random.beta(3, 50, size=[batch_size, 1]) * cap, dtype = 'float32')
    t = tf.Variable(np.random.uniform(0,T, size=[batch_size, 1]), dtype = 'float32')
    
    # generate a boundary point
    S1_bound1 = cap * tf.ones(shape=[bound_size, 1], dtype = 'float32')
    S2_bound1 = cap * tf.Variable(np.random.beta(3, 50, size=[bound_size, 1]), dtype = 'float32')
    T_bound1 = tf.random.uniform(minval=0, maxval=T, shape=[bound_size, 1], dtype = 'float32')
    X_bound1 = tf.concat([S1_bound1, S2_bound1, T_bound1], axis = 1)
    bound1 = S1_bound1 - S2_bound1 - strike * tf.exp(-r*(T-T_bound1))

    S1_bound2 = cap * tf.Variable(np.random.beta(4, 50, size=[bound_size, 1]), dtype = 'float32')
    S2_bound2 = tf.zeros(shape=[bound_size, 1], dtype = 'float32')
    T_bound2 = tf.random.uniform(minval=0, maxval=T, shape=[bound_size, 1], dtype = 'float32')
    X_bound2 = tf.concat([S1_bound2, S2_bound2, T_bound2], axis = 1)
    bound2 = tf_bsm(S1_bound2, T_bound2, strike)

    S1_bound3 = tf.zeros(shape=[bound_size, 1], dtype = 'float32')
    S2_bound3 = cap * tf.Variable(np.random.beta(3, 50, size=[bound_size, 1]), dtype = 'float32')
    T_bound3 = tf.random.uniform(minval=0, maxval=T, shape=[bound_size, 1], dtype = 'float32')
    X_bound3 = tf.concat([S1_bound3, S2_bound3, T_bound3], axis = 1)
    bound3 = tf.zeros(shape=[bound_size, 1], dtype = 'float32')
    
    S1_bound4 = cap * tf.Variable(np.random.beta(4, 50, size=[bound_size, 1]), dtype = 'float32')
    S2_bound4 = cap * tf.ones(shape=[bound_size, 1], dtype = 'float32')
    T_bound4 = tf.random.uniform(minval=0, maxval=T, shape=[bound_size, 1], dtype = 'float32')
    X_bound4 = tf.concat([S1_bound4, S2_bound4, T_bound4], axis = 1)
    bound4 = tf.zeros(shape=[bound_size, 1], dtype = 'float32')
    
    X_bound = tf.concat([X_bound1, X_bound2, X_bound3, X_bound4], axis = 0)
    bound = tf.concat([bound1, bound2, bound3, bound4], axis = 0)
    
    # generate an initial or terminal point 
    S1_int = cap * tf.Variable(np.random.beta(4, 50, size=[batch_size, 1]), dtype = 'float32')
    S2_int = cap * tf.Variable(np.random.beta(3, 50, size=[batch_size, 1]), dtype = 'float32')
    T_int = T * tf.Variable(np.ones(shape=[batch_size,1]), dtype = 'float32')
    int_input = tf.concat([S1_int, S2_int, T_int], axis=1)
    initial = tf.Variable(tf.maximum(S1_int - S2_int - strike, 0), dtype = 'float32')
    
    return S1, S2, t, initial, int_input, X_bound, bound#, X_bound2, bound2, X_bound3, bound3, X_bound4, bound4

def CLoss(model, b_size):
    S1, S2, t , initial, int_input, X_bound, bound = gen(b_size) #, X_bound2, bound2, X_bound3, bound3, X_bound4, bound4 = gen(b_size)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(S1)
        tape2.watch(S2)
        with tf.GradientTape(persistent=True) as tape:
            tape2.watch(S1)
            tape2.watch(S2)
            tape2.watch(t)
            X = tf.concat([S1,S2,t], axis=1)
            V = model(X)
        dC_dS1 = tape.gradient(V, S1)
        dC_dS2 = tape.gradient(V, S2)
        dC_dt = tape.gradient(V, t)
    d2C_dS1 = tape2.gradient(dC_dS1, S1)
    d2C_dS2 = tape2.gradient(dC_dS2, S2)
    d2C_dS1S2 = tape2.gradient(dC_dS1, S2)
    d2C_dS2S1 = tape2.gradient(dC_dS2, S1)
    d2C_dS12 = (d2C_dS1S2 + d2C_dS2S1) / 2
    
    v11 = 0.5 * d2C_dS1 * sigma1**2 * S1**2 
    v12 = d2C_dS12 * rho * sigma1 * sigma2 * S1 * S2 
    v22 = 0.5 * sigma2**2 * S2**2 * d2C_dS2

    L1 = dC_dt + r * S1 * dC_dS1 + r * S2 * dC_dS2 + v11 + v12 + v22 - r * V
    L2 = model(int_input) - initial
    L3 = model(X_bound) - bound

    L2 = model(int_input) - initial

    L3 = model(X_bound) - bound
    
    return tf.math.reduce_mean(tf.square(L1)), 0.1 * tf.math.reduce_mean(tf.square(L2)), 0.1 * tf.math.reduce_mean(tf.abs(L3))#, 0.05*tf.math.reduce_mean(tf.square(L4)), 0.05*tf.math.reduce_mean(tf.square(L5)), 0.05*tf.math.reduce_mean(tf.square(L6))


# In[5]:

epoch= 50
batch_number = 100
batch_size = 1024


batch_loss = 0
cost = []
learning_rate = 0.0005
optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
#learning_rate = learning_rate * 0.5 ** (k/epoch) 
for k in range(epoch):
    #optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
    #learning_rate = learning_rate * 0.5 ** (k/epoch) 
    for j in range(batch_number):
        with tf.GradientTape() as tape:
            L1, L2, L3 = CLoss(model,batch_size)
            batch_loss = L1 + L2 + L3# + L4 + L5 + L6
        grads = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        print("Epoch: ",k+1, ". Batch ", j+1, " Loss: ", K.get_value(batch_loss),". Learning Rate: ",learning_rate)
        cost.append(math.log(K.get_value(batch_loss)))
        batch_loss = 0
    print("L1: ", K.get_value(L1), "L2: ", K.get_value(L2), "L3: ", K.get_value(L3))#, "L4: ", K.get_value(L4), "L5: ", K.get_value(L5), "L6: ", K.get_value(L6))
    graph(model,k+1)
    #model.save_weights('/Users/Jkzhang/Desktop/GitHub/DGM/SpreadTransfer32/weight' + str(k))
plt.plot(np.squeeze(cost))
plt.ylabel('cost')
plt.xlabel('Batches')
plt.show()   

#model.save('/Users/Jkzhang/Desktop/GitHub/DGM/Spread32.h5')

#print(DGM-M)
