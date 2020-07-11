

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from scipy.stats import norm
import math
from tensorflow import keras
from tensorflow.keras import layers

import DGMnets
import DGMnet2


# In[2]:

#model = DGMnets.DGMNet(3,50)


model = DGMnet2.DGMNet(3,50)

def BSM(S):
    K = 100
    T = 1
    r = 0.05
    sigma = 0.4
    d1 = (math.log(S/K) + (r+sigma**2/2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)

def graph(model):

    B = []
    index = []
    for i in range(1,200):
        index.append(i)
        B.append(BSM(i))
    
    S= tf.Variable(np.reshape(np.linspace(1,199,num=199),(199,1)), dtype = 'float32')
    t = tf.Variable(np.zeros(shape=(199,1)), dtype = 'float32')
    
    X_test = tf.concat([t, S], axis = 1)
    
    DGM = model(X_test)
    
    plt.plot(index,B, color="blue")
    plt.plot(index,DGM, color="red")
    plt.xlabel('Stock Price')
    plt.ylabel('Option Price')
    plt.legend()
    plt.show()

# In[3]:

def gen(batch_size):   
    K = 100
    T = 1

    r = 0.05
    cap = 300 
    # generate a point in the interior
    S = tf.Variable(np.random.beta(6,10, size=[batch_size, 1])*cap, dtype = 'float32')
    t = tf.Variable(np.random.uniform(0,T, size=[batch_size, 1]), dtype = 'float32')
    
    # generate a point on the boundary
    S_bound1 = tf.Variable(cap * np.ones(shape=[int(batch_size/2),1]), dtype = 'float32')
    T_bound1 = tf.Variable(np.random.uniform(0,T,size=[int(batch_size/2), 1]), dtype = 'float32')
    X_bound1 = tf.concat([T_bound1, S_bound1], axis = 1)
    bound1 = S_bound1 - K * tf.exp(-r*(T-T_bound1))

    S_bound2 = tf.Variable(np.zeros(shape=[int(batch_size/2), 1]), dtype = 'float32')
    T_bound2 = tf.Variable(np.random.uniform(0, T, size=[int(batch_size/2), 1]), dtype = 'float32')
    X_bound2 = tf.concat([T_bound2, S_bound2], axis = 1)
    bound2 = tf.Variable(np.zeros(shape=[int(batch_size/2),1]), dtype = 'float32')
    
    # generate an initial or terminal point 
    S_int = tf.Variable(np.random.beta(6,10, size=[batch_size, 1])*cap, dtype = 'float32')
    T_int = tf.Variable(T*np.ones(shape=[batch_size,1]), dtype = 'float32')
    X_int = tf.concat([T_int, S_int], axis = 1)
    initial = tf.Variable(tf.maximum(S_int - K, 0), dtype = 'float32')

    return S, t, X_int, initial, X_bound1, bound1, X_bound2, bound2
    """S_bound1, T_bound1, bound1, S_bound2, T_bound2, bound2,"""
def CLoss(model, batch_size):

    sigma = 0.4
    r = 0.05

    S, t, X_int, initial, X_bound1, bound1, X_bound2, bound2 = gen(batch_size)

    with tf.GradientTape() as tape2:
        tape2.watch(S)
        with tf.GradientTape(persistent=True) as tape:
            tape2.watch(S)
            tape2.watch(t)
            X = tf.concat([t, S], axis = 1)
            pred_inter = model(X)
        dC_dS = tape.gradient(pred_inter, S)
        dC_dt = tape.gradient(pred_inter, t)
    d2C_dS2 = tape2.gradient(dC_dS, S)
    L1 = dC_dt + 0.5 * sigma**2 * S**2 * d2C_dS2 + r * S * dC_dS - tf.math.scalar_mul(0.05, pred_inter)
    L2 = model(X_int) - initial
    L3 = tf.concat([model(X_bound1) - bound1, model(X_bound2) - bound2], axis = 0)
    
    return tf.math.reduce_mean(tf.square(L1) + 0.1 * tf.square(L2)+ 0.1 * tf.square(L3))


# In[5]:

enum = 10
epoch = 100
batch = 1024


K = tf.keras.backend
batch_loss = 0
cost = []
learning_rate = 0.01
    
for k in range(enum):
    learning_rate = learning_rate * 0.5 ** (k/enum)
    optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
    for j in range(epoch):
        with tf.GradientTape() as tape:
            batch_loss = CLoss(model,batch)
        grads = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print("Enum: ",k+1, ". Batch ", j+1, " Loss: ", K.get_value(batch_loss),". Learning Rate: ",learning_rate)
        cost.append(math.log(K.get_value(batch_loss)))
        batch_loss = 0
    model1 = model
    graph(model1)
    
plt.plot(np.squeeze(cost))
plt.ylabel('cost')
plt.xlabel('Batches')
plt.show()   
 
