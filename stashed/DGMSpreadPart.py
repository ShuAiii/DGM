import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from stashed import DGMnet2

import pandas as pd
import seaborn as sns

import math
from tensorflow import keras
from matplotlib import cm

K = tf.keras.backend

T = 1
sigma1 = 0.4
sigma2 = 0.2
r = 0.05
rho = 0.5
strike = 4
cap = 1000
floor = 0

spreaddata = pd.read_csv("/Users/Jkzhang/Desktop/GitHub/DGM/SpreadFourierK4.csv", header=None)
spreaddata = spreaddata.iloc[:,0].to_numpy()

def graph(model,model_bsm,k):
    B = []
    DGM = []
    X = []
    Y = []
    I = []
    for i in range(5,105,5):
        I.append(i)
        for j in range(5,105,5):
            X.append(j)
            Y.append(i)
    
    
    
    #M = np.transpose(np.reshape(spreaddata, newshape=(22,22)))
    
    X = np.array(X)
    Y = np.array(Y)
    XX, YY = np.meshgrid(I, I)
    
    S1_test = tf.Variable(X, dtype = 'float64')
    S2_test = tf.Variable(Y, dtype = 'float64')
    T_test = tf.Variable(np.zeros(shape=(400)), dtype = 'float64')
    
    
    Input = tf.stack([S1_test, S2_test, T_test], axis=1)
    DGM = model(Input)
    DGM = K.get_value(tf.reshape(DGM, shape=(20,20)))
    
    M = model_bsm(Input)
    M = K.get_value(tf.reshape(M, shape=(20,20)))
    
    error = np.round(DGM - M, decimals=2)
    Mdata = pd.DataFrame(error, columns=I, index=I)
    
    ax = sns.heatmap(Mdata, cmap="coolwarm")
    ax.invert_yaxis()
    plt.title("Liquidity Value Adjustment (Part Impact) on epoch: " + str(k))
    plt.xlabel(r"$S_1$")
    plt.ylabel(r"$S_2$")
    plt.show()
    
    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    surf1 = ax1.plot_surface(XX, YY, Mdata, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig1.colorbar(surf1, shrink=0.7, aspect=10)
    ax1.invert_yaxis()
    plt.title("LVA    " + r"$\epsilon=0.3$    Batch: " + str(k))
    plt.xlabel(r"$S_1$")
    plt.ylabel(r"$S_2$")
    plt.tight_layout()
    plt.show()
    
    '''
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    surf2 = ax2.plot_surface(XX, YY, DGM, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig2.colorbar(surf2, shrink=0.7, aspect=10)
    plt.title("Option Price    " + r"$\epsilon=0.3$    Batch: " + str(k))
    plt.xlabel(r"$S_1$")
    plt.ylabel(r"$S_2$")
    ax2.invert_yaxis()
    plt.tight_layout()
    plt.show()
    '''


model = DGMnet2.DGMNet(3, 50, 2)

model_bsm = keras.models.load_model('/Users/Jkzhang/Desktop/GitHub/DGM/Spread32.h5')
old_weights = model_bsm.get_weights()


K.set_floatx('float64')
wsp = [w.astype(K.floatx()) for w in old_weights]

model.set_weights(wsp)

model_bsm.set_weights(wsp)

def tfnormcdf(tensor):
    return tf.math.scalar_mul(0.5, tf.math.erfc(-tf.math.scalar_mul(math.sqrt(0.5), tensor)))

def tf_bsm(S1, t, strike):
    d1 = (tf.math.log(S1 / strike) + (r + 0.5 * sigma1 * sigma1) * (T - t)) / (tf.math.sqrt(T - t) * sigma1)
    d2 = d1 - math.sqrt(T) * sigma1
    return S1 * tfnormcdf(d1) - strike * tf.math.exp(-r * (T - t)) * tfnormcdf(d2)

def tf_impact(s1,s2,t):
    eta = 0.03
    beta = 100
    s1I = tf.cast(tf.math.greater(s1,0), tf.float64) * tf.cast(tf.math.less(s1,cap), tf.float64)
    s2I = tf.cast(tf.math.greater(s2,0), tf.float64) * tf.cast(tf.math.less(s2,cap), tf.float64)
    I = s1I * s2I
    return eta * (1-tf.math.exp(-beta * tf.math.pow(T - t,3/2))) * I
    

# In[3]:
def gen(batch_size):
    
    bound_size = int(batch_size/4)
    
    # generate a point in the interior
    S1 = tf.Variable(np.random.beta(4, 50, size=[batch_size, 1]) * cap, dtype = 'float64')
    S2 = tf.Variable(np.random.beta(3, 50, size=[batch_size, 1]) * cap, dtype = 'float64')
    t = tf.Variable(np.random.uniform(0,T, size=[batch_size, 1]), dtype = 'float64')
    
    # generate a boundary point
    S1_bound1 = cap * tf.ones(shape=[bound_size, 1], dtype = 'float64')
    S2_bound1 = cap * tf.Variable(np.random.beta(3, 50, size=[bound_size, 1]), dtype = 'float64')
    T_bound1 = tf.random.uniform(minval=0, maxval=T, shape=[bound_size, 1], dtype = 'float64')
    X_bound1 = tf.concat([S1_bound1, S2_bound1, T_bound1], axis = 1)
    bound1 = S1_bound1 - S2_bound1 - strike * tf.exp(-r*(T-T_bound1))

    S1_bound2 = cap * tf.Variable(np.random.beta(4, 50, size=[bound_size, 1]), dtype = 'float64')
    S2_bound2 = tf.zeros(shape=[bound_size, 1], dtype = 'float64')
    T_bound2 = tf.random.uniform(minval=0, maxval=T, shape=[bound_size, 1], dtype = 'float64')
    X_bound2 = tf.concat([S1_bound2, S2_bound2, T_bound2], axis = 1)
    bound2 = tf_bsm(S1_bound2, T_bound2, strike)

    S1_bound3 = tf.zeros(shape=[bound_size, 1], dtype = 'float64')
    S2_bound3 = cap * tf.Variable(np.random.beta(3, 50, size=[bound_size, 1]), dtype = 'float64')
    T_bound3 = tf.random.uniform(minval=0, maxval=T, shape=[bound_size, 1], dtype = 'float64')
    X_bound3 = tf.concat([S1_bound3, S2_bound3, T_bound3], axis = 1)
    bound3 = tf.zeros(shape=[bound_size, 1], dtype = 'float64')
    
    S1_bound4 = cap * tf.Variable(np.random.beta(4, 50, size=[bound_size, 1]), dtype = 'float64')
    S2_bound4 = cap * tf.ones(shape=[bound_size, 1], dtype = 'float64')
    T_bound4 = tf.random.uniform(minval=0, maxval=T, shape=[bound_size, 1], dtype = 'float64')
    X_bound4 = tf.concat([S1_bound4, S2_bound4, T_bound4], axis = 1)
    bound4 = tf.zeros(shape=[bound_size, 1], dtype = 'float64')
    
    X_bound = tf.concat([X_bound1, X_bound2, X_bound3, X_bound4], axis = 0)
    bound = tf.concat([bound1, bound2, bound3, bound4], axis = 0)
    
    # generate an initial or terminal point 
    S1_int = cap * tf.Variable(np.random.beta(4, 50, size=[batch_size, 1]), dtype = 'float64')
    S2_int = cap * tf.Variable(np.random.beta(3, 50, size=[batch_size, 1]), dtype = 'float64')
    T_int = T * tf.Variable(np.ones(shape=[batch_size,1]), dtype = 'float64')
    int_input = tf.concat([S1_int, S2_int, T_int], axis=1)
    initial = tf.Variable(tf.maximum(S1_int - S2_int - strike, 0), dtype = 'float64')
    
    return S1, S2, t, initial, int_input, X_bound, bound

def CLoss(model,model_bsm, b_size):
    S1, S2, t, initial, int_input, X_bound, bound = gen(b_size) #, X_bound1, bound1, X_bound2, bound2, X_bound3, bound3, X_bound4, bound4 = gen(b_size)
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(S1)
        tape2.watch(S2)
        with tf.GradientTape(persistent=True) as tape:
            tape2.watch(S1)
            tape2.watch(S2)
            X = tf.concat([S1,S2,t], axis=1)
            Vbs = model_bsm(X)
        bs1 = tape.gradient(Vbs, S1)
        bs2 = tape.gradient(Vbs, S2)
    bs11 = tape2.gradient(bs1, S1)
    bs12n = tape2.gradient(bs1, S2)
    bs21n = tape2.gradient(bs2, S1)
    bs12 = (bs12n + bs21n) / 2
    
    
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
    
    
    impact = tf_impact(S1,S2,t)
    deno = 1 - bs11 * impact
    
    v11 =  d2C_dS1 * (sigma1**2 * S1**2 + impact**2 * bs12**2 * sigma2**2 * S2**2 + 2 * impact * bs12 * rho * sigma1 * sigma2 * S1 * S2) / (2 * deno**2)
    v12 = d2C_dS12 * (rho * sigma1 * sigma2 * S1 * S2 + impact * bs12 * sigma2**2 * S2**2) / deno
    v22 = 0.5 * sigma2**2 * S2**2 * d2C_dS2
    L1 = dC_dt + r * S1 * dC_dS1 + r * S2 * dC_dS2 + v11 + v12 + v22 - r * V
    L2 = model(int_input) - initial
    L3 = model(X_bound) - bound
    
    return tf.math.reduce_mean(tf.square(L1) + 0.1 * tf.square(L2) + 0.1*tf.square(L3))


# In[5]:

epoch= 200
batch_number = 10
batch_size = 1024


batch_loss = 0
cost = []
learning_rate = 0.0000005
#learning_rate = learning_rate * 0.5 ** (k/epoch)
optimizer = tf.optimizers.Adam(learning_rate = learning_rate)

for k in range(epoch):
    for j in range(batch_number):
        with tf.GradientTape() as tape:
            batch_loss = CLoss(model, model_bsm, batch_size)
        grads = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        print("Epoch: ",k+1, ". Batch ", j+1, " Loss: ", K.get_value(batch_loss),". Learning Rate: ",learning_rate)
        cost.append(math.log(K.get_value(batch_loss)))
        batch_loss = 0
    graph(model,model_bsm,k+1)
    model.save_weights('/Users/Jkzhang/Desktop/GitHub/DGM/SpreadPart64/003/weight' + str(k+1))
plt.plot(np.squeeze(cost))
plt.ylabel('cost')
plt.xlabel('Batches')
plt.show()   

#print(DGM-M)
