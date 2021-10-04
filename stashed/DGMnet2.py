"""
Created on Tue Jun 23 11:17:22 2020

@author: Kevin Shuai Zhang
"""


"""

@Inspired by : Justin Sirignano and Adolfo Correia
"""



import tensorflow as tf



class DGMNet(tf.keras.Model):
    def __init__(self, n_layers, n_nodes, dimensions=1):
        """
        Parameters:
            - n_layers:     number of layers
            - n_nodes:      number of nodes in (inner) layers
            - dimensions:   number of spacial dimensions
        """
        super().__init__()
        
        self.n_layers = n_layers

        self.initial_layer = DenseLayer(dimensions + 1, n_nodes)
        self.lstmlikelist = []
        for _ in range(self.n_layers):
            self.lstmlikelist.append(LSTMLikeLayer(dimensions + 1, n_nodes))
        self.final_layer = DenseLayer(n_nodes, 1)


    def call(self, X):
        
        H = self.initial_layer.call(X)
        for i in range(self.n_layers):
            H = self.lstmlikelist[i].call({'H': H, 'X': X})
        result = self.final_layer.call(H)

        return result
    


# Neural network layers

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs):
        """
        Parameters:
            - n_inputs:     number of inputs
            - n_outputs:    number of outputs
            - activation:   activation function
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        initializer_X = tf.initializers.GlorotUniform()
        initializer_Z = tf.initializers.zeros()
        
        self.W = self.add_weight("W", shape=[self.n_inputs, self.n_outputs],
                                   initializer=initializer_X)
        self.b = self.add_weight("b", shape=[1, self.n_outputs], initializer=initializer_Z)
    
    
    def call(self, inputs):
        H = tf.nn.swish(tf.add(tf.matmul(inputs, self.W), self.b))
        return H



class LSTMLikeLayer(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs):
        """
        Parameters:
            - n_inputs:     number of inputs
            - n_outputs:    number of outputs
            - activation:   activation function
        """
        super().__init__()

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        
        initializer_X = tf.initializers.GlorotUniform()
        initializer_Z = tf.initializers.zeros()
        
        # Forget Gate
        self.Wf = self.add_weight("Wf", shape=[self.n_inputs, self.n_outputs],
                                    initializer = initializer_X)
        self.Uf = self.add_weight("Uf", shape=[self.n_outputs, self.n_outputs],
                                    initializer = initializer_X)
        self.bf = self.add_weight("bf", shape=[1, self.n_outputs], initializer = initializer_Z)
        
        
        # Update Gate
        self.Wu = self.add_weight("Wu", shape=[self.n_inputs, self.n_outputs],
                                    initializer = initializer_X)
        self.Uu = self.add_weight("Uu", shape=[self.n_outputs, self.n_outputs],
                                    initializer = initializer_X)
        self.bu = self.add_weight("bu", shape=[1, self.n_outputs], initializer = initializer_Z)
        
        # Output Gate
        
        self.Wo1 = self.add_weight("Wo1", shape=[self.n_inputs, self.n_outputs],
                                    initializer = initializer_X)
        self.Uo1 = self.add_weight("Uo1", shape=[self.n_outputs, self.n_outputs],
                                    initializer = initializer_X)
        self.bo1 = self.add_weight("bo1", shape=[1, self.n_outputs], initializer = initializer_Z)
        
        self.Wo2 = self.add_weight("Wo2", shape=[self.n_outputs, self.n_outputs],
                                    initializer = initializer_X)
        self.bo2 = self.add_weight("bo2", shape=[1, self.n_outputs], initializer = initializer_Z)

    
    def call(self, inputs):
        H = inputs['H']
        X = inputs['X']
        
        F = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(X, self.Wf), tf.matmul(H, self.Uf)), self.bf))
        U = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(X, self.Wu), tf.matmul(H, self.Uu)), self.bu))
        O1 = tf.nn.tanh(tf.add(tf.add(tf.matmul(X, self.Wo1), tf.matmul(H, self.Uo1)), self.bo1))
        O2 = tf.nn.swish(tf.add(tf.matmul(U*O1,self.Wo2), self.bo2))
        
        Hnew = tf.add(tf.multiply(F, H), O2)

        return Hnew