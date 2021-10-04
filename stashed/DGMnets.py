"""

@author: By Adolfo Correia
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

        self.initial_layer = DenseLayer(dimensions + 1, n_nodes, activation="tanh")
        self.lstmlikelist = []
        for _ in range(self.n_layers):
            self.lstmlikelist.append(LSTMLikeLayer(dimensions + 1, n_nodes, activation="tanh"))
        self.final_layer = DenseLayer(n_nodes, 1, activation=None)


    def call(self, X):
        
        S = self.initial_layer.call(X)
        for i in range(self.n_layers):
            S = self.lstmlikelist[i].call({'S': S, 'X': X})
        result = self.final_layer.call(S)

        return result
    


# Neural network layers

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs, activation):
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
        self.activation = _get_function(activation)
    
    
    def call(self, inputs):
        S = tf.add(tf.matmul(inputs, self.W), self.b)
        S = self.activation(S)

        return S



class LSTMLikeLayer(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs, activation):
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

        self.Uz = self.add_weight("Uz", shape=[self.n_inputs, self.n_outputs],
                                    initializer = initializer_X)
        self.Ug = self.add_weight("Ug", shape=[self.n_inputs, self.n_outputs],
                                    initializer = initializer_X)
        self.Ur = self.add_weight("Ur", shape=[self.n_inputs, self.n_outputs],
                                    initializer = initializer_X)
        self.Uh = self.add_weight("Uh", shape=[self.n_inputs, self.n_outputs],
                                    initializer = initializer_X)
        self.Wz = self.add_weight("Wz", shape=[self.n_outputs, self.n_outputs],
                                    initializer = initializer_X)
        self.Wg = self.add_weight("Wg", shape=[self.n_outputs, self.n_outputs],
                                    initializer = initializer_X)
        self.Wr = self.add_weight("Wr", shape=[self.n_outputs, self.n_outputs],
                                    initializer = initializer_X)
        self.Wh = self.add_weight("Wh", shape=[self.n_outputs, self.n_outputs],
                                    initializer = initializer_X)
        self.bz = self.add_weight("bz", shape=[1, self.n_outputs], initializer = initializer_Z)
        self.bg = self.add_weight("bg", shape=[1, self.n_outputs], initializer = initializer_Z)
        self.br = self.add_weight("br", shape=[1, self.n_outputs], initializer = initializer_Z)
        self.bh = self.add_weight("bh", shape=[1, self.n_outputs], initializer = initializer_Z)

        self.activation = _get_function(activation)
        self.activation1 = tf.nn.softplus

    
    def call(self, inputs):
        S = inputs['S']
        X = inputs['X']

        Z = self.activation(tf.add(tf.add(tf.matmul(X, self.Uz), tf.matmul(S, self.Wz)), self.bz))
        G = self.activation(tf.add(tf.add(tf.matmul(X, self.Ug), tf.matmul(S, self.Wg)), self.bg))
        R = self.activation(tf.add(tf.add(tf.matmul(X, self.Ur), tf.matmul(S, self.Wr)), self.br))
        H = self.activation(tf.add(tf.add(tf.matmul(X, self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))
        Snew = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z, S))

        return Snew



def _get_function(name):
    f = None
    if name == "tanh":
        f = tf.nn.tanh
    elif name == "sigmoid":
        f = tf.nn.sigmoid
    elif name == "swish":
        f = tf.nn.swish
    elif not name:
        f = tf.identity
    
    assert f is not None
    
    return f