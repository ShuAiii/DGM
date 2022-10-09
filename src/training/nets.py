import pdb

import tensorflow as tf


class DGMNet(tf.keras.Model):
    def __init__(self, input_dim: int, width: int, depth: int):
        """"""
        super().__init__()

        self.n_layers = depth

        self.initial_layer = DenseLayer(input_dim, width, activation="swish")
        self.lstmlikelist = []
        for _ in range(self.n_layers):
            self.lstmlikelist.append(LSTMLikeLayer(input_dim, width, activation="tanh"))
        self.final_layer = DenseLayer(width, 1, None)

    def __call__(self, x):
        h = self.initial_layer(x)
        for i in range(self.n_layers):
            h = self.lstmlikelist[i](h, x)
        result = self.final_layer(h)

        return result


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs, activation):
        """"""
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        initializer_x = tf.initializers.GlorotUniform()
        initializer_z = tf.initializers.zeros()

        self.W = self.add_weight("W", shape=[self.n_inputs, self.n_outputs],
                                 initializer=initializer_x)
        self.b = self.add_weight("b", shape=[1, self.n_outputs], initializer=initializer_z)
        self.activation = _get_activation(activation)

    def __call__(self, inputs):
        return self.activation(tf.add(tf.matmul(inputs, self.W), self.b))


class LSTMLikeLayer(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs, activation):
        """"""
        super().__init__()

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs

        initializer_x = tf.initializers.GlorotUniform()
        initializer_z = tf.initializers.zeros()

        self.Uz = self.add_weight("Uz", shape=[self.n_inputs, self.n_outputs],

                                  initializer=initializer_x)
        self.Ug = self.add_weight("Ug", shape=[self.n_inputs, self.n_outputs],
                                  initializer=initializer_x)
        self.Ur = self.add_weight("Ur", shape=[self.n_inputs, self.n_outputs],
                                  initializer=initializer_x)
        self.Uh = self.add_weight("Uh", shape=[self.n_inputs, self.n_outputs],
                                  initializer=initializer_x)
        self.Wz = self.add_weight("Wz", shape=[self.n_outputs, self.n_outputs],
                                  initializer=initializer_x)
        self.Wg = self.add_weight("Wg", shape=[self.n_outputs, self.n_outputs],
                                  initializer=initializer_x)
        self.Wr = self.add_weight("Wr", shape=[self.n_outputs, self.n_outputs],
                                  initializer=initializer_x)
        self.Wh = self.add_weight("Wh", shape=[self.n_outputs, self.n_outputs],
                                  initializer=initializer_x)
        self.bz = self.add_weight("bz", shape=[1, self.n_outputs], initializer=initializer_z)
        self.bg = self.add_weight("bg", shape=[1, self.n_outputs], initializer=initializer_z)
        self.br = self.add_weight("br", shape=[1, self.n_outputs], initializer=initializer_z)
        self.bh = self.add_weight("bh", shape=[1, self.n_outputs], initializer=initializer_z)

        self.activation = _get_activation(activation)
        self.activation1 = tf.nn.swish

    def __call__(self, S, X):
        Z = self.activation(tf.add(tf.add(tf.matmul(X, self.Uz), tf.matmul(S, self.Wz)), self.bz))
        G = self.activation(tf.add(tf.add(tf.matmul(X, self.Ug), tf.matmul(S, self.Wg)), self.bg))
        R = self.activation(tf.add(tf.add(tf.matmul(X, self.Ur), tf.matmul(S, self.Wr)), self.br))
        H = self.activation(tf.add(tf.add(tf.matmul(X, self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))
        return tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z, S))


def _get_activation(name: str = None):
    if name == "tanh":
        func = tf.nn.tanh
    elif name == "sigmoid":
        func = tf.nn.sigmoid
    elif name == "swish":
        func = tf.nn.swish
    elif not name:
        func = tf.identity
    else:
        raise ValueError(f"{name} is not a supported activation!")
    return func


def get_model(network_kwargs: dict) -> tf.keras.Model:
    model = network_kwargs.pop("net")
    return DGMNet(**network_kwargs)


