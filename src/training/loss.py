"""
"""
import abc

import tensorflow as tf


class DGMLoss(abc.ABC, metaclass=abc.ABCMeta):

    @staticmethod
    def interior_loss(*args) -> tf.Tensor:
        return tf.zeros([])

    @staticmethod
    def initial_loss(*args) -> tf.Tensor:
        raise tf.zeros([])

    @staticmethod
    def boundary_loss(*args) -> tf.Tensor:
        raise tf.zeros([])

    @staticmethod
    def asymptotic_loss(*args) -> tf.Tensor:
        raise tf.zeros([])

    @abc.abstractmethod
    def loss(self, input_batch: tf.Tensor,
             model: tf.keras.Sequential) -> tf.Tensor:
        raise NotImplementedError("ERROR: `loss` method must be implemented.")

    def __call__(self, input_batch: tf.Tensor,
                 model: tf.keras.Sequential) -> tf.Tensor:
        return self.loss(input_batch, model)


class BlackScholesLoss(DGMLoss):
    """"""
    def __init__(self, params: dict):
        self.maturity = params["T"]
        super().__init__()

    @staticmethod
    def interior_loss(inputs: tf.Tensor, y_hat: tf.Tensor,
                      grad: tf.Tensor, hess: tf.Tensor):
        return tf.square(grad[:, 0] + 0.5 * inputs[:, 3] ** 2 * hess[:, 1] + \
                         inputs[:, 2] * grad[:, 1] -
                         tf.math.multiply(inputs[:, 2], y_hat))

    @staticmethod
    def initial_loss(inputs: tf.Tensor, y_hat: tf.Tensor):
        return tf.square(y_hat - tf.maximum(inputs[:, 1] - 1, 0))

    def loss(self,
             input_batch: tf.Tensor, model: tf.keras.Sequential) -> tf.Tensor:
        """"""
        input_batch = tf.Variable(input_batch)
        with tf.GradientTape() as tape2:
            tape2.watch(input_batch)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(input_batch)
                y_hat_interior = model(input_batch)
            grad = tape.gradient(y_hat_interior, input_batch)
        hess = tape2.gradient(grad, input_batch)

        loss_interior = BlackScholesLoss.interior_loss(
            input_batch, y_hat_interior, grad, hess
        )

        inputs_initial = tf.Variable(
            tf.concat(
                [self.maturity * tf.ones(shape=[input_batch.shape[0], 1],
                          dtype=tf.dtypes.float32),
                 input_batch[:, 1:]],
                axis=1
            )
        )

        y_hat_initial = tf.reshape(model(inputs_initial), [-1])

        loss_initial = BlackScholesLoss.initial_loss(
            inputs_initial, y_hat_initial
        )

        return tf.math.reduce_mean(loss_interior + loss_initial)


LOSS_FUNCS = {
    "bsm_call": BlackScholesLoss
}


def get_loss(func, params: dict):
    """"""
    loss_func = LOSS_FUNCS.get(func)
    if loss_func:
        return loss_func(params)
    else:
        raise NotImplementedError(f"{func} not implemented")
