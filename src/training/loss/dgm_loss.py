"""
"""
import abc

import tensorflow as tf


class DGMLoss(abc.ABC, metaclass=abc.ABCMeta):

    @staticmethod
    def interior_loss(*args) -> tf.Tensor:
        pass

    @staticmethod
    def initial_loss(*args) -> tf.Tensor:
        pass

    @staticmethod
    def boundary_loss(*args) -> tf.Tensor:
        pass

    @staticmethod
    def asymptotic_loss(*args) -> tf.Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def loss(batch: tf.Tensor,
             model: tf.keras.Sequential) -> tf.Tensor:

        raise NotImplementedError("ERROR: `loss` method must be implemented.")

    @staticmethod
    @abc.abstractmethod
    def mae(batch: tf.Tensor,
            model: tf.keras.Sequential) -> tf.Tensor:

        raise NotImplementedError("ERROR: `mae` method must be implemented.")

    def __call__(self, input_batch: tf.Tensor,
                 model: tf.keras.Sequential) -> tf.Tensor:
        return self.loss(input_batch, model)
