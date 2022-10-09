from typing import Dict

import numpy as np
from scipy.stats import norm
import tensorflow as tf

from .dgm_loss import DGMLoss


class BlackScholesLoss(DGMLoss):
    """"""
    def __init__(self):

        super().__init__()

    @staticmethod
    def interior_loss(batch: Dict,
                      model: tf.keras.Sequential):

        # Use AAD to get the numerical derivatives
        with tf.GradientTape() as tape2:
            tape2.watch(batch["S"])
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(batch["S"])
                tape.watch(batch["t"])

                inputs = tf.concat([batch["t"], batch["S"]], axis=1)
                y_hat_interior = model(inputs)

            grad_s = tape.gradient(y_hat_interior, batch["S"])
            grad_t = tape.gradient(y_hat_interior, batch["t"])
        hess_s = tape2.gradient(grad_s, batch["S"])

        # The PDE's interior
        return tf.square(
            grad_t +
            0.5 * batch["sigma"] ** 2 * batch["S"] ** 2 * hess_s +
            batch["r"] * batch["S"] * grad_s -
            batch["r"] * y_hat_interior
        )

    @staticmethod
    def initial_loss(batch: Dict,
                     model: tf.keras.Sequential):

        inputs_initial = tf.concat(
            [
                batch["T"] * tf.ones(shape=batch["t"].shape),
                batch["S"]
            ],
            axis=1
        )

        y_hat_initial = model(inputs_initial)

        return tf.square(y_hat_initial - tf.maximum(batch["S"] - batch["K"], 0))

    def loss(self,
             batch: Dict,
             model: tf.keras.Sequential) -> tf.Tensor:
        """

        """

        loss_interior = self.interior_loss(
            batch,
            model
        )

        loss_initial = self.initial_loss(
            batch,
            model
        )

        return tf.math.reduce_mean(loss_interior + loss_initial)

    def mae(self,
            batch: tf.Tensor,
            model: tf.keras.Sequential) -> tf.Tensor:

        d1 = (np.log(batch["S"]) + (batch["r"] + batch["sigma"] ** 2 / 2) * (batch["T"] - batch["t"])) \
             / (batch["sigma"] * np.sqrt(batch["T"] - batch["t"]))
        d2 = d1 - batch["sigma"] * np.sqrt(batch["T"] - batch["t"])

        y_true = batch["S"] * norm.cdf(d1) - \
                 batch["K"] * np.exp(-batch["r"] * (batch["T"] - batch["t"])) * norm.cdf(d2)

        inputs = tf.concat([batch["t"], batch["S"]], axis=1)
        y_hat = model(inputs)

        return tf.math.reduce_mean(tf.abs(y_true - y_hat))
