from typing import Dict

import numpy as np
from scipy.stats import norm
import tensorflow as tf

from .dgm_loss import DGMLoss


class SpreadLoss(DGMLoss):
    """"""
    def __init__(self):

        super().__init__()

    @staticmethod
    def interior_loss(batch: Dict,
                      model: tf.keras.Sequential):

        # Use AAD to get the numerical derivatives
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(batch["S1"])
            tape2.watch(batch["S2"])
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(batch["S1"])
                tape.watch(batch["S2"])
                tape.watch(batch["t"])

                inputs = tf.concat([batch["t"], batch["S1"], batch["S2"]], axis=1)
                y_hat_interior = model(inputs)

            grad_s1 = tape.gradient(y_hat_interior, batch["S1"])
            grad_s2 = tape.gradient(y_hat_interior, batch["S2"])
            grad_t = tape.gradient(y_hat_interior, batch["t"])
        hess_s1_s1 = tape2.gradient(grad_s1, batch["S1"])
        hess_s2_s2 = tape2.gradient(grad_s2, batch["S2"])
        hess_s1_s2 = (tape2.gradient(grad_s1, batch["S2"]) + tape2.gradient(grad_s2, batch["S1"])) / 2

        # The PDE's interior
        v11 = 0.5 * batch["sigma1"] ** 2 * batch["S1"] ** 2 * hess_s1_s1
        v22 = 0.5 * batch["sigma2"] ** 2 * batch["S2"] ** 2 * hess_s2_s2
        v12 = batch["rho"] * batch["sigma1"] * batch["sigma2"] * batch["S1"] * batch["S2"] * hess_s1_s2

        return tf.square(
            grad_t +
            v11 + v22 + v12 +
            batch["r"] * batch["S1"] * grad_s1 +
            batch["r"] * batch["S2"] * grad_s2 -
            batch["r"] * y_hat_interior
        )

    @staticmethod
    def initial_loss(batch: Dict,
                     model: tf.keras.Sequential):

        inputs_initial = tf.concat(
            [
                batch["T"] * tf.ones(shape=batch["t"].shape),
                batch["S1"],
                batch["S2"]
            ],
            axis=1
        )

        y_hat_initial = model(inputs_initial)

        return tf.square(y_hat_initial - tf.maximum(batch["S1"] - batch["S2"] - batch["K"], 0))

    @staticmethod
    def loss(batch: Dict,
             model: tf.keras.Sequential) -> tf.Tensor:
        """

        """

        loss_interior = SpreadLoss.interior_loss(
            batch,
            model
        )

        loss_initial = SpreadLoss.initial_loss(
            batch,
            model
        )

        return tf.math.reduce_mean(loss_interior + loss_initial)

    @staticmethod
    def mae(batch: tf.Tensor,
            model: tf.keras.Sequential) -> tf.Tensor:

        return tf.zeros(1)
